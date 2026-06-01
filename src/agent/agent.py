import os
import re
import json
from typing import List, Dict, Any, Optional
from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger
from src.telemetry.metrics import tracker
from src.tools.nutrition_tools import (
    calculate_bmi,
    calculate_bmr,
    calculate_tdee,
    search_food,
    generate_meal_plan,
    validate_meal_plan,
    replace_food,
    get_user_profile
)

def parse_arguments(args_str: str) -> dict:
    """
    Robust parsing of key-value arguments, JSON, or colon-separated values.
    """
    args_str = args_str.strip()
    if not args_str:
        return {}
        
    # Try parsing as JSON first
    if args_str.startswith("{") and args_str.endswith("}"):
        try:
            return json.loads(args_str)
        except Exception:
            pass
            
    # Try keyword arguments (key=value, key2=value2...)
    kwargs = {}
    pattern = r"(\w+)\s*=\s*(['\"]?.*?['\"]?)(?=\s*,\s*\w+\s*=|$)"
    matches = re.findall(pattern, args_str)
    if matches:
        for key, val in matches:
            val = val.strip().strip("'").strip('"')
            try:
                if "." in val:
                    kwargs[key] = float(val)
                else:
                    kwargs[key] = int(val)
            except ValueError:
                kwargs[key] = val
        return kwargs

    # Try colon-separated key-value pairs (key: value)
    if ":" in args_str:
        try:
            pairs = args_str.split(",")
            for pair in pairs:
                if ":" in pair:
                    k, v = pair.split(":", 1)
                    kwargs[k.strip()] = v.strip().strip("'").strip('"')
            return kwargs
        except Exception:
            pass
            
    return {"raw": args_str}


class ReActAgent:
    """
    A ReAct-style Agent that follows the Thought-Action-Observation loop
    specifically tailored for nutritional planning and meal suggestions.
    """
    
    def __init__(self, llm: LLMProvider, tools: List[Dict[str, Any]], max_steps: int = 8):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.history = []

    def get_system_prompt(self) -> str:
        """
        System prompt that instructs the agent to follow ReAct for Nutrition.
        """
        tool_descriptions = "\n".join([f"- {t['name']}: {t['description']}" for t in self.tools])
        return f"""You are an expert AI Nutrition Planner Agent. You help users calculate their nutritional needs, look up food values, generate balanced meal plans, validate plans for allergies/diseases, and suggest healthy food replacements.

You have access to the following tools:
{tool_descriptions}

CRITICAL RULES:
1. Always calculate nutritional needs (BMI, BMR, TDEE) first before generating any meal plan if user details are provided.
2. When creating or validating meal plans, always check for specified allergies or excluded foods.
3. Keep the target calories and macros in mind. Ensure meal plans are validated before recommending them as final.
4. If a meal plan validation fails, use your reasoning to adjust the menu or replace offending items, and validate again.
5. Do not offer medical diagnosis. Emphasize that this is nutritional guidance.

Use the following exact format:
Thought: your line of reasoning about what step to take next.
Action: tool_name(arguments)
Observation: the result of the tool execution (supplied by the environment).
... (repeat Thought/Action/Observation if needed)
Final Answer: your final response to the user, explaining the nutrition values, meal choices, and validation results clearly.

Example Flow:
Thought: I need to calculate the user's BMI and TDEE first.
Action: calculate_tdee(gender="Nam", weight_kg=68.0, height_cm=175.0, age=24, activity_level="moderate", goal="build_muscle")
Observation: {{"bmr": 1648.75, "tdee": 2555.56, "target_calories_kcal": 2755.56, "target_protein_g": 149.6, "target_carbohydrates_g": 328.7, "target_fat_g": 76.54}}
Thought: Now I need to generate a meal plan matching these target values.
Action: generate_meal_plan(target_calories=2755.56, target_protein=149.6, target_carb=328.7, target_fat=76.54)
Observation: ...
Thought: I should validate the meal plan to make sure it matches tolerances.
Action: validate_meal_plan(meal_plan=..., target_calories=2755.56, target_protein=149.6, target_carb=328.7, target_fat=76.54)
Observation: {{"status": "PASS", ...}}
Thought: The validation passed. I can now present the final answer.
Final Answer: [Detailed response to the user]
"""

    def run(self, user_input: str) -> str:
        logger.log_event("AGENT_START", {"input": user_input, "model": self.llm.model_name})
        
        self.steps_history = []
        current_prompt = f"User: {user_input}"
        steps = 0
        
        while steps < self.max_steps:
            steps += 1
            logger.log_event("AGENT_STEP_START", {"step": steps})
            
            # Generate LLM response
            response_dict = self.llm.generate(
                current_prompt,
                system_prompt=self.get_system_prompt(),
                stop=["Observation:", "Observation: ", "\nObservation:"]
            )
            content = response_dict.get("content", "").strip()
            
            # Track LLM performance
            tracker.track_request(
                provider=response_dict.get("provider", "openai"),
                model=self.llm.model_name,
                usage=response_dict.get("usage", {}),
                latency_ms=response_dict.get("latency_ms", 0)
            )
            
            logger.log_event("LLM_RESPONSE", {"content": content})
            
            # Append LLM output to conversation context
            current_prompt += f"\n{content}"
            
            # Extract thought
            thought = content
            if "Action:" in content:
                thought = content.split("Action:")[0].strip()
            elif "Final Answer:" in content:
                thought = content.split("Final Answer:")[0].strip()
            
            # Check for Final Answer
            final_answer_match = re.search(r"Final\s+Answer:\s*(.*)", content, re.IGNORECASE | re.DOTALL)
            if final_answer_match:
                final_answer = final_answer_match.group(1).strip()
                self.steps_history.append({
                    "step": steps,
                    "thought": thought,
                    "action": None,
                    "observation": None,
                    "final_answer": final_answer
                })
                logger.log_event("AGENT_END", {"steps": steps, "status": "SUCCESS"})
                return final_answer
                
            # Parse Action
            action_match = re.search(r"Action:\s*(\w+)\((.*)\)", content, re.IGNORECASE)
            if action_match:
                tool_name = action_match.group(1).strip()
                args_str = action_match.group(2).strip()
                
                # Execute tool
                observation = self._execute_tool(tool_name, args_str)
                logger.log_event("TOOL_OBSERVATION", {"tool": tool_name, "observation": observation})
                
                # Append Observation to context
                current_prompt += f"\nObservation: {observation}"
                
                self.steps_history.append({
                    "step": steps,
                    "thought": thought,
                    "action": f"{tool_name}({args_str})",
                    "observation": observation,
                    "final_answer": None
                })
            else:
                # If LLM didn't output Action or Final Answer, treat the whole content as Final Answer
                # to prevent infinite loops, but log a warning.
                logger.log_event("AGENT_PARSING_WARNING", {"content": content})
                logger.log_event("AGENT_END", {"steps": steps, "status": "FALLBACK_SUCCESS"})
                self.steps_history.append({
                    "step": steps,
                    "thought": thought,
                    "action": None,
                    "observation": None,
                    "final_answer": content
                })
                return content
                
        logger.log_event("AGENT_END", {"steps": steps, "status": "TIMEOUT"})
        return "Xin lỗi, tôi không thể hoàn thành yêu cầu trong số bước cho phép. Vui lòng thử lại với yêu cầu chi tiết hơn."

    def _execute_tool(self, tool_name: str, args_str: str) -> str:
        """
        Helper method to execute tools by name.
        """
        try:
            args = parse_arguments(args_str)
            logger.log_event("TOOL_EXECUTION_START", {"tool": tool_name, "arguments": args})
            
            result = None
            if tool_name == "get_user_profile":
                uid = str(args.get("user_id_or_name", args.get("user_id", args.get("name", args.get("raw", "")))))
                result = get_user_profile(user_id_or_name=uid)
                
            elif tool_name == "calculate_bmi":
                w = float(args.get("weight_kg", args.get("weight", 0)))
                h = float(args.get("height_cm", args.get("height", 0)))
                result = calculate_bmi(weight_kg=w, height_cm=h)
                
            elif tool_name == "calculate_bmr":
                g = str(args.get("gender", "Nam"))
                w = float(args.get("weight_kg", args.get("weight", 0)))
                h = float(args.get("height_cm", args.get("height", 0)))
                a = int(args.get("age", 0))
                result = calculate_bmr(gender=g, weight_kg=w, height_cm=h, age=a)
                
            elif tool_name == "calculate_tdee":
                g = str(args.get("gender", "Nam"))
                w = float(args.get("weight_kg", args.get("weight", 0)))
                h = float(args.get("height_cm", args.get("height", 0)))
                a = int(args.get("age", 0))
                al = str(args.get("activity_level", "sedentary"))
                goal = str(args.get("goal", "maintain"))
                result = calculate_tdee(gender=g, weight_kg=w, height_cm=h, age=a, activity_level=al, goal=goal)
                
            elif tool_name == "search_food":
                fn = str(args.get("food_name", args.get("query", args.get("raw", ""))))
                result = search_food(food_name=fn)
                
            elif tool_name == "generate_meal_plan":
                cal = float(args.get("target_calories", args.get("target_calories_kcal", 0)))
                prot = float(args.get("target_protein", args.get("target_protein_g", 0)))
                carb = float(args.get("target_carb", args.get("target_carbohydrates_g", 0)))
                fat = float(args.get("target_fat", args.get("target_fat_g", 0)))
                exc = args.get("exclude_foods", args.get("exclude", None))
                if exc and isinstance(exc, str):
                    exc = [x.strip() for x in exc.split(",")]
                result = generate_meal_plan(target_calories=cal, target_protein=prot, target_carb=carb, target_fat=fat, exclude_foods=exc)
                
            elif tool_name == "validate_meal_plan":
                plan = args.get("meal_plan", args.get("plan", {}))
                if isinstance(plan, str):
                    try:
                        plan = json.loads(plan)
                    except Exception:
                        pass
                cal = float(args.get("target_calories", args.get("target_calories_kcal", 0)))
                prot = float(args.get("target_protein", args.get("target_protein_g", 0)))
                carb = float(args.get("target_carb", args.get("target_carbohydrates_g", 0)))
                fat = float(args.get("target_fat", args.get("target_fat_g", 0)))
                exc = args.get("exclude_foods", args.get("exclude", None))
                if exc and isinstance(exc, str):
                    exc = [x.strip() for x in exc.split(",")]
                result = validate_meal_plan(meal_plan=plan, target_calories=cal, target_protein=prot, target_carb=carb, target_fat=fat, exclude_foods=exc)
                
            elif tool_name == "replace_food":
                fn = str(args.get("food_name", args.get("food", "")))
                cal = float(args.get("target_calories", args.get("target_calories_kcal", 0)))
                prot = float(args.get("target_protein", args.get("target_protein_g", 0)))
                carb = float(args.get("target_carb", args.get("target_carbohydrates_g", 0)))
                fat = float(args.get("target_fat", args.get("target_fat_g", 0)))
                exc = args.get("exclude_foods", args.get("exclude", None))
                if exc and isinstance(exc, str):
                    exc = [x.strip() for x in exc.split(",")]
                result = replace_food(food_name=fn, target_calories=cal, target_protein=prot, target_carb=carb, target_fat=fat, exclude_foods=exc)
                
            else:
                result = {"error": f"Tool {tool_name} not found."}
                
            logger.log_event("TOOL_EXECUTION_END", {"tool": tool_name, "result": result})
            return json.dumps(result, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            logger.log_event("TOOL_EXECUTION_ERROR", {"tool": tool_name, "error": str(e)})
            return json.dumps({"error": f"Exception during execution: {str(e)}"}, ensure_ascii=False)

