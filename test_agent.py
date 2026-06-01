import os
import sys
import argparse
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.openai_provider import OpenAIProvider
from src.core.gemini_provider import GeminiProvider
from src.core.local_provider import LocalProvider
from src.agent.agent import ReActAgent
from chatbot import BaselineChatbot
from src.telemetry.logger import logger
from src.telemetry.metrics import tracker

# Define tool specs
TOOLS_SPEC = [
    {
        "name": "get_user_profile",
        "description": "get_user_profile(user_id_or_name: str) -> dict. Look up a user's details like age, gender, height, weight, activity level, and goals by their name or user ID."
    },
    {
        "name": "calculate_bmi",
        "description": "calculate_bmi(weight_kg: float, height_cm: float) -> dict. Computes the user's BMI and status."
    },
    {
        "name": "calculate_bmr",
        "description": "calculate_bmr(gender: str, weight_kg: float, height_cm: float, age: int) -> float. Computes Basal Metabolic Rate."
    },
    {
        "name": "calculate_tdee",
        "description": (
            "calculate_tdee(gender: str, weight_kg: float, height_cm: float, age: int, activity_level: str, goal: str) -> dict. "
            "Computes TDEE, adjusted daily calories, and recommended macro targets (protein, carbs, fat, fiber, water). "
            "activity_level options: sedentary, light, moderate, active, very_active. goal options: lose_weight, gain_weight, build_muscle, maintain."
        )
    },
    {
        "name": "search_food",
        "description": "search_food(food_name: str) -> list. Searches the database for matching food items and returns their calories and macros."
    },
    {
        "name": "generate_meal_plan",
        "description": (
            "generate_meal_plan(target_calories: float, target_protein: float, target_carb: float, target_fat: float, exclude_foods: list) -> dict. "
            "Generates a 1-day meal plan from the database matching the target calories and macros, excluding specified items."
        )
    },
    {
        "name": "validate_meal_plan",
        "description": (
            "validate_meal_plan(meal_plan: dict, target_calories: float, target_protein: float, target_carb: float, target_fat: float, exclude_foods: list) -> dict. "
            "Checks if a meal plan's total calories (within 5%), macros (within 10%), and allergens are acceptable. Returns status: PASS/FAIL."
        )
    },
    {
        "name": "replace_food",
        "description": (
            "replace_food(food_name: str, target_calories: float, target_protein: float, target_carb: float, target_fat: float, exclude_foods: list) -> dict. "
            "Finds alternative food items from the database to replace a specific food item while matching targets and exclusions."
        )
    }
]

TEST_SCENARIOS = [
    {
        "name": "1. User profile lookup, TDEE calculation & Muscle Building plan",
        "query": "Tôi là Nguyễn Văn A (user_1). Hãy tính nhu cầu dinh dưỡng của tôi và thiết kế cho tôi một thực đơn ăn uống trong ngày để tăng cơ."
    },
    {
        "name": "2. Weight loss plan with specific exclusion (Allergy)",
        "query": "Tôi là Trần Thị B (user_2). Hãy tính nhu cầu năng lượng của tôi và lập cho tôi một thực đơn giảm cân. Hãy chú ý tôi bị dị ứng với bún riêu cua."
    },
    {
        "name": "3. Dietary assessment of a daily food intake",
        "query": "Hôm nay tôi đã ăn: 1 tô Phở bò, 1 đĩa Cơm tấm sườn nướng, và 1 bát Chè chuối. Hãy tra cứu calo và đánh giá dinh dưỡng giúp tôi."
    }
]

def init_llm():
    provider_name = os.getenv("DEFAULT_PROVIDER", "openai").lower()
    model_name = os.getenv("DEFAULT_MODEL", "gpt-4o")
    
    if provider_name == "openai":
        return OpenAIProvider(model_name=model_name, api_key=os.getenv("OPENAI_API_KEY"))
    elif provider_name in ["gemini", "google"]:
        return GeminiProvider(model_name=model_name, api_key=os.getenv("GEMINI_API_KEY"))
    elif provider_name == "local":
        model_path = os.getenv("LOCAL_MODEL_PATH", "./models/Phi-3-mini-4k-instruct-q4.gguf")
        return LocalProvider(model_path=model_path)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")

def run_tests():
    parser = argparse.ArgumentParser(description="Run Nutrition Agent Evaluation Suite")
    parser.add_argument("--mode", choices=["agent", "chatbot", "both"], default="both", help="Which model mode to evaluate")
    args = parser.parse_args()
    
    llm = init_llm()
    agent = ReActAgent(llm=llm, tools=TOOLS_SPEC, max_steps=8)
    chatbot = BaselineChatbot()
    
    print("=" * 60)
    print(f"Starting Nutrition Agent Evaluation Suite")
    print(f"LLM Provider: {llm.model_name}")
    print("=" * 60 + "\n")
    
    for i, scenario in enumerate(TEST_SCENARIOS):
        print(f"[Scenario] {scenario['name']}")
        print(f"Query: \"{scenario['query']}\"")
        print("-" * 40)
        
        # 1. Run ReAct Agent
        if args.mode in ["agent", "both"]:
            print("[Running ReAct Agent...]")
            # Reset tracker metrics for this run
            tracker.session_metrics = []
            agent_res = agent.run(scenario["query"])
            print(f"\nReAct Agent Final Response:\n{agent_res}")
            
            # Print stats
            if tracker.session_metrics:
                tot_latency = sum(m["latency_ms"] for m in tracker.session_metrics)
                tot_tokens = sum(m["total_tokens"] for m in tracker.session_metrics)
                tot_cost = sum(m["cost_estimate"] for m in tracker.session_metrics)
                steps = len(tracker.session_metrics)
                print(f"\n[Agent Stats] Steps: {steps} | Total Latency: {tot_latency}ms | Total Tokens: {tot_tokens} | Total Cost: ${tot_cost:.6f}")
            print("-" * 40)
            
        # 2. Run Baseline Chatbot
        if args.mode in ["chatbot", "both"]:
            print("[Running Baseline Chatbot...]")
            tracker.session_metrics = []
            chatbot_res = chatbot.run(scenario["query"])
            print(f"\nBaseline Chatbot Response:\n{chatbot_res}")
            
            # Print stats
            if tracker.session_metrics:
                m = tracker.session_metrics[0]
                print(f"\n[Chatbot Stats] Total Latency: {m['latency_ms']}ms | Total Tokens: {m['total_tokens']} | Total Cost: ${m['cost_estimate']:.6f}")
            print("-" * 40)
            
        import time
        print("Cooling down for 12 seconds to respect Gemini API rate limits...")
        time.sleep(12)
        print("=" * 60 + "\n")

if __name__ == "__main__":
    run_tests()
