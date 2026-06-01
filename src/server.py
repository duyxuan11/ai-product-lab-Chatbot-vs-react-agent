import os
import sys
import json
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

from src.core.gemini_provider import GeminiProvider
from src.agent.agent import ReActAgent
from tools.db_utils import load_users, save_users, load_meals, load_chat_history, save_chat_history
from tools.tdee_calculator import calculate_tdee
from tools.meal_logger import log_meal
from tools.summary_viewer import get_daily_summary
from tools.menu_recommendation import recommend_daily_menu

# Load env variables
load_dotenv()

app = FastAPI(title="AI Nutrition Agent API", version="1.0.0")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Gemini Provider
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("WARNING: GEMINI_API_KEY not found in .env file! Agent features will fail.")

# Pydantic models for request bodies
class UserProfileRequest(BaseModel):
    id: Optional[str] = None
    name: str
    age: int
    gender: str
    weight_kg: float
    height_cm: float
    activity_level: str
    goal: str

class LogMealRequest(BaseModel):
    meal_type: str  # Breakfast, Lunch, Dinner, Snack
    dish_name: str
    portion_size: float = 1.0

class ChatRequest(BaseModel):
    user_id: Optional[str] = None
    message: str

@app.get("/")
def read_root():
    return {"status": "ok", "message": "AI Nutrition Agent API is running."}

@app.get("/api/users")
def get_users():
    """Retrieve all mock users."""
    return load_users()

@app.post("/api/users")
def create_or_update_user(profile: UserProfileRequest):
    """Creates a new user profile or updates an existing one, auto-calculating TDEE targets."""
    users = load_users()
    
    # Calculate targets based on stats
    tdee_json_str = calculate_tdee(
        weight_kg=profile.weight_kg,
        height_cm=profile.height_cm,
        age=profile.age,
        gender=profile.gender,
        activity_level=profile.activity_level,
        goal=profile.goal
    )
    tdee_data = json.loads(tdee_json_str)
    if "error" in tdee_data:
        raise HTTPException(status_code=400, detail=tdee_data["error"])
        
    user_id = profile.id if profile.id else f"user_{len(users) + 1}"
    
    # Find existing or create new
    user = next((u for u in users if u["id"] == user_id), None)
    if user:
        # Update profile stats
        user["name"] = profile.name
        user["age"] = profile.age
        user["gender"] = profile.gender
        user["weight_kg"] = profile.weight_kg
        user["height_cm"] = profile.height_cm
        user["activity_level"] = profile.activity_level
        user["goal"] = profile.goal
        user["target_calories"] = tdee_data["target_calories"]
        user["target_protein_g"] = tdee_data["target_protein_g"]
        user["target_carbs_g"] = tdee_data["target_carbs_g"]
        user["target_fat_g"] = tdee_data["target_fat_g"]
    else:
        # Create new
        user = {
            "id": user_id,
            "name": profile.name,
            "age": profile.age,
            "gender": profile.gender,
            "weight_kg": profile.weight_kg,
            "height_cm": profile.height_cm,
            "activity_level": profile.activity_level,
            "goal": profile.goal,
            "target_calories": tdee_data["target_calories"],
            "target_protein_g": tdee_data["target_protein_g"],
            "target_carbs_g": tdee_data["target_carbs_g"],
            "target_fat_g": tdee_data["target_fat_g"],
            "logged_meals": []
        }
        users.append(user)
        
    save_users(users)
    return user

@app.get("/api/users/{user_id}/summary")
def get_user_summary(user_id: str):
    """Get nutrition summary for a user today."""
    summary_str = get_daily_summary(user_id)
    summary_data = json.loads(summary_str)
    if "error" in summary_data:
        raise HTTPException(status_code=404, detail=summary_data["error"])
    return summary_data

@app.get("/api/users/{user_id}/recommend_menu")
def get_user_menu_recommendation(
    user_id: str,
    preferred_dishes: Optional[str] = None,
    allergies: Optional[str] = None
):
    """Generate a recommended menu for the user based on their goals, preferred dishes, and allergies."""
    pref_list = [d.strip() for d in preferred_dishes.split(",")] if preferred_dishes else None
    allergy_list = [a.strip() for a in allergies.split(",")] if allergies else None
    
    menu_str = recommend_daily_menu(user_id, preferred_dishes=pref_list, allergies=allergy_list)
    menu_data = json.loads(menu_str)
    if "error" in menu_data:
        raise HTTPException(status_code=400, detail=menu_data["error"])
    return menu_data

@app.post("/api/users/{user_id}/log")
def user_log_meal(user_id: str, request: LogMealRequest):
    """Log a meal for a user."""
    log_result_str = log_meal(
        user_id=user_id,
        meal_type=request.meal_type,
        dish_name=request.dish_name,
        portion_size=request.portion_size
    )
    log_result = json.loads(log_result_str)
    if "error" in log_result:
        raise HTTPException(status_code=400, detail=log_result["error"])
    return log_result

@app.get("/api/dishes")
def get_dishes(q: Optional[str] = None):
    """Retrieve or search the mock dish database."""
    meals = load_meals()
    if not q:
        return meals
    query = q.lower().strip()
    return [m for m in meals if query in m["name"].lower()]

@app.post("/api/chat")
def chat_with_agent(request: ChatRequest):
    """Interact with the ReAct Nutrition Agent."""
    if not api_key:
        raise HTTPException(status_code=500, detail="Gemini API Key is not configured in .env file.")
        
    # Retrieve user context to guide the agent
    user_context = None
    if request.user_id:
        users = load_users()
        user_context = next((u for u in users if u["id"] == request.user_id), None)

    try:
        # Load conversation history for context
        history = []
        if request.user_id:
            history = load_chat_history(request.user_id)

        # Initialize ReAct Agent
        provider = GeminiProvider(model_name="gemini-3.5-flash", api_key=api_key)
        agent = ReActAgent(llm=provider)
        
        # Run agent with history context
        result = agent.run(user_input=request.message, user_context=user_context, chat_history=history)
        
        # Save new messages to history
        if request.user_id:
            history.append({"sender": "user", "text": request.message, "trace": None})
            history.append({
                "sender": "assistant",
                "text": result["final_answer"],
                "trace": result["history"]
            })
            save_chat_history(request.user_id, history)
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in Agent reasoning: {str(e)}")

@app.get("/api/chat/history/{user_id}")
def get_user_chat_history(user_id: str):
    """Retrieve chat history for a user. If none exists, return a default welcome message."""
    history = load_chat_history(user_id)
    if not history:
        users = load_users()
        user = next((u for u in users if u["id"] == user_id), None)
        name = user["name"] if user else "bạn"
        return [
            {
                "sender": "assistant",
                "text": f"Chào mừng {name} trở lại! Hôm nay bạn cần tôi hỗ trợ gì về dinh dưỡng hoặc lập kế hoạch ăn uống?",
                "trace": None
            }
        ]
    return history

@app.post("/api/chat/history/{user_id}/clear")
def clear_user_chat_history(user_id: str):
    """Clear chat history for a user."""
    save_chat_history(user_id, [])
    users = load_users()
    user = next((u for u in users if u["id"] == user_id), None)
    name = user["name"] if user else "bạn"
    return [
        {
            "sender": "assistant",
            "text": f"Chào mừng {name} trở lại! Hôm nay bạn cần tôi hỗ trợ gì về dinh dưỡng hoặc lập kế hoạch ăn uống?",
            "trace": None
        }
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
