import os
import json
import random
from typing import Dict, Any, List, Optional

# Determine base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MEAL_DATA_PATH = os.path.join(BASE_DIR, "mock-data", "MealMockData.Json")
USER_DATA_PATH = os.path.join(BASE_DIR, "mock-data", "MockUser.json")

def load_meal_data() -> List[Dict[str, Any]]:
    if not os.path.exists(MEAL_DATA_PATH):
        raise FileNotFoundError(f"Meal mock data not found at {MEAL_DATA_PATH}")
    with open(MEAL_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def load_user_data() -> List[Dict[str, Any]]:
    if not os.path.exists(USER_DATA_PATH):
        raise FileNotFoundError(f"User mock data not found at {USER_DATA_PATH}")
    with open(USER_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def calculate_bmi(weight_kg: float, height_cm: float) -> Dict[str, Any]:
    """
    Calculate Body Mass Index (BMI).
    Formula: BMI = weight_kg / (height_m ^ 2)
    """
    if height_cm <= 0 or weight_kg <= 0:
        return {"error": "Weight and height must be positive values."}
    
    height_m = height_cm / 100.0
    bmi = round(weight_kg / (height_m ** 2), 2)
    
    if bmi < 18.5:
        category = "Thiếu cân (Underweight)"
    elif bmi < 24.9:
        category = "Bình thường (Normal)"
    elif bmi < 29.9:
        category = "Thừa cân (Overweight)"
    else:
        category = "Béo phì (Obese)"
        
    return {
        "bmi": bmi,
        "category": category
    }

def calculate_bmr(gender: str, weight_kg: float, height_cm: float, age: int) -> float:
    """
    Calculate Basal Metabolic Rate (BMR) using Mifflin-St Jeor Equation.
    Nam: BMR = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    Nữ: BMR = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    """
    gender_norm = gender.strip().lower()
    if gender_norm in ["nam", "male", "m"]:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    return round(bmr, 2)

def calculate_tdee(gender: str, weight_kg: float, height_cm: float, age: int, activity_level: str, goal: str) -> Dict[str, Any]:
    """
    Calculate Total Daily Energy Expenditure (TDEE) and recommended target macros.
    """
    bmr = calculate_bmr(gender, weight_kg, height_cm, age)
    
    # Activity factor mapping
    activity_factors = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "very_active": 1.9
    }
    
    act_level = activity_level.strip().lower()
    factor = activity_factors.get(act_level, 1.2) # default to sedentary
    tdee = bmr * factor
    
    # Goal adjustment
    goal_norm = goal.strip().lower()
    if goal_norm in ["lose_weight", "giam_can", "giảm cân"]:
        target_calories = tdee - 400
        protein_multiplier = 2.0
    elif goal_norm in ["gain_weight", "tang_can", "tăng cân"]:
        target_calories = tdee + 400
        protein_multiplier = 1.8
    elif goal_norm in ["build_muscle", "tang_co", "tăng cơ"]:
        target_calories = tdee + 200
        protein_multiplier = 2.2
    else:
        target_calories = tdee
        protein_multiplier = 1.5

    target_calories = max(1200.0, round(target_calories, 2)) # ensure healthy floor
    
    # Calculate Macro breakdown
    # Protein: multiplier * weight
    protein_g = round(protein_multiplier * weight_kg, 2)
    protein_kcal = protein_g * 4
    
    # Fat: 25% of target calories
    fat_kcal = target_calories * 0.25
    fat_g = round(fat_kcal / 9.0, 2)
    
    # Carb: remaining calories
    carb_kcal = target_calories - protein_kcal - fat_kcal
    carb_g = round(max(0.0, carb_kcal / 4.0), 2)
    
    fiber_g = 30.0
    water_ml = round(35.0 * weight_kg, 2)
    
    return {
        "bmr": round(bmr, 2),
        "tdee": round(tdee, 2),
        "target_calories_kcal": round(target_calories, 2),
        "target_protein_g": protein_g,
        "target_carbohydrates_g": carb_g,
        "target_fat_g": fat_g,
        "target_fiber_g": fiber_g,
        "target_water_ml": water_ml
    }

def search_food(food_name: str) -> List[Dict[str, Any]]:
    """
    Search for food items in the database by name (case-insensitive substring match).
    """
    meals = load_meal_data()
    query = food_name.strip().lower()
    results = []
    for m in meals:
        if query in m["name"].lower():
            results.append(m)
    return results

def generate_meal_plan(target_calories: float, target_protein: float, target_carb: float, target_fat: float, exclude_foods: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Generate a meal plan matching target calories and macros within tolerances.
    Excludes any foods specified in exclude_foods.
    """
    meals = load_meal_data()
    
    # Normalize exclude list
    exclude_list = [x.strip().lower() for x in (exclude_foods or [])]
    
    # Filter meals
    filtered_meals = []
    for m in meals:
        excluded = False
        for ex in exclude_list:
            if ex in m["name"].lower():
                excluded = True
                break
        if not excluded:
            filtered_meals.append(m)
            
    if not filtered_meals:
        return {"error": "Không tìm thấy món ăn nào phù hợp sau khi lọc chất dị ứng/loại trừ."}

    # Algorithm: Random search to find a combination of 3 to 4 meals that matches target calories within +/- 10%
    best_plan = None
    best_diff = float("inf")
    
    # Try 2000 random combinations to get a good fit
    for _ in range(3000):
        # Pick 3 or 4 meals randomly
        num_meals = random.choice([3, 4])
        selected = random.sample(filtered_meals, min(num_meals, len(filtered_meals)))
        
        tot_cal = sum(m["calories_kcal"] for m in selected)
        tot_prot = sum(m["protein_g"] for m in selected)
        tot_carb = sum(m["carbohydrates_g"] for m in selected)
        tot_fat = sum(m["fat_g"] for m in selected)
        
        cal_diff = abs(tot_cal - target_calories)
        
        # If this is within +/- 10% of calories, check macros
        if tot_cal * 0.9 <= target_calories <= tot_cal * 1.1:
            score = cal_diff + abs(tot_prot - target_protein)*4 + abs(tot_carb - target_carb)*4 + abs(tot_fat - target_fat)*9
            if score < best_diff:
                best_diff = score
                best_plan = {
                    "meals": selected,
                    "total_calories_kcal": tot_cal,
                    "total_protein_g": tot_prot,
                    "total_carbohydrates_g": tot_carb,
                    "total_fat_g": tot_fat
                }
                
    if best_plan:
        return best_plan
        
    # If no plan fell within 10% range, return the closest one we found
    # Let's try to just build a simple greedy one if random search failed
    filtered_meals.sort(key=lambda x: x["calories_kcal"])
    selected = []
    tot_cal = 0
    for m in filtered_meals:
        if tot_cal + m["calories_kcal"] <= target_calories * 1.05:
            selected.append(m)
            tot_cal += m["calories_kcal"]
            if len(selected) >= 4:
                break
    
    tot_prot = sum(m["protein_g"] for m in selected)
    tot_carb = sum(m["carbohydrates_g"] for m in selected)
    tot_fat = sum(m["fat_g"] for m in selected)
    
    return {
        "meals": selected,
        "total_calories_kcal": tot_cal,
        "total_protein_g": tot_prot,
        "total_carbohydrates_g": tot_carb,
        "total_fat_g": tot_fat
    }

def validate_meal_plan(meal_plan: Dict[str, Any], target_calories: float, target_protein: float, target_carb: float, target_fat: float, exclude_foods: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Validate a meal plan's total calories and macros against targets and allergen/exclusion lists.
    - Calories tolerance: +/- 5%
    - Macro tolerances: +/- 10%
    """
    meals = meal_plan.get("meals", [])
    total_cal = sum(m["calories_kcal"] for m in meals)
    total_prot = sum(m["protein_g"] for m in meals)
    total_carb = sum(m["carbohydrates_g"] for m in meals)
    total_fat = sum(m["fat_g"] for m in meals)
    
    exclude_list = [x.strip().lower() for x in (exclude_foods or [])]
    
    violations = []
    
    # Calorie check (+/- 5%)
    cal_min, cal_max = target_calories * 0.95, target_calories * 1.05
    if not (cal_min <= total_cal <= cal_max):
        violations.append(f"Calories ({total_cal} kcal) nằm ngoài sai số 5% của mục tiêu ({target_calories} kcal). Khoảng cho phép: {round(cal_min, 1)} - {round(cal_max, 1)} kcal.")
        
    # Macros check (+/- 10%)
    prot_min, prot_max = target_protein * 0.90, target_protein * 1.10
    if not (prot_min <= total_prot <= prot_max):
        violations.append(f"Protein ({total_prot}g) nằm ngoài sai số 10% của mục tiêu ({target_protein}g). Khoảng cho phép: {round(prot_min, 1)} - {round(prot_max, 1)}g.")
        
    carb_min, carb_max = target_carb * 0.90, target_carb * 1.10
    if not (carb_min <= total_carb <= carb_max):
        violations.append(f"Carbohydrates ({total_carb}g) nằm ngoài sai số 10% của mục tiêu ({target_carb}g). Khoảng cho phép: {round(carb_min, 1)} - {round(carb_max, 1)}g.")
        
    fat_min, fat_max = target_fat * 0.90, target_fat * 1.10
    if not (fat_min <= total_fat <= fat_max):
        violations.append(f"Fat ({total_fat}g) nằm ngoài sai số 10% của mục tiêu ({target_fat}g). Khoảng cho phép: {round(fat_min, 1)} - {round(fat_max, 1)}g.")
        
    # Exclusion check
    excluded_found = []
    for m in meals:
        for ex in exclude_list:
            if ex in m["name"].lower():
                excluded_found.append(m["name"])
                break
                
    if excluded_found:
        violations.append(f"Thực đơn chứa thực phẩm cấm/dị ứng: {', '.join(set(excluded_found))}.")
        
    status = "PASS" if not violations else "FAIL"
    
    return {
        "status": status,
        "total_calories_kcal": total_cal,
        "total_protein_g": total_prot,
        "total_carbohydrates_g": total_carb,
        "total_fat_g": total_fat,
        "violations": violations
    }

def replace_food(food_name: str, target_calories: float, target_protein: float, target_carb: float, target_fat: float, exclude_foods: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Suggest alternative foods from the database that can replace a specific food,
    matching similar calories and macros while avoiding excluded/allergic items.
    """
    meals = load_meal_data()
    exclude_list = [x.strip().lower() for x in (exclude_foods or [])]
    
    # Try to find the target food profile first (case-insensitive substring)
    target_profile = None
    for m in meals:
        if food_name.strip().lower() in m["name"].lower():
            target_profile = m
            break
            
    if not target_profile:
        return {"error": f"Không tìm thấy thông tin món ăn '{food_name}' trong cơ sở dữ liệu."}
        
    # Search for alternative foods that match target_profile's calorie and protein count closest
    # while excluding the food itself and allergen list
    alternatives = []
    for m in meals:
        if target_profile["id"] == m["id"] or food_name.strip().lower() in m["name"].lower():
            continue
            
        # check allergens
        is_excluded = False
        for ex in exclude_list:
            if ex in m["name"].lower():
                is_excluded = True
                break
        if is_excluded:
            continue
            
        # compute similarity score (lower is closer)
        score = abs(m["calories_kcal"] - target_profile["calories_kcal"]) + \
                abs(m["protein_g"] - target_profile["protein_g"]) * 4 + \
                abs(m["carbohydrates_g"] - target_profile["carbohydrates_g"]) * 4 + \
                abs(m["fat_g"] - target_profile["fat_g"]) * 9
        alternatives.append((score, m))
        
    if not alternatives:
        return {"error": "Không tìm thấy món ăn thay thế nào phù hợp."}
        
    # Sort by score and pick top 3
    alternatives.sort(key=lambda x: x[0])
    return {
        "original_food": target_profile,
        "replacements": [x[1] for x in alternatives[:3]]
    }

def get_user_profile(user_id_or_name: str) -> Dict[str, Any]:
    """
    Look up a user profile in the MockUser.json database by ID or name (case-insensitive substring).
    """
    try:
        users = load_user_data()
        query = user_id_or_name.strip().lower()
        for u in users:
            if query == u["id"].lower() or query in u["name"].lower():
                return u
        return {"error": f"Không tìm thấy hồ sơ người dùng cho '{user_id_or_name}'."}
    except Exception as e:
        return {"error": f"Error loading user profile: {str(e)}"}

