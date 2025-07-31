# Personalized_Diet_Recommendation_System.py
import streamlit as st
import pandas as pd

# Calculate BMR using Mifflin-St Jeor Equation
def calculate_bmr(weight, height, age, gender):
    if gender.lower() == 'male':
        bmr = 10*weight + 6.25*height - 5*age + 5
    else:
        bmr = 10*weight + 6.25*height - 5*age - 161
    return bmr

# Calculate TDEE based on activity level
def calculate_tdee(bmr, activity_level):
    activity_multipliers = {
        "Sedentary (little or no exercise)": 1.2,
        "Lightly active (light exercise/sports 1-3 days/week)": 1.375,
        "Moderately active (moderate exercise/sports 3-5 days/week)": 1.55,
        "Very active (hard exercise/sports 6-7 days a week)": 1.725,
        "Extra active (very hard exercise/sports & physical job)": 1.9
    }
    return bmr * activity_multipliers.get(activity_level, 1.2)

# Generate macro nutrient distribution based on goal
def macro_distribution(calories, goal):
    # Protein: 4 cal/g, Carbs: 4 cal/g, Fat: 9 cal/g
    if goal == 'Weight Loss':
        protein = 0.35 * calories / 4
        carbs = 0.40 * calories / 4
        fat = 0.25 * calories / 9
    elif goal == 'Muscle Gain':
        protein = 0.30 * calories / 4
        carbs = 0.50 * calories / 4
        fat = 0.20 * calories / 9
    else:  # Maintenance
        protein = 0.25 * calories / 4
        carbs = 0.50 * calories / 4
        fat = 0.25 * calories / 9
    return round(protein,1), round(carbs,1), round(fat,1)

# Sample food database (simple)
food_db = pd.DataFrame({
    'Food': ['Chicken Breast', 'Brown Rice', 'Broccoli', 'Almonds', 'Greek Yogurt', 'Olive Oil'],
    'Calories_per_100g': [165, 110, 55, 579, 59, 884],
    'Protein_g': [31, 2.6, 3.7, 21, 10, 0],
    'Carbs_g': [0, 23, 11, 22, 3.6, 0],
    'Fat_g': [3.6, 0.9, 0.6, 50, 0.4, 100]
})

def recommend_foods(protein_g, carbs_g, fat_g):
    # Simple proportional recommendation (not optimized)
    recommendations = []
    total_macro = protein_g + carbs_g + fat_g
    for idx, row in food_db.iterrows():
        score = 0
        score += min(protein_g, row['Protein_g']) / protein_g if protein_g else 0
        score += min(carbs_g, row['Carbs_g']) / carbs_g if carbs_g else 0
        score += min(fat_g, row['Fat_g']) / fat_g if fat_g else 0
        recommendations.append((row['Food'], round(score,2)))
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:3]

def main():
    st.title("Personalized Diet Recommendation System")

    st.header("Enter your details:")
    age = st.number_input("Age (years)", min_value=10, max_value=100, value=25)
    gender = st.selectbox("Gender", ['Male', 'Female'])
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
    height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)
    activity_level = st.selectbox("Activity Level", [
        "Sedentary (little or no exercise)",
        "Lightly active (light exercise/sports 1-3 days/week)",
        "Moderately active (moderate exercise/sports 3-5 days/week)",
        "Very active (hard exercise/sports 6-7 days a week)",
        "Extra active (very hard exercise/sports & physical job)"
    ])
    goal = st.selectbox("Goal", ['Weight Loss', 'Maintenance', 'Muscle Gain'])

    if st.button("Get Recommendation"):
        bmr = calculate_bmr(weight, height, age, gender)
        tdee = calculate_tdee(bmr, activity_level)

        # Adjust calories based on goal
        if goal == 'Weight Loss':
            calories = tdee * 0.8
        elif goal == 'Muscle Gain':
            calories = tdee * 1.2
        else:
            calories = tdee

        protein_g, carbs_g, fat_g = macro_distribution(calories, goal)

        st.subheader("Your daily calorie and macro targets:")
        st.write(f"Calories: {int(calories)} kcal")
        st.write(f"Protein: {protein_g} g")
        st.write(f"Carbohydrates: {carbs_g} g")
        st.write(f"Fat: {fat_g} g")

        st.subheader("Top food recommendations:")
        recommendations = recommend_foods(protein_g, carbs_g, fat_g)
        for food, score in recommendations:
            st.write(f"- {food} (match score: {score})")

if __name__ == "__main__":
    main()


# streamlit run personalized_diet_recommendation.py
# Summary
# Calculates BMR and TDEE based on user inputs
# Adjusts calorie needs per goal
# Provides macro nutrient targets and food recommendations
# Simple, extendable framework for personalized diet suggestions

