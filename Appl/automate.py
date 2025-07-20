import os

# Get base directory (where the script is)
base_dir = os.path.dirname(os.path.abspath(__file__))

# List of project names
project_names = [
    "Customer Churn Prediction System",
    "Movie Recommendation System",
    "Spam Email Classifier",
    "Fake News Detection App",
    "Real Estate Price Predictor",
    "Stock Market Trend Analyzer",
    "Credit Card Fraud Detection System",
    "Loan Approval Prediction Model",
    "Resume Screening Bot",
    "Sentiment Analysis Dashboard",
    "E-commerce Product Recommendation Engine",
    "Voice Command Recognition App",
    "Facial Emotion Detection System",
    "Image Classification Web App (e.g., Dog vs Cat)",
    "COVID-19 Data Analysis and Forecasting Tool",
    "Traffic Signs Recognition App",
    "Air Quality Index Predictor",
    "Handwritten Digit Recognition Web App",
    "Mental Health Chatbot using NLP",
    "Smart Expense Tracker with Anomaly Detection",
    "Personalized Diet Recommendation System",
    "Employee Attrition Prediction System",
    "Text Summarization Tool",
    "News Article Topic Classification System",
    "Interactive Data Dashboard for Sales Analytics"
]

for i, project in enumerate(project_names, 1):
    # Format names
    dir_name = f"{i}_{project.replace(' ', '_').replace('(', '').replace(')', '')}"
    file_base = project.replace(' ', '_').replace('(', '').replace(')', '')

    dir_path = os.path.join(base_dir, dir_name)
    os.makedirs(dir_path, exist_ok=True)

    # Create .py file
    py_path = os.path.join(dir_path, f"{file_base}.py")
    with open(py_path, "w") as py_file:
        py_file.write(f"# {file_base}.py\n")

    # Create .txt file
    txt_path = os.path.join(dir_path, f"{file_base} notes.txt")
    with open(txt_path, "w") as txt_file:
        txt_file.write(f"{file_base} notes\n")
