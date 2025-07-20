import os

headings = [
    "ML_Workflow",
    "Data_Cleaning",
    "EDA",
    "Classification_Models",
    "Regression_Models",
    "Feature_Engineering",
    "Hyperparameter_Tuning",
    "ML_Algorithms",
    "Flask_REST_APIs",
    "Flask_Frontend",
    "Model_Deployment",
    "Real_World_Datasets",
    "Recommendation_Systems",
    "NLP_Techniques",
    "Real_Time_Inference",
    "Evaluation_Metrics",
    "Version_Control",
    "Time_Series",
    "Image_Classification",
    "ML_Visualizations",
    "Model_Interpretability",
    "Flask_Web_App",
    "JSON_APIs",
    "Best_Practices",
    "Problem_Solving"
]

# print(os.getcwd())

# for i, heading in enumerate(headings, 1):
#     dir_name = f"{i}_{heading}"
#     os.makedirs(dir_name, exist_ok=True)

currdir = os.path.dirname(os.path.abspath(__file__))
for i, heading in enumerate(headings, 1):
    dir_name = f"{i}_{heading}"
    dir_path = os.path.join(currdir, dir_name)
    
    # create directory if it doesn't exist
    os.makedirs(dir_path, exist_ok=True)
    
    filenamebase = heading
    pypath = os.path.join(dir_path, f"{filenamebase}.py")
    with open(pypath, "w") as py_file:
        py_file.write(f"# {filenamebase}.py\n")
        
    txtpath = os.path.join(dir_path, f"{filenamebase} notes.txt")
    with open(txtpath, "w") as txt_file:
        txt_file.write(f"# {filenamebase} notes.txt\n")
