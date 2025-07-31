# Resume_Screening_Bot.py
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def get_resume_texts(folder_path):
    texts = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            full_path = os.path.join(folder_path, filename)
            text = extract_text_from_txt(full_path)
            texts[filename] = preprocess_text(text)
    return texts

def score_resumes(resumes, job_description):
    documents = list(resumes.values()) + [job_description]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # Cosine similarity between job description and each resume
    job_vec = tfidf_matrix[-1]
    resume_vecs = tfidf_matrix[:-1]
    
    similarities = cosine_similarity(resume_vecs, job_vec)
    scores = {filename: sim[0] for filename, sim in zip(resumes.keys(), similarities)}
    return scores

def main():
    # Example: folder 'resumes' with .txt resumes and a job description string
    folder_path = 'resumes'
    job_description = """
    We are looking for a Data Scientist with experience in Python, machine learning,
    data analysis, and SQL. Knowledge of NLP and deep learning is a plus.
    """
    job_description = preprocess_text(job_description)
    
    resumes = get_resume_texts(folder_path)
    scores = score_resumes(resumes, job_description)
    
    # Sort by score descending
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    print("Resume Ranking based on relevance to job description:\n")
    for filename, score in sorted_scores:
        print(f"{filename}: {score:.4f}")

if __name__ == "__main__":
    main()
