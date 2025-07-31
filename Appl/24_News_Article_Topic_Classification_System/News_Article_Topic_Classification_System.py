# News_Article_Topic_Classification_System.py
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

def load_data():
    categories = ['rec.sport.baseball', 'comp.graphics', 'sci.med', 'talk.politics.misc']
    data = fetch_20newsgroups(subset='all', categories=categories, remove=('headers','footers','quotes'))
    return data.data, data.target, data.target_names

def preprocess_and_vectorize(texts):
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

def train_model(X_train, y_train):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(clf, X_test, y_test, target_names):
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=target_names))

def main():
    print("Loading data...")
    texts, labels, target_names = load_data()

    print("Splitting data...")
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    print("Vectorizing text data...")
    X_train, vectorizer = preprocess_and_vectorize(X_train_texts)
    X_test = vectorizer.transform(X_test_texts)

    print("Training model...")
    clf = train_model(X_train, y_train)

    print("Evaluating model...")
    evaluate_model(clf, X_test, y_test, target_names)

    # Sample prediction
    sample_text = "The team won the baseball game in the final inning with a home run."
    sample_vec = vectorizer.transform([sample_text])
    pred = clf.predict(sample_vec)
    print(f"\nSample text prediction:\n'{sample_text}'\n=> Topic: {target_names[pred[0]]}")

if __name__ == "__main__":
    main()
