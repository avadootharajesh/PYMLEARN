# Fake_News_Detection_App.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

def load_data():
    # Using Kaggle Fake News dataset sample from URL (or replace with local file)
    url = "https://raw.githubusercontent.com/laxmimerit/fake-real-news-dataset/master/fake_or_real_news.csv"
    df = pd.read_csv(url)
    return df

def preprocess_data(df):
    # Label encode: 'FAKE' = 1, 'REAL' = 0
    df['label_num'] = df['label'].map({'REAL':0, 'FAKE':1})
    return df

def train_and_evaluate(df):
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label_num'], test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

def main():
    df = load_data()
    df = preprocess_data(df)
    train_and_evaluate(df)

if __name__ == "__main__":
    main()
