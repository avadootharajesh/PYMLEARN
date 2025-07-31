# Spam_Email_Classifier.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

def load_data():
    # Using SMS Spam Collection Dataset (UCI)
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])
    return df

def preprocess_data(df):
    df['label_num'] = df.label.map({'ham':0, 'spam':1})
    return df

def train_and_evaluate(df):
    X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label_num'], test_size=0.2, random_state=42)

    # Vectorize text using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train Naive Bayes classifier
    clf = MultinomialNB()
    clf.fit(X_train_tfidf, y_train)

    # Predict and evaluate
    y_pred = clf.predict(X_test_tfidf)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

def main():
    df = load_data()
    df = preprocess_data(df)
    train_and_evaluate(df)

if __name__ == "__main__":
    main()
