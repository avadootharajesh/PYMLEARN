import streamlit as st
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import matplotlib.pyplot as plt

nltk.download('vader_lexicon')

def analyze_sentiment(texts):
    sid = SentimentIntensityAnalyzer()
    sentiments = [sid.polarity_scores(text) for text in texts]
    df = pd.DataFrame(sentiments)
    df['text'] = texts
    return df

def plot_sentiment_distribution(df):
    counts = df['compound'].apply(lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral')).value_counts()
    fig, ax = plt.subplots()
    counts.plot(kind='bar', color=['green', 'red', 'gray'], ax=ax)
    ax.set_title("Sentiment Distribution")
    ax.set_ylabel("Number of Texts")
    st.pyplot(fig)

def main():
    st.title("Sentiment Analysis Dashboard")
    
    st.write("Enter texts (one per line) to analyze sentiment:")
    user_input = st.text_area("Texts", height=200)
    
    if st.button("Analyze"):
        texts = [line.strip() for line in user_input.split('\n') if line.strip()]
        if not texts:
            st.warning("Please enter at least one text line.")
            return
        
        df = analyze_sentiment(texts)
        
        st.subheader("Sentiment Scores")
        st.dataframe(df[['text', 'compound', 'pos', 'neu', 'neg']])
        
        plot_sentiment_distribution(df)

if __name__ == "__main__":
    main()
