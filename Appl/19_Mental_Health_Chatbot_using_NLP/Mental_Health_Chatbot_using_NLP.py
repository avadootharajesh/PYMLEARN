# Mental_Health_Chatbot_using_NLP.py
import streamlit as st
from transformers import pipeline

# Load sentiment analysis pipeline (cached)
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis")

# Simple rule-based responses based on keywords and sentiment
def generate_response(user_input, sentiment):
    user_input_lower = user_input.lower()
    
    # Basic keyword-based rules
    if any(word in user_input_lower for word in ['sad', 'depressed', 'unhappy', 'down']):
        return "I'm sorry you're feeling this way. Remember, it's okay to have bad days. Would you like some tips to feel better?"
    elif any(word in user_input_lower for word in ['anxious', 'nervous', 'worried']):
        return "Anxiety can be tough. Try deep breathing exercises or grounding techniques. Want me to share some?"
    elif any(word in user_input_lower for word in ['happy', 'good', 'great', 'fine']):
        return "That's wonderful to hear! Keep focusing on what makes you happy."
    elif any(word in user_input_lower for word in ['help', 'support', 'talk']):
        return "I'm here to listen. Feel free to share more about what's on your mind."
    else:
        # Use sentiment to guide neutral responses
        if sentiment == 'NEGATIVE':
            return "It sounds like you're going through a hard time. I'm here to support you."
        else:
            return "Thanks for sharing. Tell me more about how you're feeling."

def main():
    st.title("Mental Health Chatbot")
    st.write("I'm here to listen and support you. Type anything to start the conversation.")

    sentiment_model = load_sentiment_model()

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("You:", key="input")

    if user_input:
        sentiment_result = sentiment_model(user_input)[0]
        sentiment_label = sentiment_result['label']
        response = generate_response(user_input, sentiment_label)

        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))

    # Display chat history
    for speaker, msg in st.session_state.chat_history:
        if speaker == "You":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**Bot:** {msg}")

if __name__ == "__main__":
    main()
