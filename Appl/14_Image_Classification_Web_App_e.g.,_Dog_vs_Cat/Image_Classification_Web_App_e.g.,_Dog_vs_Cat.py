# Image_Classification_Web_App_e.g.,_Dog_vs_Cat.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('dog_cat_model.h5')
    return model

def preprocess_image(img: Image.Image):
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Batch dimension
    return img_array

def predict(img, model):
    processed_img = preprocess_image(img)
    preds = model.predict(processed_img)[0][0]
    # Binary classification sigmoid output: closer to 1 -> dog, closer to 0 -> cat
    label = "Dog" if preds > 0.5 else "Cat"
    confidence = preds if preds > 0.5 else 1 - preds
    return label, confidence

def main():
    st.title("Dog vs Cat Image Classification")
    st.write("Upload an image of a dog or cat, and the model will predict the class.")

    model = load_model()
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption='Uploaded Image', use_column_width=True)
        
        if st.button("Predict"):
            label, confidence = predict(img, model)
            st.success(f"Prediction: {label} ({confidence*100:.2f}% confidence)")

if __name__ == "__main__":
    main()

# How to run
# Train a simple binary classifier on dog/cat images or download a pretrained model as dog_cat_model.h5. Example:

# # Quick example to train (not included in this file) using TensorFlow/Keras:
# # Use transfer learning with MobileNet or similar, fine-tune last layer to binary output.
# Install dependencies:

# pip install tensorflow streamlit pillow numpy
# Run the app:

# streamlit run image_classification_web_app.py