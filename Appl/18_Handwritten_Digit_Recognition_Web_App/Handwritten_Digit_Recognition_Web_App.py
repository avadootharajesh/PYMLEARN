# Handwritten_Digit_Recognition_Web_App.py
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('mnist_cnn_model.h5')
    return model

def preprocess_image(image: Image.Image):
    # Convert to grayscale, resize to 28x28, invert colors (MNIST black background)
    img = image.convert('L')
    img = ImageOps.invert(img)
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

def main():
    st.title("Handwritten Digit Recognition Web App")
    st.write("Upload a handwritten digit image, and the app will predict the digit.")

    model = load_model()

    uploaded_file = st.file_uploader("Upload an image of a digit (jpg/png)", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button("Predict"):
            processed_img = preprocess_image(image)
            predictions = model.predict(processed_img)
            predicted_digit = np.argmax(predictions)
            confidence = np.max(predictions)

            st.success(f"Predicted Digit: {predicted_digit} (Confidence: {confidence*100:.2f}%)")

if __name__ == "__main__":
    main()


# ------------------------------------------------------------------------------------  

import tensorflow as tf
from tensorflow.keras import layers, models

def train_and_save_model():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    model.save('mnist_cnn_model.h5')

if __name__ == "__main__":
    train_and_save_model()

# streamlit run handwritten_digit_recognition_web_app.py
