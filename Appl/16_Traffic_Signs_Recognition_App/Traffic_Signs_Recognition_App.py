# Traffic_Signs_Recognition_App.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load GTSRB class names (43 classes)
CLASS_NAMES = [
    'Speed limit 20 km/h', 'Speed limit 30 km/h', 'Speed limit 50 km/h', 'Speed limit 60 km/h',
    'Speed limit 70 km/h', 'Speed limit 80 km/h', 'End of speed limit 80 km/h', 'Speed limit 100 km/h',
    'Speed limit 120 km/h', 'No passing', 'No passing for vehicles over 3.5 tons',
    'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop',
    'No vehicles', 'Vehicles over 3.5 tons prohibited', 'No entry', 'General caution',
    'Dangerous curve to the left', 'Dangerous curve to the right', 'Double curve',
    'Bumpy road', 'Slippery road', 'Road narrows on the right', 'Road work',
    'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing',
    'Beware of ice/snow', 'Wild animals crossing', 'End of all speed and passing limits',
    'Turn right ahead', 'Turn left ahead', 'Ahead only', 'Go straight or right',
    'Go straight or left', 'Keep right', 'Keep left', 'Roundabout mandatory',
    'End of no passing', 'End of no passing by vehicles over 3.5 tons'
]

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('traffic_sign_model.h5')
    return model

def preprocess_image(image: Image.Image):
    img = image.resize((32, 32))  # GTSRB images are 32x32
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize to [0,1]
    img_array = img_array.reshape(1, 32, 32, 3)
    return img_array

def predict(image, model):
    processed_img = preprocess_image(image)
    preds = model.predict(processed_img)
    pred_class = np.argmax(preds)
    confidence = preds[0][pred_class]
    return CLASS_NAMES[pred_class], confidence

def main():
    st.title("Traffic Signs Recognition App")
    st.write("Upload an image of a traffic sign, and the model will predict its class.")

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
