# Facial_Emotion_Detection_System.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load face detector and emotion model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_model = load_model('fer_model.h5')  # You need to download or train this model

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (48,48))
    face_img = face_img.astype('float32') / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    face_img = np.expand_dims(face_img, axis=-1)  # Model expects (1,48,48,1)
    return face_img

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        for (x,y,w,h) in faces:
            face = gray[y:y+h, x:x+w]
            processed = preprocess_face(face)
            preds = emotion_model.predict(processed)[0]
            emotion_idx = np.argmax(preds)
            emotion_label = EMOTIONS[emotion_idx]
            confidence = preds[emotion_idx]
            
            # Draw rectangle and label
            color = (255, 0, 0)
            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            text = f"{emotion_label}: {confidence*100:.1f}%"
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imshow('Facial Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
