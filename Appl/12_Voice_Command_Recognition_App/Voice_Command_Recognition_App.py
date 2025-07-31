# Voice_Command_Recognition_App.py
import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Commands to recognize
COMMANDS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

def load_audio_files(data_path, commands=COMMANDS, sample_rate=16000, max_files_per_command=100):
    X, y = [], []
    for idx, command in enumerate(commands):
        folder = os.path.join(data_path, command)
        files = os.listdir(folder)[:max_files_per_command]  # Limit files for quick training
        for f in files:
            file_path = os.path.join(folder, f)
            signal, sr = librosa.load(file_path, sr=sample_rate)
            mfcc = librosa.feature.mfcc(signal, sr=sr, n_mfcc=40)
            mfcc = mfcc.T
            if mfcc.shape[0] < 44:  # Pad/truncate to 44 frames
                pad_width = 44 - mfcc.shape[0]
                mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
            else:
                mfcc = mfcc[:44, :]
            X.append(mfcc)
            y.append(idx)
    return np.array(X), np.array(y)

def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    data_path = "speech_commands_dataset"  # Folder containing folders 'yes', 'no', etc.
    
    print("Loading audio data...")
    X, y = load_audio_files(data_path)
    X = X[..., np.newaxis]  # Add channel dimension
    y_cat = to_categorical(y, num_classes=len(COMMANDS))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y)
    
    model = build_model(input_shape=X_train.shape[1:], num_classes=len(COMMANDS))
    print(model.summary())
    
    print("Training model...")
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)
    
    print("Evaluating model...")
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc:.4f}")
    
    # Save model
    model.save("voice_command_recognition_model.h5")
    print("Model saved as voice_command_recognition_model.h5")

if __name__ == "__main__":
    main()
