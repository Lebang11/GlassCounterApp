import os
import numpy as np
import cv2
import glob
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from utils import load_dataset, VECTOR_SIZE

# Configuration
MODEL_PATH = os.path.join('.', 'glass_counter_hybrid.h5')

def estimate_count_fourier(signal):
    """
    Uses FFT to find the dominant frequency of the glass sheets.
    This gives a very robust baseline estimate.
    """
    # Remove DC component
    signal_centered = signal - np.mean(signal)
    
    # Compute FFT
    fft = np.abs(np.fft.rfft(signal_centered))
    freqs = np.fft.rfftfreq(len(signal_centered))
    
    # Find peak frequency (ignoring very low frequencies)
    # We look for peaks in the range that corresponds to 10-60 sheets
    peak_idx = np.argmax(fft[10:100]) + 10
    dominant_freq = freqs[peak_idx]
    
    # Count is approximately the number of cycles in the length
    est_count = dominant_freq * len(signal_centered)
    return est_count

def build_advanced_model():
    """
    Builds an improved 1D CNN with Batch Normalization.
    """
    model = Sequential([
        Conv1D(64, 15, padding='same', activation='relu', input_shape=(VECTOR_SIZE, 1)),
        BatchNormalization(),
        Conv1D(64, 15, padding='same', activation='relu'),
        BatchNormalization(),
        GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse', metrics=['mae'])
    return model

def main():
    # Setup paths
    train_dir = os.path.join('.', 'train')
    labels_csv = os.path.join('.', 'labels.csv')
    
    X, y, _ = load_dataset(train_dir, labels_csv)
    if len(X) == 0:
        print("Error: No training data found.")
        return
    
    X = X.reshape(X.shape[0], X.shape[1], 1)
    model = build_advanced_model()
    
    print("Training Advanced Hybrid Model (Target MAE < 0.5)...")
    history = model.fit(X, y, epochs=150, batch_size=16, validation_split=0.2, verbose=1)
    
    model.save(MODEL_PATH)
    print(f"Hybrid model saved to {MODEL_PATH}")
    
    # Plot MAE
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title('Advanced Hybrid Model - MAE Progression')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
