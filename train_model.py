import os
import glob
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# Parameters matching the mobile app
VECTOR_SIZE = 1024
INPUT_SHAPE = (VECTOR_SIZE, 1)

def preprocess_image_1d(img_path):
    # Match the updated ResultScreen.js logic
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
        
    # Note: Training images (Portrait) have Horizontal sheets (stacking along Y).
    # Mobile app uses Vertical sheets in Landscape (also stacking along X, swept along Width).
    # We always sweep the dimension that has the sheets. 
    # For training images, that is Y (Height=768).
    
    # 1. Apply Vertical Sobel (detects the horizontal lines in these specific training images)
    # This matches the 'edges' logic in the app.
    sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.abs(sobel)
    sobel = (255 * sobel / (np.max(sobel) + 1e-8)).astype(np.uint8)
    
    h, w = sobel.shape
    signals = []
    num_strips = 5
    
    # Replicate the 5-strip averaging
    for i in range(1, num_strips + 1):
        center_w = (w * i) // (num_strips + 1)
        strip_w = w // 20
        start_x = max(0, center_w - strip_w // 2)
        strip = sobel[:, start_x : start_x + strip_w]
        signals.append(np.mean(strip, axis=1)) # Average across width of strip -> result is along Height
        
    avg_signal = np.mean(signals, axis=0).astype(np.float32)
    
    # Resize to 1024
    avg_signal = cv2.resize(avg_signal, (1, VECTOR_SIZE), interpolation=cv2.INTER_LINEAR).flatten()
    
    # Adaptive normalization (Min-Max)
    min_val, max_val = np.min(avg_signal), np.max(avg_signal)
    signal_norm = (avg_signal - min_val) / (max_val - min_val + 1e-8)
    
    # Standardization (Z-score)
    mean, std = np.mean(signal_norm), np.std(signal_norm)
    if std > 0:
        final_signal = (signal_norm - mean) / std
    else:
        final_signal = signal_norm
        
    return final_signal

def build_model():
    # Strategy 3 Hybrid Architecture
    model = Sequential([
        Conv1D(64, 15, padding='same', activation='relu', input_shape=INPUT_SHAPE),
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
    print("Loading Dataset...")
    df = pd.read_csv('labels.csv')
    df['img_number'] = df['img_number'].astype(str).str.zfill(3)
    labels_dict = dict(zip(df['img_number'], df['num_sheets']))
    
    X, y = [], []
    train_paths = sorted(glob.glob('train/*.jpg'))
    for path in train_paths:
        img_id = os.path.basename(path).split('.')[0]
        if img_id in labels_dict:
            signal = preprocess_image_1d(path)
            if signal is not None:
                X.append(signal)
                y.append(labels_dict[img_id])
                
    X = np.array(X).reshape(-1, VECTOR_SIZE, 1)
    y = np.array(y).astype(np.float32)
    
    print(f"Loaded {len(X)} samples.")
    model = build_model()
    
    print("Retraining Model...")
    model.fit(X, y, epochs=150, batch_size=16, validation_split=0.2, verbose=1)
    
    model_name = 'glass_counter_hybrid_v54.h5'
    model.save(model_name)
    print(f"Saved: {model_name}")
    
    # Convert to TFJS
    print("Converting to TFJS...")
    output_dir = 'assets/web_model'
    cmd = (f'"{os.path.join(os.getcwd(), "venv", "Scripts", "python.exe")}" -m tensorflowjs.converters.converter '
           f'--input_format=keras {model_name} {output_dir}')
    os.system(cmd)
    print(f"Conversion complete. Updated {output_dir}")

if __name__ == "__main__":
    main()
