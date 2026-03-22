import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.signal import find_peaks
from utils import load_dataset, preprocess_image_1d, extract_signal

# Paths
TRAIN_DIR = os.path.join('.', 'train')
LABELS_CSV = os.path.join('.', 'labels.csv')

def eval_baseline():
    model_path = os.path.join('.', 'glass_counter_model.h5')
    if not os.path.exists(model_path): return "N/A"
    model = load_model(model_path, compile=False)
    X, y, _ = load_dataset(TRAIN_DIR, LABELS_CSV)
    preds = model.predict(X, verbose=0).flatten()
    return np.mean(np.abs(y - preds))

def eval_classical():
    import glob
    X_paths = sorted(glob.glob(os.path.join(TRAIN_DIR, '*.jpg')))
    df = pd.read_csv(LABELS_CSV)
    df['img_number'] = df['img_number'].astype(str).str.zfill(3)
    labels = dict(zip(df['img_number'], df['num_sheets']))
    
    # Grid Search for best Parameters
    prominences = np.linspace(0.01, 0.3, 10)
    distances = np.arange(2, 15, 2)
    best_mae = float('inf')
    
    signals = [extract_signal(p) for p in X_paths]
    true_counts = [labels.get(os.path.basename(p).split('.')[0], 0) for p in X_paths]
    
    for p in prominences:
        for d in distances:
            maes = []
            for i in range(len(signals)):
                peaks, _ = find_peaks(signals[i], prominence=p, distance=d)
                maes.append(abs(true_counts[i] - len(peaks)))
            avg_mae = np.mean(maes)
            if avg_mae < best_mae:
                best_mae = avg_mae
    return best_mae

def eval_hybrid_test():
    # Use the labeled data split for a real test MAE
    X, y, _ = load_dataset(TRAIN_DIR, LABELS_CSV)
    if len(X) == 0: return "N/A"
    
    # Use a fixed 20% hold-out for 'Test' metrics
    test_idx = int(len(X) * 0.8)
    X_test = X[test_idx:]
    y_test = y[test_idx:]
    
    model_path = os.path.join('.', 'glass_counter_hybrid.h5')
    if not os.path.exists(model_path): return "N/A"
    
    model = load_model(model_path, compile=False)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    print(f"Evaluating Hybrid on {len(X_test)} hold-out samples...")
    preds = model.predict(X_test, verbose=0).flatten()
    return np.mean(np.abs(y_test - preds))

if __name__ == "__main__":
    import glob
    import pandas as pd
    print(f"Strategy 1 (Baseline Train): {eval_baseline()}")
    print(f"Strategy 2 (Classical Grid Search): {eval_classical()}")
    print(f"Strategy 3 (Hybrid Test MAE): {eval_hybrid_test()}")
