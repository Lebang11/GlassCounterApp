import os
import glob
import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# Configuration
CSV_PATH = os.path.join('..', 'labels.csv')
TRAIN_DIR = os.path.join('..', 'train')
TEST_DIR = os.path.join('..', 'test')

def extract_signal_for_cv(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape
    new_w = 400
    new_h = int(h * (new_w / w))
    img = cv2.resize(img, (new_w, new_h))
    sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.abs(sobel)
    sobel = np.uint8(255 * sobel / np.max(sobel))
    strip_w = 40
    center_w = new_w // 2
    strip = sobel[:, center_w - strip_w // 2 : center_w + strip_w // 2]
    signal = np.mean(strip, axis=1)
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(signal, kernel, mode='same')
    if np.max(smoothed) > 0:
        smoothed = smoothed / np.max(smoothed)
    return smoothed

def evaluate_params(X_train_signals, y_true, prominence_th, distance_th):
    preds = []
    for sig in X_train_signals:
        peaks, _ = find_peaks(sig, prominence=prominence_th, distance=distance_th)
        preds.append(len(peaks))
    preds = np.array(preds)
    mae = np.mean(np.abs(y_true - preds))
    return mae, preds

def main():
    print("Loading labels...")
    df = pd.read_csv(CSV_PATH)
    df['img_number'] = df['img_number'].astype(str).str.zfill(3)
    labels_dict = dict(zip(df['img_number'], df['num_sheets']))

    print("Loading training signals...")
    train_paths = sorted(glob.glob(os.path.join(TRAIN_DIR, '*.jpg')))
    X_train_signals = []
    y_true = []
    for path in train_paths:
        basename = os.path.basename(path)
        img_id = basename.split('.')[0]
        sig = extract_signal_for_cv(path)
        if sig is not None:
            X_train_signals.append(sig)
            y_true.append(labels_dict.get(img_id, 0))

    prominences = [0.05, 0.1, 0.15, 0.2]
    distances = [5, 8, 10, 12]
    best_mae = float('inf')
    best_params = (0.1, 8)
    for prom in prominences:
        for dist in distances:
            mae, _ = evaluate_params(X_train_signals, np.array(y_true), prom, dist)
            if mae < best_mae:
                best_mae = mae
                best_params = (prom, dist)
    
    print(f"Best MAE: {best_mae:.4f} @ Prom: {best_params[0]}, Dist: {best_params[1]}")

    print("Processing test images...")
    test_paths = sorted(glob.glob(os.path.join(TEST_DIR, '*.jpg')))
    results = []
    for path in test_paths:
        basename = os.path.basename(path)
        img_id = basename.split('.')[0]
        sig = extract_signal_for_cv(path)
        if sig is not None:
            peaks, _ = find_peaks(sig, prominence=best_params[0], distance=best_params[1])
            results.append({'image_number': img_id, 'number_of_sheets': len(peaks)})
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join('..', 'results_classical.csv'), index=False)
    print("Saved results_classical.csv")

if __name__ == "__main__":
    main()
