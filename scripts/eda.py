import os
import glob
import cv2
import numpy as np
import pandas as pd
from utils import extract_signal # Not in utils yet, but I'll add it or keep a version here

# Configuration
TRAIN_DIR = os.path.join('..', 'train')
LABELS_CSV = os.path.join('..', 'labels.csv')

def extract_signal_naive(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape
    center_w = w // 2
    strip_w = w // 5
    strip = img[:, center_w - strip_w // 2 : center_w + strip_w // 2]
    signal_1d = np.mean(strip, axis=1)
    return signal_1d

def main():
    print("Running EDA...")
    if not os.path.exists(TRAIN_DIR):
        print(f"Error: {TRAIN_DIR} not found.")
        return

    df = pd.read_csv(LABELS_CSV)
    df['img_number'] = df['img_number'].astype(str).str.zfill(3)
    
    # Just look at the first image properties
    first_img = os.path.join(TRAIN_DIR, '000.jpg')
    img = cv2.imread(first_img)
    if img is not None:
        print(f"Image Shape: {img.shape}")
        print(f"Image Type: {img.dtype}")
        
    print("Done with EDA info.")

if __name__ == "__main__":
    main()
