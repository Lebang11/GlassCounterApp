import os
import glob
import cv2
import numpy as np
import pandas as pd

VECTOR_SIZE = 1024

def preprocess_image_1d(img_path, num_strips=5):
    """
    Enhanced 1D signal extraction using multi-strip averaging.
    Instead of one strip, we average 5 strips across the image width.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
        
    # Apply vertical Sobel filter
    sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.abs(sobel)
    sobel = np.uint8(255 * sobel / np.max(sobel))
    
    h, w = sobel.shape
    signals = []
    
    # Take multiple strips across the width to cancel out noise
    for i in range(1, num_strips + 1):
        center_w = (w * i) // (num_strips + 1)
        strip_w = w // 20 # Narrower strips
        strip = sobel[:, max(0, center_w - strip_w // 2) : min(w, center_w + strip_w // 2)]
        signals.append(np.mean(strip, axis=1))
        
    # Average the signals
    signal_1d = np.mean(signals, axis=0).astype(np.float32)
    
    # Reshape and resize
    signal_1d = signal_1d.reshape(-1, 1)
    signal_resized = cv2.resize(signal_1d, (1, VECTOR_SIZE), interpolation=cv2.INTER_LINEAR)
    signal_resized = signal_resized.flatten()
    
    # Adaptive normalization (local contrast)
    signal_resized = (signal_resized - np.min(signal_resized)) / (np.max(signal_resized) - np.min(signal_resized) + 1e-8)
    mean, std = np.mean(signal_resized), np.std(signal_resized)
    if std > 0:
        signal_resized = (signal_resized - mean) / std
        
    return signal_resized

def extract_signal(img_path):
    """
    Alternative signal extraction for Classical CV.
    """
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

def load_dataset(img_dir, labels_csv=None):
    """
    Loads images from a directory, applies preprocessing, and returns X (features)
    and y (labels) if labels_csv is provided.
    """
    # Ensure relative paths work regardless of where the script is called from
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
    except NameError:
        # Fallback for Jupyter Notebooks
        project_root = os.getcwd()
    
    if not os.path.isabs(img_dir):
        img_dir = os.path.join(project_root, img_dir)
    if labels_csv and not os.path.isabs(labels_csv):
        labels_csv = os.path.join(project_root, labels_csv)
        
    print(f"Loading images from {img_dir}...")
    img_paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
    
    X = []
    y = []
    
    labels_dict = None
    if labels_csv and os.path.exists(labels_csv):
        df = pd.read_csv(labels_csv)
        df['img_number'] = df['img_number'].astype(str).str.zfill(3)
        labels_dict = dict(zip(df['img_number'], df['num_sheets']))

    for path in img_paths:
        basename = os.path.basename(path)
        img_id = basename.split('.')[0]
        
        signal = preprocess_image_1d(path)
        if signal is not None:
            X.append(signal)
            if labels_dict:
                y.append(labels_dict.get(img_id, 0))
    
    X = np.array(X)
    X = np.expand_dims(X, axis=-1)
    
    if labels_dict:
        return X, np.array(y), img_paths
    return X, None, img_paths
