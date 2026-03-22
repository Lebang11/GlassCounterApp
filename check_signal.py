import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    img_path = 'train/000.jpg'
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Image not found.")
        return
        
    print(f"Image shape: {img.shape}")
    
    # 1. Apply vertical Sobel filter (dy=1, detections horizontal edges)
    sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.abs(sobel)
    
    # 2. Extract signal along Y (mean across columns)
    signal = np.mean(sobel, axis=1)
    
    print(f"Signal Length: {len(signal)}")
    print(f"Signal Stats - Min: {np.min(signal):.2f}, Max: {np.max(signal):.2f}, Mean: {np.mean(signal):.2f}")
    
    # Check if there are peaks
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(signal, prominence=np.max(signal)/4)
    print(f"Found {len(peaks)} peaks in raw signal.")

if __name__ == "__main__":
    main()
