import os
import numpy as np
from tensorflow.keras.models import load_model
from utils import load_dataset

# Configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
TRAIN_DIR = os.path.join(project_root, 'train')
LABELS_CSV = os.path.join(project_root, 'labels.csv')
MODEL_PATH = os.path.join(project_root, 'glass_counter_model_finetuned.h5')
# Fallback to the original model if finetuned doesn't exist
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(project_root, 'glass_counter_model.h5')

def main():
    print(f"Using model: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print("Error: Model not found. Please train the model first.")
        return
        
    model = load_model(MODEL_PATH, compile=False)
    X_val, y_true, img_paths = load_dataset(TRAIN_DIR, LABELS_CSV)
    
    # We only evaluate on a subset of training data as a verification
    X_subset = X_val[-20:]
    y_subset = y_true[-20:]
    path_subset = img_paths[-20:]
    
    print("Running evaluation...")
    preds = model.predict(X_subset).flatten()
    
    mae = np.mean(np.abs(y_subset - preds))
    print("\n--- Accuracy Report (Subset of 20 images) ---")
    for i in range(len(preds)):
        print(f"Image: {os.path.basename(path_subset[i])} | True: {y_subset[i]:.0f} | Predicted: {preds[i]:.1f}")
        
    print(f"\nFinal Mean Absolute Error (MAE): {mae:.3f}")

if __name__ == "__main__":
    main()
