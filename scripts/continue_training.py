import os
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from utils import load_dataset, VECTOR_SIZE

# Configuration
TRAIN_DIR = os.path.join('..', 'train')
LABELS_CSV = os.path.join('..', 'labels.csv')
MODEL_PATH = os.path.join('..', 'glass_counter_model.h5')
FINETUNED_PATH = os.path.join('..', 'glass_counter_model_finetuned.h5')

def main():
    print("Checking for existing model...")
    if not os.path.exists(MODEL_PATH):
        print("Model not found! Train the initial model first.")
        return
        
    model = load_model(MODEL_PATH, compile=False)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])
    
    X_train, y_train, _ = load_dataset(TRAIN_DIR, LABELS_CSV)
    
    print("Continuing training for 50 more epochs...")
    model.fit(X_train, y_train, epochs=50, batch_size=8, validation_split=0.1, verbose=1)
    
    model.save(FINETUNED_PATH)
    print(f"Fine-tuned model saved as '{FINETUNED_PATH}'")

if __name__ == "__main__":
    main()
