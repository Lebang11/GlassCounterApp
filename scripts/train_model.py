import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from utils import load_dataset, VECTOR_SIZE

# Configuration
TRAIN_DIR = os.path.join('..', 'train')
LABELS_CSV = os.path.join('..', 'labels.csv')
MODEL_PATH = os.path.join('..', 'glass_counter_hybrid_finetuned.h5')

def build_model():
    model = Sequential([
        Conv1D(64, kernel_size=10, activation='relu', input_shape=(VECTOR_SIZE, 1)),
        MaxPooling1D(pool_size=4),
        Conv1D(128, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=4),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.02), loss='mse', metrics=['accuracy'])
    return model

def main():
    X_train, y_train, _ = load_dataset(TRAIN_DIR, LABELS_CSV)
    
    if len(X_train) == 0:
        print("Error: No training data found.")
        return

    print(f"Loaded {len(X_train)} samples. Building and training model...")
    model = build_model()
    model.summary()

    history = model.fit(X_train, y_train, epochs=100, batch_size=8, validation_split=0.1, verbose=1)

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE Progression')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.savefig(os.path.join('..', 'training_history.png'))
    print("Training plot saved to root directory.")

    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
