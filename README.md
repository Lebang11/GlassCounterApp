# Glass Counting Challenge - Solution Documentation

This project provides an end-to-end solution for counting glass sheets in industrial stack images.

## 🚀 How to Run
The entire pipeline is contained within `Glass_Counting_Pipeline.ipynb`.
1.  Open the notebook in Jupyter or VS Code.
2.  Run all cells sequentially.
3.  The final cell (Section 7) will generate `results.csv` for the 50 test images.

## 🛠️ Components
1.  **Ingestion**: Handled by `load_dataset()`. Automatically maps 200 training images from `train/` using `labels.csv`. Any images in `test/` are treated as the evaluation set.
2.  **Modelling**: 
    *   **Strategy 3 (Hybrid Fourier-CNN)**: Replicates the high-frequency detection of glass edges using multi-strip Sobel averaging and FFT density estimation, refined by a 1D Global Average Pooling CNN.
    *   **Preprocessing**: Replicates the mobile app's pipeline (Vertical Sobel on Height-axis signal, normalization, and resize to 1024).
3.  **Reporting**: Outputs `results.csv` with `image_number` and `number_of_sheets`.

## 📦 External Libraries
The solution requires the following Python packages:
- `opencv-python`: Image processing and Sobel filtering.
- `tensorflow`: Model building and training.
- `numpy`: Numerical analysis and FFT.
- `pandas`: Data handling and CSV reporting.
- `scipy`: Signal processing (peak finding/FFT).
- `matplotlib`: Visualization.

## 📉 Results
- **Target MAE**: < 0.5 sheets.
- **Current Performance**: ~1.9 MAE (v1). 
- **Strategy for Target**: The model uses a frequency-aware architecture to handle the specified ±5% width tolerance and 20-35 sheet constraints.

---
Developed as part of the Glass Counting Challenge.
