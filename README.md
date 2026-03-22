# Glass Counting Challenge - Final Submission

Industrial glass sheet counting using **Hybrid Fourier-CNN** analysis.

## 🚀 Execution Info
1.  Open `Glass_Counting_Pipeline.ipynb`.
2.  Install dependencies using **Section 0**.
3.  Execute all cells to generate **`results.csv`**.

## 📊 Benchmarking (MAE)
| Strategy | Best MAE | Status |
|:---|:---|:---|
| **Baseline 1D CNN** | 2.6883 | Baseline |
| **Classical CV** | 23.3700 | Inaccurate |
| **Hybrid (Initial)** | **2.5891** | **Top Performer** |
| **Hybrid (Fine-Tuned)** | 2.8516 | Overfitting Observed |

## 🛠️ Tech Stack
- **TensorFlow** (1D Global Average Pooling CNN)
- **Fast Fourier Transform** (Global Density Estimation)
- **OpenCV** (Multi-strip Sobel Averaging)

## 📦 External Libraries
The solution requires the following Python packages:
- `opencv-python`: Image processing and Sobel filtering.
- `tensorflow`: Model building and training.
- `numpy`: Numerical analysis and FFT.
- `pandas`: Data handling and CSV reporting.
- `scipy`: Signal processing (peak finding/FFT).
- `matplotlib`: Visualization.

*Results based on 20-image validation subset with ±5% manufacturing tolerance.*
