# Glass Counting Challenge - Development & Evaluation

Industrial counting of glass sheet stacks using **Hybrid 1D Fourier-CNN** methods. 
Our primary objective is to achieve a Mean Absolute Error (MAE) of **< 0.5**.

## 🚀 Environment Setup
Ensure dependencies are installed before running:
```bash
pip install opencv-python tensorflow numpy pandas scipy matplotlib tensorflowjs
```
*Note: Section 0 of the notebook implements this automatically.*

## 📊 Final Leaderboard (Latest Run)
| Strategy | MAE | Evaluation Status |
|:---|:---|:---|
| **#1 Hybrid Initial (Fourier-CNN)** | **2.8964** | **🏆 Best Overall Strategy** |
| **#2 Strategy 4 (Synthetic Pre-trained)** | 3.2689 | Robust but requires tuning |
| **#3 Baseline 1D CNN** | 4.5776 | Initial Prototype |
| **#4 Hybrid (Fine-Tuned)** | 5.3574 | Overfitted (High Variance) |
| **#5 Classical CV** | 22.2000 | Baseline (No ML) |

## 🛠️ Implementation Highlights
- **Dilated 1D Convolutions**: Captures periodic sheet features at multiple scales.
- **Global Average & Max Concatenation**: Extracts global signal density accurately.
- **Hybrid Domain Analysis**: Merges Frequency Domain (FFT) with Spatial Domain (CNN).
- **Synthetic Pre-training**: Attempts to bootstrap counting logic for small datasets.

## 📦 Result Generation
Run the final cell in the notebook to generate the competition submission: **`results.csv`**.

---
*Optimized for ±5% manufacturing tolerance across 20-35 sheet stacks.*
