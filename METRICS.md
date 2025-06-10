# 📊 Segmentation Metrics Explained: Dice, IoU, Precision, Recall, F1

This document explains the key metrics used to evaluate brain tumor detection and segmentation models, especially for medical imaging projects like BraTS. It also compares these metrics with standard classification metrics and provides practical code examples.

---

## 🚦 Why Not Just Accuracy?

In medical image segmentation, especially for tumors, the region of interest (tumor) is often much smaller than the background. Accuracy is misleading here because a model could predict "background" everywhere and still achieve high accuracy. Instead, we need metrics that focus on the overlap and correctness of the tumor regions.

---

## 🟩 Intersection over Union (IoU) / Jaccard Index

**Definition:**  
Measures the overlap between the predicted and ground truth tumor regions.

**Formula:**  
\[
\text{IoU} = \frac{\text{Area of Overlap (Intersection)}}{\text{Area Covered by Both (Union)}}
\]

**Interpretation:**  
- IoU = 1: Perfect overlap
- IoU = 0: No overlap

**Typical Range (Brain Tumor Segmentation):** 0.7 – 0.9

---

## 🟦 Dice Coefficient (Dice Similarity Coefficient, DSC)

**Definition:**  
Measures the similarity between the predicted and ground truth tumor masks.  
Equivalent to the F1 score for segmentation.

**Formula:**  
\[
\text{Dice} = \frac{2 \times \text{True Positives}}{2 \times \text{True Positives} + \text{False Positives} + \text{False Negatives}}
\]

**Interpretation:**  
- Dice = 1: Perfect match
- Dice = 0: No overlap

**Typical Range (Brain Tumor Segmentation):** 0.8 – 0.95

---

## 🔁 Relationship Between Dice, IoU, and F1

- **Dice and F1 Score** are mathematically identical for binary segmentation.
- **IoU and Dice** are related:
  \[
  \text{IoU} = \frac{\text{Dice}}{2 - \text{Dice}}
  \]
  \[
  \text{Dice} = \frac{2 \times \text{IoU}}{1 + \text{IoU}}
  \]

**Example:**  
- Dice = 0.9 → IoU ≈ 0.82  
- IoU = 0.7 → Dice ≈ 0.82

---

## 🧮 Precision, Recall, F1 Score (for Segmentation)

- **Precision:** Of all pixels predicted as tumor, how many are actually tumor?
- **Recall:** Of all actual tumor pixels, how many did we find?
- **F1 Score:** Harmonic mean of Precision and Recall (same as Dice for segmentation).

---

## 🏥 Why These Metrics Matter in Medical Imaging

| Metric     | What It Tells Us                                    | Typical Good Value |
|------------|-----------------------------------------------------|-------------------|
| Dice       | Overlap of predicted and true tumor regions         | >0.85             |
| IoU        | Strict overlap, penalizes boundary errors           | >0.75             |
| Precision  | Avoids false alarms (healthy tissue as tumor)       | >0.85             |
| Recall     | Avoids missing tumor regions                        | >0.85             |
| F1/Dice    | Balanced measure of overlap and correctness         | >0.85             |

---

## 🧑‍💻 Code Examples (Numpy)

```bash
import numpy as np

def calculate_iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    return intersection / union if union != 0 else 0

def calculate_dice(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    return (2. * intersection) / (y_true.sum() + y_pred.sum()) if (y_true.sum() + y_pred.sum()) != 0 else 0
```

---

## 🏆 BraTS Challenge Context

The BraTS challenge uses Dice to evaluate three tumor subregions:
- **Enhancing Tumor (ET)**
- **Tumor Core (TC)**
- **Whole Tumor (WT)**

**State-of-the-Art Benchmarks:**

| Subregion       | Dice (Mean) | IoU (Mean) |
|-----------------|-------------|------------|
| Enhancing Tumor | 0.85        | 0.74       |
| Tumor Core      | 0.90        | 0.82       |
| Whole Tumor     | 0.92        | 0.85       |

---

## 📝 Key Takeaways

- **Dice** and **IoU** are the gold-standard metrics for segmentation tasks in medical imaging.
- **Precision, Recall, F1** are still important, especially for detection tasks and understanding model behavior.
- For your project, always report Dice, IoU, Precision, and Recall for a comprehensive evaluation.
