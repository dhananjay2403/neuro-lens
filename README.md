# NeuroLens 🧠 - MRI Brain Tumor Detection & Segmentation Pipeline 

---

## 📦 BraTS2020 Dataset Overview

- **Gold standard** for brain tumor segmentation (MICCAI BraTS challenge)
- **369** multi-institutional 3D MRI volumes (T1, T1ce, T2, FLAIR)
- **Expert-annotated** masks: necrotic core, edema, enhancing tumor
- **240×240×155** voxels, 1mm³ isotropic, skull-stripped, co-registered
- Used for **training, validation, and benchmarking** of segmentation models

---

## 🚀 Overview

An end-to-end deep learning solution for automated brain tumor detection and segmentation in MRI scans, developed during my research internship at **DRDO-INMAS**.

This project accelerates radiological workflows using 2D/3D MRI datasets and scalable AI pipelines. The pipeline includes detection (**YOLOv8**), 3D U-Net segmentation, and visualization via **Streamlit**.

---

## 🏗️ Architecture

![Brain Tumor Detection and Segmentation Pipeline](img/Full_Architecture.png)

---

## 📊 Key Metrics (BraTS2020 & Pipeline)

| Task           | Metric                | Value      |
|----------------|----------------------|------------|
| Detection      | mAP@50                | 91.4%      |
| Detection      | Precision             | 90.8%      |
| Detection      | Recall                | 86.5%      |
| Segmentation   | Dice Score (mean)     | 0.98       |
| Segmentation   | HD95                  | 3.8 mm     |
| Segmentation   | Inference Time        | 55 ms      |
| Region-wise    | Dice (Enhancing Tumor)| 0.79       |
| Region-wise    | Dice (Tumor Core)     | 0.83       |
| Region-wise    | Dice (Whole Tumor)    | 0.87       |

---

## ⚡ Usage

1. **Clone the repository**
2. **Create and activate a conda environment:**

   ```bash
   conda create -n brats python=3.8 -y
   conda activate brats
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the main notebook for segmentation:**

   ```bash
   jupyter notebook brain-tumor-segmentation-model.ipynb
   ```

---

## 📁 Project Structure

```bash
/brain-tumor-segmentation-model.ipynb   # Main pipeline notebook
/Segmentation_Project/                  # Modular code (data, model, generator)
/input_images/                          # Example input NPY images
/results/                              # Model weights, predictions, metrics
/requirements.txt                      # Dependencies
```

---

## 📚 References

- [BraTS 2020 Dataset](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)


