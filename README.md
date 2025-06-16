# NeuroLens - MRI Brain Tumor Detection & Segmentation Pipeline 🧠

**DRDO INMAS Summer Research Internship Project**  
**Mentor:** Sh. Rajesh Kumar Tiwari  
**Date:** 1st June - 30th June 2025

---

## Overview

An end-to-end deep learning solution for automated brain tumor detection and segmentation in MRI scans, developed during my research internship at DRDO-INMAS.

This project aims to accelerate and support radiological workflows by leveraging both 2D and 3D MRI datasets and deploying scalable, metric-driven AI pipelines. The pipeline consists of initial detection using YOLOv8, followed by semantic segmentation using 3D U-Net on multimodal volumetric MRI data, with results visualized through a Streamlit web interface.

---

## 🏗️ Architecture

![Brain Tumor Detection and Segmentation Pipeline](img/Full_Architecture.png)

The architecture shows the complete pipeline from data ingestion through detection, segmentation, and visualization. The workflow follows a progressive approach, starting with initial detection using YOLOv8 on 2D slices, followed by 3D segmentation with U-Net, and finally presenting the results through an interactive Streamlit interface.

---

## 🛠️ Technologies Used

- **Programming Languages:** Python
- **Detection Frameworks:** YOLOv8 (Ultralytics)
- **Segmentation Frameworks:** TensorFlow, Keras (for 3D U-Net)
- **Data Augmentation:** MONAI
- **Medical Imaging Tools:** ITK-SNAP, Nibabel
- **Visualization Libraries:** Matplotlib, Seaborn
- **Image Processing:** OpenCV, SimpleITK
- **Web Interface:** Streamlit
- **Data Handling:** Pandas, NumPy

---

## 📊 Key Features

- Multi-stage pipeline: Detection (YOLOv8) → Segmentation (3D U-Net)
- Support for both 2D (Kaggle) and 3D (BraTS 2020) MRI datasets
- Preprocessing pipeline for medical imaging data (normalization, mask reclassification)
- Multimodal analysis combining FLAIR, T1ce, and T2 sequences
- Clinical workflow integration and benchmarking
- Streamlit-based visualization and deployment

---

## 🗂 Dataset Strategy

### Primary Datasets

| Dataset      | Modalities             | Samples | Key Features                  |
|--------------|------------------------|---------|-------------------------------|
| Kaggle MRI   | T2-weighted            | 2,176   | 2D slices, 3 tumor types ([Kaggle Link](https://www.kaggle.com/datasets/pkdarabi/medical-image-dataset-brain-tumor-detection)) |
| BraTS 2020   | T1, T1CE, T2, FLAIR    | 369     | Multimodal 3D volumes with ground-truth segmentations ([BraTS 2020 Dataset on Kaggle](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)) |

---

## 🧠 Progressive Model Development

### ✅ Stage 1: Tumor Detection with YOLOv8

- Model: YOLOv8m
- Dataset: [Kaggle Brain Tumor Detection MRI Dataset](https://www.kaggle.com/datasets/pkdarabi/medical-image-dataset-brain-tumor-detection)
- Tumor Types: Glioma, Meningioma, Pituitary
- Image Size: 640x640
- Optimizer: Adamax
- Augmentations: Mosaic, HSV, rotation

**Final Metrics on Validation Set:**

- **mAP@50:** 91.4%
- **Precision:** 90.8%
- **Recall:** 86.5%

> 🎯 Tumor-wise Breakdown:
>
> - Glioma: mAP@50 = 0.812
> - Meningioma: mAP@50 = 0.973
> - Pituitary: mAP@50 = 0.955

---

### 🧠 Stage 2: 3D Semantic Segmentation with 3D U-Net

- Dataset: BraTS 2020 (multimodal MRI volumes)
- Modalities: T1, T1Gd, T2, FLAIR
- Architecture: **3D U-Net**
- Preprocessing:
  - Intensity normalization
  - NIfTI handling with `nibabel`
  - Mask reclassification (4→3 classes)
  - Slice orientation unification
  - Cropping to focus on brain region
- Data Augmentation:
  - Custom 3D augmentation
  - MONAI-based transforms
- Model Training:
  - Loss Function: Dice + Focal Loss
  - Optimizer: Adam
  - Callbacks: Early stopping, LR reduction
  - Data Split: 80:20 training/validation
  - Metrics: MeanIoU, Accuracy
- Expected Outcome: Class-wise tumor segmentation masks with Dice ≥ 0.85

---

## 📈 Benchmark Goals

| Stage       | Model         | Metric        | Target     | Current    |
|-------------|---------------|---------------|------------|------------|
| Detection   | YOLOv8m       | mAP@50        | ≥ 0.90     | 0.914      |
|             |               | Precision     | ≥ 0.92     | 0.908      |
| Segmentation| 3D U-Net      | Dice Score    | ≥ 0.85     | 0.83       |
|             |               | HD95 (mm)     | ≤ 3.5      | 3.8        |
|             |               | Inference Time| ≤ 60 ms    | 55 ms      |

### BraTS Segmentation Performance by Region

| Subregion       | Dice (Mean) | IoU (Mean) | Target Dice |
|-----------------|-------------|------------|-------------|
| Enhancing Tumor | 0.79        | 0.70       | 0.85        |
| Tumor Core      | 0.83        | 0.74       | 0.90        |
| Whole Tumor     | 0.87        | 0.79       | 0.92        |

---

## 🚀 Workflow

### Data Preprocessing

1. **Loading & Initial Processing**
   - Load NIfTI files using `nibabel`
   - Extract multiple modalities (FLAIR, T1ce, T2)
   - Normalize intensity values per modality
   - Reclassify mask labels for segmentation (4→3 classes)

2. **Augmentation Pipeline**
   - Custom data generator for 3D volumes
   - MONAI-based transforms for medical image augmentation
   - Combine modalities for multimodal input

3. **Training Pipeline**
   - Model: 3D U-Net with TensorFlow/Keras
   - Loss: Combination of Dice and Focal loss
   - Optimizer: Adam with learning rate scheduling
   - Callbacks: Early stopping, model checkpointing, TensorBoard logging

4. **Inference & Visualization**
   - Load saved model from Kaggle or local storage
   - Process new MRI volumes through the pipeline
   - Visualize results with overlay of segmentation masks
   - Deploy through Streamlit web interface

---

## 🗂 Project Structure

```bash
neuro-lens/
├── data/
│   ├── train/          # Training data
│   │   ├── images/     # Input MRI images
│   │   └── labels/     # Segmentation masks
│   ├── valid/          # Validation data
│   └── test/           # Test data
├── notebooks/
│   ├── 1_BraTS_data_preprocessing.ipynb
│   ├── 2_custom_data_generator.ipynb
│   ├── 3_model_training_&_prediction.ipynb
│   └── yolov8_detection.ipynb
├── config/             # Configuration files
├── docs/               # Documentation
├── results/            # Model outputs and visualizations
├── simple_3d_unet.py   # 3D U-Net model definition
└── requirements.txt    # Project dependencies
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for training)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/neuro-lens.git
cd neuro-lens

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Run YOLOv8 detection training
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt').train(data='config/data.yaml', epochs=100)"

# Run 3D U-Net segmentation (via notebook)
jupyter notebook notebooks/3_model_training_&_prediction.ipynb
```

### Inference

```bash
# Launch the Streamlit web interface
streamlit run app.py
```

---

## 💡 Future Directions

- Multimodal fusion with CT and clinical metadata
- Transformer-based 3D segmentation (e.g., SwinUNETR)
- Integration with DICOM for hospital workflow testing
- Lightweight models for edge inference on MR workstations
- Survival prediction based on tumor characteristics

---

## 🤝 Acknowledgments

- DRDO-INMAS mentorship and infrastructure
- Medical imaging community for open datasets
- Ultralytics team for YOLOv8 framework
- BraTS organizers and contributors

---

## 📚 References

1. [BraTS 2020 Dataset](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)
2. [YOLOv8 Documentation](https://docs.ultralytics.com)
3. [ITK-SNAP](http://www.itksnap.org)
4. [3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation](https://arxiv.org/abs/1606.06650)
5. [Kaggle Tumor Detection Dataset](https://www.kaggle.com/datasets/pkdarabi/medical-image-dataset-brain-tumor-detection)

