# BraTS 2020 Dataset Documentation 🧠
**BRA** in **T**umor **S**egmentation Challenge 2020  
*The Gold Standard for Brain Tumor Segmentation Research*

---

## 📌 Overview
BraTS 2020 is the premier dataset for brain tumor segmentation research, curated by clinical experts and used in the annual MICCAI BraTS challenge. It provides standardized, multi-institutional data for developing AI solutions in neuro-oncology.

---

## 🔑 Key Features
| Feature                  | Details                                                                 |
|--------------------------|-------------------------------------------------------------------------|
| **Tumor Types**          | Gliomas (High-Grade and Low-Grade Gliomas)                             |
| **Modalities**           | T1, T1CE (Contrast-Enhanced T1), T2, FLAIR                             |
| **Annotations**          | Expert-labeled segmentation masks for tumor subregions                 |
| **Clinical Relevance**   | Used for treatment planning, survival prediction, and therapy response |

---

## 📊 Dataset Composition
### **Quantitative Analysis**
- **3D Volumes**: 369 patients (259 training + 110 validation)
- **2D Slices**: 200,000+ (369 patients × 4 modalities × 155 slices/volume)
- **Resolution**: 240×240×155 voxels @ 1mm³ isotropic resolution

### **Quality Attributes**
- **Multi-Institutional**: Collected from 19+ medical centers worldwide
- **Preprocessed**: Skull-stripped, co-registered, and resampled
- **Standardized**: Uniform intensity normalization across scanners

---

## 🗂 File Structure & Format
```bash
BraTS2020/
├── Training/
│   ├── BraTS20_Training_001/
│   │   ├── BraTS20_Training_001_t1.nii.gz
│   │   ├── BraTS20_Training_001_t1ce.nii.gz
│   │   ├── BraTS20_Training_001_t2.nii.gz
│   │   ├── BraTS20_Training_001_flair.nii.gz
│   │   └── BraTS20_Training_001_seg.nii.gz
│   └── ... (258 more patients)
└── Validation/
    └── ... (110 patients)
```

**File Format**: NIfTI (.nii.gz)  
**Mask Labels**: 
- 0: Background 
- 1: Necrotic core
- 2: Edema
- 4: Enhancing tumor

---

## ⚙️ Preprocessing Requirements
1. **Label Correction**: Map label 4 → 3 for framework compatibility
2. **Intensity Normalization**: Per-modality Z-score normalization
3. **Patch Extraction**: 128×128×128 voxel patches with 50% overlap
4. **Data Augmentation**: 3D rotations, flips, gamma adjustments

---

## 🏆 Benchmark Metrics
**Evaluation Criteria** (Used in BraTS Challenges):
1. **Dice Similarity Coefficient (DSC)**  
   - Whole Tumor (WT): 0.85-0.92  
   - Tumor Core (TC): 0.75-0.85  
   - Enhancing Tumor (ET): 0.70-0.80  

2. **Hausdorff Distance (95%)**  
   - Measures boundary segmentation accuracy