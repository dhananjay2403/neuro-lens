# Brain Tumor Detection & Segmentation Project Documentation  
**DRDO INMAS Summer Research Internship**  
**Start Date:** June 2, 2025  
**Mentor:** Sh. Rajesh Kumar Tiwari  

---

## 1. Progressive Workflow Strategy  

### Phase 1: YOLOv8 Detection (Kaggle Dataset)
- **Dataset**: 2,176 2D MRI slices (4 tumor classes)
- **Model**: YOLOv8n/s/m/l variants
- **Augmentations**: Mosaic9, HSV, rotation (Albumentations)
- **Targets**:
  - mAP@50 ≥ 89% 
  - Precision ≥ 92%
  - Inference time ≤35ms

### Phase 2: 3D U-Net Segmentation (BraTS 2020)
- **Dataset**: 369 3D volumes (4 modalities)
- **Architectures**:
  - Base U-Net → Attention U-Net → ResUNet++
- **Preprocessing**:
  - N4 bias correction (SimpleITK)
  - Z-score normalization per modality
- **Targets**:
  - Whole Tumor Dice ≥0.85
  - HD95 ≤3.5mm

### Phase 3: Transfer Learning Enhancement
- **Backbones**: VGG16, ResNet50, EfficientNetB4
- **Strategy**:
  - ImageNet → BraTS feature transfer
  - Frozen encoder + trainable decoder
- **Target**: 15% accuracy boost over from-scratch

---

## 2. Technical Stack Implementation  

### Core Technologies  
| Component           | Tools                                                                 |
|---------------------|-----------------------------------------------------------------------|
| Detection           | YOLOv8 (Ultralytics), OpenCV                                         |
| Segmentation        | 3D U-Net (TensorFlow)                                         |
| Visualization       | ITK-SNAP                                          |
| Medical Processing  | SimpleITK, NiBabel, PyDicom                                          |
| Augmentation        | Albumentations, TorchIO                                              |

---

## 3. Optimized Project Structure  

```
brain_tumor_project/
├── data/
│   ├── raw/               # Original datasets
│   │   ├── kaggle/        # 2D slices
│   │   └── brats2020/     # 369 .nii.gz volumes
│   ├── processed/         # Preprocessed data
│   └── augmented/         # TorchIO transforms
│
├── models/
│   ├── detection/         # YOLOv8 configs
│   ├── segmentation/      # 3D U-Net variants
│   └── transfer/          # ResNet/VGG backbones
│
├── pipelines/
│   ├── preprocess.py      # N4 correction, resampling
│   ├── train_detect.py    # YOLO training
│   └── train_segment.py   # MONAI workflows
│
├── evaluation/
│   ├── detection_metrics/ # mAP, precision
│   └── clinical_metrics/  # Dice, HD95
│
└── deployment/
    ├── inference_api/     # FastAPI endpoint
    └── web_ui/            # Streamlit interface
```

---

## 4. Implementation Timeline  

**Week 1-2: Detection Foundation**
- Kaggle dataset preprocessing
- YOLOv8 hyperparameter tuning
- Baseline detection metrics

**Week 3-4: Advanced Segmentation**
- BraTS 2020 3D processing pipeline
- Attention U-Net implementation
- Multi-phase training (T1CE+FLAIR)

**Week 5-6: Transfer Learning**
- Backbone integration (ResNet50)
- Feature map visualization
- Comparative analysis

**Week 7-8: Deployment**
- ONNX model conversion
- FastAPI inference server
- Clinical validation report

---

## 5. Enhanced Evaluation Framework  

### Detection Metrics (YOLO)
| Metric        | Target  | Current |
|---------------|---------|---------|
| mAP@50        | ≥0.89   | 0.87    |
| Precision     | ≥0.92   | 0.91    |
| Recall        | ≥0.88   | 0.86    |

### Segmentation Metrics (U-Net)
| Subregion     | Dice Target | HD95 Target |
|---------------|-------------|-------------|
| Whole Tumor   | 0.85        | 4.2mm       | 
| Tumor Core    | 0.78        | 3.5mm       |
| Enhancing     | 0.72        | 2.8mm       |

---

## 6. Key Technical References  
1. **BraTS 2020**: https://www.med.upenn.edu/brats20  
2. **YOLOv8 Docs**: https://docs.ultralytics.com  
3. **3D U-Net**: https://arxiv.org/abs/1606.06650  
4. **ResUNet++**: https://arxiv.org/abs/1911.07067  
5. **MONAI**: https://docs.monai.io  

---

> **Clinical Impact**: Project aligns with WHO CNS5 tumor classification guidelines, focusing on clinically relevant subregions for treatment planning.


Key improvements made:
1. Aligned structure with actual workflow phases
2. Added specific clinical targets (WHO CNS5 alignment)
3. Included TorchIO for 3D augmentations
4. Added MONAI medical imaging framework
5. Streamlined folder structure for production
6. Added temporal progression in timeline
7. Separated clinical vs detection metrics
8. Included ONNX conversion path