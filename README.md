# Lung Segmentation with Deep Learning

**Event:** Chest MRI School - September 2026, Aberdeen  
**Author:** Nathalie Barrau  
**Notebook:** `TP_lung_segmentation.ipynb`

## Overview

This hands-on lab introduces modern deep learning approaches for medical image segmentation, using **automatic lung segmentation from chest CT scans** as a practical use case. Participants will explore how segmentation methods have evolved over the last decade, from task-specific supervised networks to large-scale foundation models.

Although the school is dedicated to chest MRI, the concepts, architectures, and training pipelines covered here using chest CT images are directly transferable to MRI segmentation tasks.

## Learning objectives

By the end of the lab, participants will have practical experience with three major paradigms in modern medical image segmentation:

1. **Supervised segmentation** with a U-Net trained from scratch
2. **Prompt-based foundation-model segmentation** with SAM (Segment Anything Model)
3. **Foundation-model representations** with DINOv2 as a frozen vision backbone

## Lab content

### 0. Setup
- Install Python dependencies (PyTorch, SimpleITK, segment-anything, etc.)
- Download the SAM ViT-B checkpoint
- Configure GPU/CPU device and reproducibility seed

### 1. Dataset - COVID-19 CT Seg
- Download 9 chest CT volumes and lung masks from the [COVID-19 CT Lung and Infection Segmentation Dataset](https://zenodo.org/records/3757476) (Ma et al.)
- Apply lung windowing (`[-1000, 400] HU`) and intensity normalization
- Extract, filter, and resize 2D axial slices to `256 x 256`
- **Patient-level split** into training / validation / testing to avoid data leakage

### 2. U-Net segmentation
- Implement a modernized U-Net (BatchNorm, padded convolutions)
- Combined **BCE + Dice loss** with batch-Dice evaluation
- Full training loop with AdamW and cosine learning-rate scheduler
- Convergence analysis, quantitative Dice evaluation, and qualitative visualization
- Brief overview of **nnU-Net** as a self-configuring pipeline

### 3. SAM - Promptable foundation model
- Introduction to the SAM architecture (ViT encoder + prompt encoder + mask decoder)
- **Bounding-box prompt** segmentation on 2D slices
- **Interactive click-based** point prompts (positive/negative)
- **Automatic mask generation** with dense prompt grids
- **3D volume segmentation** via slice-by-slice box propagation
- Discussion of medical adaptations: **MedSAM** and **MedSAM2**

### 4. DINOv2 - Foundation-model representations
- Introduction to self-supervised Vision Transformers
- Use of the frozen DINOv2 ViT backbone as a feature extractor
- Training a lightweight decoder head on top of DINOv2 patch embeddings
- Comparison with the fully supervised U-Net baseline

## Requirements

- Python 3.10+
- GPU strongly recommended (CUDA 12.x); CPU is supported with reduced epochs
- Main dependencies: `torch`, `torchvision`, `numpy`, `scipy`, `matplotlib`, `SimpleITK`, `scikit-image`, `Pillow`, `segment-anything`,

A dedicated virtual environment is recommended:

```bash
python -m venv chest_tp
```

All dependencies are installed from the first cells of the notebook.

## Dataset & model credits

- **COVID-19 CT Seg dataset:** Ma et al., [Zenodo 3757476](https://zenodo.org/records/3757476)
- **U-Net:** Ronneberger et al., MICCAI 2015
- **nnU-Net:** Isensee et al., Nature Methods 2021
- **SAM:** Kirillov et al., Meta AI, 2023
- **MedSAM / MedSAM2:** Ma et al., 2024–2025
- **DINOv2:** Oquab et al., Meta AI, 2023

## License & usage

This material is provided for **educational purposes** in the context of the Chest MRI School 2026 (Aberdeen). Datasets and pretrained models remain under their respective original licenses.
