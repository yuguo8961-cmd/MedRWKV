# MedRWKV: Tri-Directional RWKV with Deformable Fusion for 3D Medical Image Segmentation

> Official PyTorch implementation of **MedRWKV: Tri-Directional RWKV with Deformable Fusion for 3D Medical Image Segmentation**.

**Status:** Under double-blind review  
**Task:** 3D medical image segmentation  
**Framework:** PyTorch  
**Keywords:** 3D medical image segmentation, Vision-RWKV, deformable fusion, boundary-aware segmentation

---

## 📰 News

- **[2026-XX-XX]** Code repository released for anonymous review.
- **[2026-XX-XX]** Training and evaluation scripts will be organized progressively.

---

## 📌 Overview

Accurate 3D medical image segmentation remains challenging due to blurred boundaries, irregular lesion morphology, large inter-patient variation, and the high computational cost of volumetric context modeling.

To address these challenges, we propose **MedRWKV**, a lightweight 3D encoder-decoder segmentation network that integrates efficient long-range dependency modeling and boundary-aware deformable feature fusion.

The core idea is to combine:

1. **Tri-Directional 3D RWKV modeling** for efficient volumetric global context extraction.
2. **Boundary-driven deformable fusion** to align global semantic representations with local boundary-sensitive features.
3. **Hierarchical feature interaction** for bidirectional semantic-detail communication across multiple encoder stages.

---

## ✨ Key Contributions

- **Tri-Directional RWKV for 3D Volumes**  
  We extend Vision-RWKV to volumetric medical image segmentation by scanning 3D features along multiple spatial orders, enabling efficient long-range dependency modeling while reducing directional bias.

- **Boundary-Driven Deformable Fusion**  
  We introduce a boundary-aware deformable fusion module that extracts high-frequency boundary cues, predicts spatial offsets, and adaptively aligns global RWKV features with local convolutional features.

- **Hierarchical Semantic-Spatial Interaction**  
  A bidirectional feature interaction mechanism is adopted to perform top-down semantic guidance and bottom-up detail refinement across multi-scale encoder features.

- **Lightweight 3D Segmentation Framework**  
  The network combines depthwise separable 3D convolutions, anisotropic downsampling, RWKV-based global modeling, and deep supervision for efficient volumetric segmentation.

---

## 🏗️ Framework Architecture

The overall framework follows a U-shaped encoder-decoder structure.

```text
Input 3D Volume
      │
      ▼
Multi-stage 3D Encoder
      │
      ├── Tri-RWKV enhanced encoding
      │
      ├── Hierarchical feature interaction
      │
      ▼
Boundary-Driven Cross-Deformable Fusion
      │
      ├── High-frequency boundary extraction
      ├── Offset prediction
      ├── Deformable feature sampling
      └── Boundary-aware gated fusion
      │
      ▼
Lightweight 3D Decoder
      │
      ▼
Segmentation Prediction
