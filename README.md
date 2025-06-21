# U-Net for Semantic Segmentation

U-Net is a convolutional neural network architecture designed primarily for semantic segmentation. It has gained popularity, especially in the biomedical imaging field, due to its ability to produce precise segmentations with limited data.

---

## 🧠 Introduction to U-Net

- U-Net is used for **semantic segmentation** — assigning a class label to each pixel.
- Originally proposed for biomedical image segmentation.
- Input image size in the original paper was **572x572**, but commonly resized to **256x256x1** for simplicity.
- Can handle:
  - Grayscale (1 channel)
  - RGB (3 channels)
  - Multispectral (10+ channels)

---

## 🔧 Basic U-Net Structure

### **1. Encoder (Contracting Path)**
Repeated blocks of:
- Two `Conv2D` layers (3×3 kernel, ReLU activation)
- One `MaxPooling2D` (2×2) to downsample
- Filters double at each level: `64 → 128 → 256 → 512 → 1024`

### **2. Decoder (Expanding Path)**
Repeated blocks of:
- `TransposeConv2D` or upsampling (doubles spatial dimensions)
- Concatenate with corresponding encoder output (skip connection)
- Two `Conv2D` layers (3×3 kernel, ReLU)

<p align="center">
  <img src="![image](https://github.com/user-attachments/assets/6eb3dad9-863a-4e0c-a8db-607f2022d75d)" alt="U-Net Architecture" width="600"/>
</p>
---

## 🏷️ Naming Conventions

Helps maintain clarity when coding:

- Encoder block:
  - `s1`: Output of convolution block
  - `p1`: Output of max pooling (input to next encoder block)
  - Pattern: `s2, p2, s3, p3, ...`
  
---

## 🧱 Defining Modular Blocks

### 🔹 Convolution Block: `conv_block`
- Input: Tensor, number of filters
- Layers:
  - `Conv2D` → `BatchNorm` (optional) → `ReLU`
  - `Conv2D` → `BatchNorm` → `ReLU`
- Output: Feature map (e.g., `s1`, `s2`, ...)

### 🔹 Encoder Block: `encoder_block`
- Calls `conv_block`
- Followed by `MaxPooling2D`
- Outputs: `conv_output`, `pooled_output`

### 🔹 Decoder Block: `decoder_block`
- Inputs: Previous decoder output, skip connection, number of filters
- Layers:
  - `TransposeConv2D` (or upsampling)
  - Concatenate with skip connection
  - Pass through `conv_block`

---

## 📏 Dimensions: Pooling & Upsampling

- **Downsampling (Max Pooling):**
  - 256 → 128 → 64 → 32 → 16
- **Upsampling (TransposeConv2D):**
  - 16 → 32 → 64 → 128 → 256
- **Filter progression:**
  - Down: 64 → 128 → 256 → 512 → 1024
  - Up: 1024 → 512 → 256 → 128 → 64

---

## ⚙️ Why Batch Normalization / Dropout?

- **Batch Normalization:**
  - Stabilizes and accelerates training
  - Not in original U-Net paper, but common in modern versions
- **Dropout:**
  - Regularization to prevent overfitting

---

## 🧩 Assembling U-Net

1. Input tensor: `256x256x1`
2. Pass through encoder blocks, storing `s1`, `s2`, ..., `p1`, `p2`, ...
3. Process through base (bottom of the "U")
4. Pass through decoder blocks, using skip connections from corresponding encoder stages
5. Final layer: output segmentation mask

---

## 🧬 Semantic Segmentation Overview

- **Definition:** Assign a class label to each **pixel** in the image.
- **Example use case:** Classify each pixel as foreground (e.g., mitochondria) or background.
- **Model Type:** Fully Convolutional Networks (FCNs)
  - Encoder: Extract features
  - Decoder: Restore resolution for pixel-wise prediction
- **U-Net:** Popular FCN architecture, especially for biomedical applications.

---

## 🏗️ Segmentation Models Library

**GitHub:** [`segmentation-models`](https://github.com/qubvel/segmentation_models)

- Ready-to-use architectures:
  - `U-Net`, `LinkNet`, `FPN`, `PSPNet`
- Customizable **encoders/backbones**:
  - `VGG16`, `VGG19`
  - `ResNet` (variants)
  - `Inception`, `EfficientNet`, `MobileNet`, etc.
- All pre-trained on **ImageNet** for faster convergence and higher accuracy.

---

## 🧪 Data Augmentation

- Use the **[albumentations](https://github.com/albumentations-team/albumentations)** library:
  - More powerful and flexible than Keras augmenters.
  - Apply identical transforms to both **images and masks**.
- ⚠️ Important:
  - Masks must retain original pixel values (e.g., `0` or `255`).
  - Avoid interpolation that changes segmentation class values.

---

## 📌 Summary

U-Net is a powerful and modular architecture for semantic segmentation. Its symmetric encoder-decoder structure with skip connections allows precise localization and classification of each pixel, making it ideal for tasks like medical image analysis, satellite image segmentation, and more.

