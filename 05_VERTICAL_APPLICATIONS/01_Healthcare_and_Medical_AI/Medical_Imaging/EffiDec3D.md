# EffiDec3D

## Description

EffiDec3D is a 3D decoder optimized for **high-performance and efficient 3D medical image segmentation**. It was proposed to address the high computational cost (high #FLOPs and #Params) of existing deep 3D networks, such as SwinUNETR and 3D UX-Net, which limit their use in real-time and resource-constrained environments. The model employs a **channel-reduction** strategy across all decoder stages and **removes high-resolution layers** when their contribution to segmentation quality is minimal. This approach sets a new standard for efficient 3D medical image segmentation, maintaining performance comparable to the original models but with a fraction of the computational resources.

## Statistics

- **Parameter Reduction (#Params):** 96.4% reduction compared to the original 3D UX-Net decoder.
- **Floating-Point Operations Reduction (#FLOPs):** 93.0% reduction compared to the original 3D UX-Net decoder.
- **Performance:** Maintains a performance level comparable to the original models (SwinUNETR, 3D UX-Net) across 12 different medical imaging tasks.
- **Publication:** Presented at **CVPR 2025** (Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition).

## Features

- **Optimized 3D Decoder:** Focused on computational efficiency.
- **Channel-Reduction Strategy:** Defines the minimum number of channels required for an accurate feature representation.
- **High-Resolution Layer Removal:** Eliminates layers with minimal contribution to segmentation quality.
- **Compatibility:** Can be integrated with existing encoders (e.g., SwinUNETR, 3D UX-Net).
- **Volumetric Segmentation:** Specialized in 3D medical image data.

## Use Cases

- **3D Medical Image Segmentation:** Primary application in volumetric segmentation tasks, such as identifying organs and anomalies in magnetic resonance imaging (MRI) and computed tomography (CT) scans.
- **Resource-Constrained Environments:** Ideal for deployment on edge devices or in systems that require real-time processing due to its high computational efficiency.
- **Research on DL Efficiency:** Serves as a *benchmark* for developing more efficient decoder architectures in 3D convolutional neural networks.

## Integration

The official PyTorch implementation is available on GitHub. The code includes training scripts for datasets such as BTCV and MSD (Task01-10), indicating that integration is achieved by using the EffiDec3D architecture within a PyTorch training pipeline, replacing the original decoder of models such as SwinUNETR or 3D UX-Net.
**Repository Link:** [https://github.com/SLDGroup/EffiDec3D](https://github.com/SLDGroup/EffiDec3D)

## URL

https://openaccess.thecvf.com/content/CVPR2025/html/Rahman_EffiDec3D_An_Optimized_Decoder_for_High-Performance_and_Efficient_3D_Medical_CVPR_2025_paper.html
