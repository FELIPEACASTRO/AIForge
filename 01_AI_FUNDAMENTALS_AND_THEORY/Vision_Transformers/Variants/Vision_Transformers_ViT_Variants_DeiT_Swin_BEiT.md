# Vision Transformers (ViT) Variants: DeiT, Swin, BEiT

## Description

**DeiT (Data-efficient Image Transformers):** These are transformers trained more efficiently for image classification, requiring far less data and computational power than the original ViT. The main innovation is the **distillation** technique, in which a Transformer model is trained using an already-trained convolutional neural network (CNN) model (the "teacher") to guide the training of the Transformer (the "student"). This allows DeiT to achieve performance competitive with CNNs and ViTs trained on large datasets, using only ImageNet-1k. **Swin Transformer (Shifted Window Transformer):** This is a Vision Transformer that serves as a **general-purpose backbone** for computer vision. The main innovation is the introduction of a **hierarchical architecture** and a **shifted window attention** mechanism. It addresses the scaling and quadratic complexity problems of the original ViT, allowing the model to adapt to a wide range of vision tasks and scale to high-resolution images. **BEiT (Bidirectional Encoder representation from Image Transformers):** This is a **self-supervised** vision representation model that follows the **BERT-style** pre-training paradigm (Masked Language Modeling) for images. The pre-training approach is called **Masked Image Modeling (MIM)**. The model reconstructs the "visual tokens" of masked image patches. It transfers the success of self-supervised language pre-training (BERT) to vision.

## Statistics

**DeiT:** The reference model (86M parameters) achieves 83.1% top-1 accuracy (single-crop) on ImageNet-1k without external data. It can be trained on a single computer in less than 3 days. **Swin Transformer:** Creates hierarchical representations, with efficiency improved by the shifted window attention scheme. Supports vision models of up to 3 billion parameters (Swin Transformer V2). **BEiT:** Demonstrated significant improvements in downstream tasks, such as image classification and semantic segmentation, after MIM pre-training. It uses large datasets of unlabeled images for pre-training.

## Features

**DeiT:** Standard Transformer architecture, introduction of the **distillation token**, focus on **data efficiency** and **training efficiency**. **Swin Transformer:** **Shifted Window Attention** that limits attention to local windows but allows communication between windows in successive layers. **Hierarchical Architecture** that generates feature maps at different resolutions. **Versatility** as a backbone for various tasks. **BEiT:** **MIM Pre-training** (Masked Image Modeling) for reconstructing masked patches. Use of discrete **Visual Tokens** to apply the BERT paradigm to vision. Standard Transformer Encoder architecture.

## Use Cases

**DeiT:** Image Classification in scenarios with limited data or computational resources. Basis for other work exploring knowledge distillation in Vision Transformers. **Swin Transformer:** Object Detection and Instance Segmentation. Semantic Segmentation. Large-scale vision models. **BEiT:** Pre-training vision models for various downstream tasks. Image Classification. Semantic Segmentation (with BEiT V2). Applications in cancer diagnosis.

## Integration

**DeiT:** Available in **Hugging Face Transformers** (e.g., `transformers.DeiTModel`). Implementations in **PyTorch** and **TensorFlow/Keras**. **Swin Transformer:** Available in **Hugging Face Transformers**. Official implementations in **PyTorch**. **BEiT:** Available in **Hugging Face Transformers**. Official implementations in **PyTorch**.

## URL

DeiT: https://github.com/facebookresearch/deit | https://arxiv.org/abs/2012.12877 | Swin: https://github.com/microsoft/Swin-Transformer | https://arxiv.org/abs/2103.14030 | BEiT: https://www.microsoft.com/en-us/research/publication/beit-bert-pre-training-of-image-transformers/ | https://arxiv.org/abs/2106.08254