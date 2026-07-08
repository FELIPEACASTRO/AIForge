# StyleGAN3 (Alias-Free Generative Adversarial Networks)

## Description

StyleGAN3, or Alias-Free Generative Adversarial Networks, is the third iteration of NVIDIA's StyleGAN series, focused on solving the "texture sticking" problem in typical convolutional GANs. The problem is caused by aliasing due to careless signal processing, resulting in details that appear fixed to pixel coordinates instead of moving coherently with the object. StyleGAN3's unique value proposition is **full equivariance to translation and rotation**, even at subpixel scales, achieved through an "Alias-Free" architecture that treats all signals in the network as continuous. This makes it significantly better suited for generating content for video and animation, where the coherence of texture motion is crucial. The model is a direct evolution of StyleGAN2, maintaining image quality metrics (such as FID) while dramatically improving the internal representation.

## Statistics

**FID (Fréchet Inception Distance):** Matches the FID of StyleGAN2. **Equivariance:** Full to translation (EQ-T) and rotation (EQ-R) at subpixel scales. **Training Datasets:** FFHQ-U, MetFaces, AFHQv2, Beaches. **Resolution:** High-resolution image generation (e.g., 1024x1024). **Publication Year:** 2021.

## Features

Alias-Free architecture based on continuous signal processing principles; Full equivariance to translation (StyleGAN3-T) and rotation (StyleGAN3-R); Resolution of the "texture sticking" problem; Image generation where details transform coherently with the object; Improved suitability for video and animation generation.

## Use Cases

Content generation for video and animation with coherent texture motion and free of "texture sticking" artifacts; Editing of unaligned images across various domains; Creation of high-quality digital assets for media and entertainment.

## Integration

StyleGAN3 is officially implemented in PyTorch. Integration is typically done by loading the pre-trained models available in NVIDIA's official repository.

**Integration Example (Conceptual - PyTorch):**
```python
import torch
import stylegan3

# Load the pre-trained model (e.g., StyleGAN3-T trained on FFHQ)
# The model is loaded from a .pkl file
model_path = 'stylegan3-t-ffhq-1024x1024.pkl'
with open(model_path, 'rb') as f:
    G = torch.load(f)['G_ema'].cuda()

# Generate a random latent vector (z)
# The latent vector is the input to the generator
z = torch.randn([1, G.z_dim]).cuda()

# Generate the image
# The generator transforms the latent vector into an image
img = G(z, None)

# Post-processing and saving the image (tensor-to-image conversion)
# ...
```

## URL

https://nvlabs.github.io/stylegan3/
