# AgriNet Models (VGG16, VGG19, Inception-v3, InceptionResNet-v2, Xception)

## Description

AgriNet is a set of models pre-trained on ImageNet architectures (VGG16, VGG19, Inception-v3, InceptionResNet-v2, and Xception) that were fine-tuned and trained on the AgriNet dataset. The AgriNet dataset is a massive collection of 160,000 agricultural images from more than 19 geographic locations, covering over 423 classes of plant species and diseases, pests, and weeds. The main goal is to provide domain-specific models to overcome the limitation of data and the absence of plant-specific pre-trained models, which are common challenges in agricultural automation. The work demonstrates the superiority of AgriNet models compared to the base ImageNet models on agricultural tasks.

## Statistics

- **Classification Accuracy:** AgriNet-VGG19 achieved the highest accuracy of 94% and the highest F1-score of 92% in classifying the 423 classes.
- **Minimum Accuracy:** All AgriNet models had a minimum accuracy of 87% (Inception-v3).
- **Dataset:** AgriNet dataset with 160,000 agricultural images and 423 classes.
- **Citations:** The original paper (arXiv:2207.03881) was cited 48 times (in 2022; the current number may be higher).
- **Publication:** Accepted by Frontiers in Plant Science.

## Features

- **Domain-Specific Pre-trained Models:** Set of models fine-tuned for the agricultural domain (AgriNet-VGG16, AgriNet-VGG19, etc.).
- **Comprehensive Dataset:** Uses the AgriNet dataset with 160k images and 423 classes.
- **Transfer Learning:** Applies the concept of Transfer Learning to accelerate training and improve accuracy on agricultural classification tasks.
- **Broad Class Coverage:** Classification of 423 classes of plant species, diseases, pests, and weeds.

## Use Cases

- **Plant Disease Detection:** Accurate identification of various diseases across different crops.
- **Plant Species Classification:** Differentiation of plant species in agricultural environments.
- **Pest and Weed Detection:** Identification of pests and weeds for crop management.
- **Agricultural Automation:** Provision of domain-specific computer vision models for automated monitoring and diagnosis systems.

## Integration

```python
# Example of Simulated Integration for AgriNet (Python/PyTorch-like)
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms

# 1. AgriNet Model Definition (Fine-tuning Simulation)
def load_agri_net_model(num_classes=423, model_name='vgg19'):
    if model_name == 'vgg19':
        model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        # Replace the final layer for the number of AgriNet classes
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    # ... (other architectures such as resnet50)
    
    # In practice, you would load the AgriNet-specific weights here:
    # model.load_state_dict(torch.load('agri_net_vgg19_weights.pth'))
    
    model.eval()
    return model

# 2. Image Preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 3. Inference Function (Simulated)
def predict_plant_disease(image_path, model, class_names):
    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted_idx = torch.max(output, 1)
    return class_names[predicted_idx.item()]

if __name__ == '__main__':
    simulated_classes = [f"Class_{i}" for i in range(423)]
    agri_net_model = load_agri_net_model(num_classes=len(simulated_classes), model_name='vgg19')
    print(f"AgriNet Model ({agri_net_model.__class__.__name__}) loaded successfully.")
    # To run the prediction, a plant image file is required.
```
**Note:** This is a conceptual example. The actual implementation requires the model weights and the exact list of classes, which must be obtained from the paper's official repository.

## URL

https://arxiv.org/abs/2207.03881
