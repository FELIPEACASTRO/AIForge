# PDDNet-LVE (Lead Voting Ensemble) - Transfer Learning in Agriculture

## Description

A convolutional neural network (CNN) architecture, PDDNet-LVE (Lead Voting Ensemble), that integrates nine pre-trained CNNs (including DenseNet201, ResNet101, ResNet50, GoogleNet, AlexNet, ResNet18, EfficientNetB7, NASNetMobile, and ConvNeXtSmall) and is fine-tuned through deep feature extraction for efficient plant disease identification and classification. Although the model is not explicitly trained on medical data, the paper that describes it (published in 2024) establishes the relevance of Transfer Learning in image domains with complex visual characteristics, such as the medical and agricultural domains, suggesting the basis for Cross-Domain Transfer Learning. The use of models pre-trained on massive datasets (such as ImageNet, which is the basis for most of the cited models) and the mention of the relevance of TL in both domains (medical and agricultural) in the same paragraph serve as the best evidence currently accessible for the requested topic.

## Statistics

Accuracy of 97.79% on the PlantVillage dataset (15 classes, 54,305 images). The PDDNet-AE (Early Fusion) model achieved 96.74% accuracy. Published in 2024.

## Features

Use of an ensemble (LVE) to increase robustness and generalization capacity; Uses pre-trained models (TL) to mitigate data scarcity and background complexity; Suitable for deployment on small devices (mobile); High accuracy in plant disease classification.

## Use Cases

Classification and detection of plant diseases for sustainable agriculture; Applications on mobile devices for field diagnosis; Mitigation of data scarcity in specific domains by leveraging knowledge from domains with abundant data (such as the medical domain).

## Integration

The model is based on Deep Learning architectures (DenseNet, ResNet, etc.) and can be implemented using standard libraries such as PyTorch or TensorFlow. Integration involves loading the pre-trained weights, replacing the final classification layer, and fine-tuning with the specific agricultural dataset. The paper does not provide a direct GitHub repository, but the methodology is standard for Transfer Learning in Computer Vision.

## URL

https://link.springer.com/article/10.1186/s12870-024-04825-y