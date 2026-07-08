# Transfer Learning Guide

## Overview

Transfer Learning is a machine learning technique where a model developed for a task is reused as the starting point for a model on a second task. It is a popular and effective method in deep learning, especially in computer vision and natural language processing, where large pre-trained models can significantly reduce training time and improve performance on smaller, domain-specific datasets.

## Key Strategies (15+ Methods)

1. **Feature Extraction (Frozen Layers):**
   - **Method:** Use the pre-trained model as a fixed feature extractor. The weights of the pre-trained layers are frozen, and only the weights of the new classifier layer are trained.
   - **Use Case:** When the new dataset is small and similar to the original dataset.

2. **Fine-Tuning (Unfrozen Layers):**
   - **Method:** Unfreeze some or all of the pre-trained layers and train them along with the new classifier layer, usually with a very small learning rate.
   - **Use Case:** When the new dataset is large and/or significantly different from the original dataset.

3. **Domain Adaptation:**
   - **Method:** Techniques used to adapt a model trained on a source domain to perform well on a different but related target domain.
   - **Examples:** Adversarial Domain Adaptation, Maximum Mean Discrepancy (MMD).

4. **Layer-wise Fine-Tuning:**
   - **Method:** Different layers are trained with different learning rates, often using smaller rates for earlier layers (which capture general features) and larger rates for later layers (which capture specific features).

5. **Knowledge Distillation:**
   - **Method:** A smaller "student" model is trained to mimic the output of a larger, more complex "teacher" model. This is often used for model compression and deployment.

6. **Prompt Tuning / Adapter Layers:**
   - **Method:** For large language models (LLMs), small, trainable layers (adapters) are inserted between the layers of the frozen pre-trained model, or small "soft prompts" are learned. This is highly efficient.

## Transfer Learning in Medical Imaging

- **Challenge:** Medical datasets are often small and expensive to label.
- **Solution:** Pre-training on large general datasets (like ImageNet) or large medical datasets (like **RadImageNet**) and then fine-tuning on the specific medical task (e.g., tumor detection).

## Resources

- **ArXiv Paper: A Survey on Transfer Learning:** [https://arxiv.org/abs/1904.05045](https://arxiv.org/abs/1904.05045)
- **RadImageNet:** [https://github.com/BMEII-AI/RadImageNet](https://github.com/BMEII-AI/RadImageNet)
