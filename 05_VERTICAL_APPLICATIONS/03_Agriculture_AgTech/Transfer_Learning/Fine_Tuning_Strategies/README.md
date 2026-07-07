# AgTech Fine-Tuning Strategies

This directory covers fine-tuning strategies for agriculture AI models.

## Scope

- Full fine-tuning, linear probes, adapters, LoRA, self-supervised pretraining, domain adaptation, and low-data crop/disease transfer.
- Track source model, target crop, target region, label count, augmentation, and validation across farms/seasons.

## Reference Links

- PyTorch transfer learning tutorial: https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- Hugging Face PEFT: https://huggingface.co/docs/peft/index
- timm documentation: https://huggingface.co/docs/timm/index
- PlantVillage dataset: https://github.com/spMohanty/PlantVillage-Dataset

## Routing Rules

- Put multi-task transfer in sibling `Multi-Task_Learning/`.
- Put plant disease use cases in plant disease directories.
