# Training

> Training pipeline for CSIRO biomass prediction, implementing a DINOv2-based model with GroupKFold validation, robust losses, mixed-precision training, and domain adaptation for unseen locations.

## Contents

| Item | Description |
| --- | --- |
| [Domain Adaptation](domain_adaptation.py) | Domain adaptation methods (MMD loss, adversarial training) to handle domain shift between training locations (NSW, VIC, QLD, SA) and new test locations. |
| [Advanced DINOv2 Training](train_dinov2_advanced.py) | Complete training pipeline with DINOv2-Base, GroupKFold by state/date, Huber loss, RAdam + Lookahead, multi-task uncertainty weighting, and FP16 mixed precision. |

## Related

- Parent: [`../`](../)

**Keywords:** model training, DINOv2, domain adaptation, MMD loss, adversarial training, GroupKFold, mixed precision, multi-task learning, biomass regression, PyTorch, self-supervised, CSIRO
