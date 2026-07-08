# CDTM (Cross-Domain Transfer Module) + SFT (Staged Fine-Tuning)

## Description

The **Cross-Domain Transfer Module (CDTM)** and the **Staged Fine-Tuning (SFT)** strategy constitute a Transfer Learning approach proposed to improve medical image analysis by overcoming the limitation of scarce data in the medical domain. The method leverages models pre-trained on large natural image datasets (such as ImageNet-21K, LAION-400M, and LAION-2B) and efficiently adapts them to the medical domain. The CDTM acts as a feature adaptation module, transferring characteristics from the natural vision domain to the medical imaging domain. The SFT, in turn, is a two-stage fine-tuning strategy that optimizes the model's performance. The work was published in the *IEEE Journal of Biomedical and Health Informatics* in 2023/2024.

## Statistics

The **ConvNeXt-BCDTM-SFT** method achieved state-of-the-art results on the **BreakHis** dataset (histopathological breast cancer classification) at the time of publication. **Performance Metrics (BreakHis):** * Accuracy: 99.80% (compared to 99.40% for the previous strongest model, BACH). * F1-Score: 99.80% (estimated from the results figure). **Datasets Used:** * **Source Domain (Natural):** ImageNet-21K, LAION-400M, LAION-2B. * **Target Domain (Medical):** BreakHis (Breast Cancer Histopathology), HCRF (Gastric Histopathology). **Citations:** 9 (Google Scholar, as of Nov 2025).

## Features

**Cross-Domain Transfer Module (CDTM):** Component that facilitates the transfer of features from the natural image domain to the medical domain, adapting the pre-trained model. **Staged Fine-Tuning (SFT) Strategy:** A two-stage fine-tuning process (freezing the *backbone* followed by full fine-tuning) that optimizes performance and efficiency. **Compatibility with Vision Models:** Demonstrated effectiveness with CNN-based (ConvNeXt) and Transformer-based (ViT) architectures. **Leveraging Large Natural Datasets:** Uses the knowledge from models pre-trained on massive datasets such as LAION-2B and ImageNet-21K.

## Use Cases

**Histopathological Image Classification:** Detection and classification of breast cancer (BreakHis) and analysis of gastric histopathology (HCRF). **Computer-Aided Diagnosis (CAD):** Building high-performance CAD systems in data-scarce domains. **Rapid Model Adaptation:** Allows models pre-trained on large natural image datasets to be rapidly adapted to specific medical tasks with superior performance. **Reducing the Need for Large Medical Datasets:** Mitigates the challenge of collecting and annotating large volumes of medical data.

## Integration

The implementation is available on GitHub and uses the PyTorch framework. The training process involves: 1. **Data Preparation:** Downloading the medical datasets (e.g., BreakHis, HCRF) and creating CSV files for organization. 2. **Weight Download:** Using pre-trained weights of models such as ViT and ConvNeXt from the `timm` library. 3. **Training:** Running Python scripts with arguments to select the model mode (`vit` or `conv`), the fine-tuning mode (`linear`, `full`, `frt`), and the CDTM/SFT configuration. SFT requires an initial *backbone* freezing stage followed by a second fine-tuning stage using the weights from the first stage. **Example Command (ViT with SFT on BreakHis):** `python -m torch.distributed.launch --nproc_per_node=1 CODE/train.py --model-mode vit --finetune-mode frt --csv-dir malignant_all_5fold.csv --config-name 'config_clip_vit' --image-size 224 --epochs 100 --init-lr 1e-4 --batch-size 8 --num-workers 8 --val_fold 0 --test_fold 1 --data-root ./ --gpu_id 5` (This command starts the first stage of SFT).

## URL

https://github.com/qklee-lz/CDTM
