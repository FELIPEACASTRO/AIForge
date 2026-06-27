# Kaggle Winning Solutions — Computer Vision

Curated index of public, verifiable winning (and select top-tier) solutions from famous Kaggle computer-vision competitions. Each entry lists competition + year, final rank, team, key techniques (backbones, augmentation, TTA, ensembling, pseudo-labeling), and a real public link to the gold write-up or solution repo. Use this as a pattern bank: the recurring winning recipe across CV is **strong CV scheme + diverse backbone ensemble + heavy aug + TTA + pseudo-labeling on external data**.

> Scope: image classification, semantic/instance segmentation, object detection, medical imaging (CT/MRI/X-ray/histology), satellite/remote sensing, metric-learning/retrieval, and one audio-vision (spectrogram) case. Links point to canonical sources: official solution repos, Kaggle Discussion/Writeups, arXiv.

---

## Quick reference table

| Competition (year) | Rank | Team / author | Domain | Core method | Link |
|---|---|---|---|---|---|
| Cassava Leaf Disease (2021) | 1st | — | Classification | Diverse ensemble (EffNet/ViT/ResNeXt), TTA | [discussion](https://www.kaggle.com/competitions/cassava-leaf-disease-classification/discussion/221957) |
| Human Protein Atlas (2019) | 1st | bestfitting | Multi-label cls | DenseNet/ResNet, focal+metric, 4-channel | [discussion](https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/78109) |
| Planet: Amazon from Space (2017) | 1st | bestfitting | Multi-label cls | 11-CNN ensemble, soft-F2, label-corr, ridge | [interview](https://medium.com/kaggle-blog/planet-understanding-the-amazon-from-space-1st-place-winners-interview-bf66fb444bc2) |
| PetFinder Pawpularity (2021) | 1st | giba/RAPIDS SVR | Regression | Swin embeddings + SVR head, "magic" | [writeup](https://www.kaggle.com/competitions/petfinder-pawpularity-score/writeups/giba-rapids-svr-magic-1st-place-winning-solution-p) |
| Carvana Image Masking (2017) | 1st | Best[over]fitting | Binary seg | U-Net (VGG11/custom enc) ensemble, refine | [repo](https://github.com/asanakoy/kaggle_carvana_segmentation) |
| TGS Salt Identification (2018) | 1st | b.e.s. & phalanx | Seismic seg | U-Net + scSE, hypercolumn, pseudo-label | [repo](https://github.com/ybabakhin/kaggle_salt_bes_phalanx) |
| Severstal Steel Defect (2019) | 1st | bibimorph | Seg+cls | Classifier gate + U-Net/FPN ensemble | [writeup](https://www.kaggle.com/competitions/severstal-steel-defect-detection/writeups/1st-place-solution) |
| Data Science Bowl 2018 (nuclei) | 1st | [ods.ai] topcoders | Instance seg | Watershed U-Net ensemble, 32-net, style aug | [repo](https://github.com/selimsef/dsb2018_topcoders) |
| SIIM-ACR Pneumothorax (2019) | 1st | sneddy (Aimoldin) | Med seg | U-Net (se-resnext), triple-threshold, sampling | [repo](https://github.com/sneddy/pneumothorax-segmentation) |
| iMaterialist Fashion (2019) | 1st | amirassov | Instance seg | Mask R-CNN (Hybrid Task Cascade), attribute cls | [repo](https://github.com/amirassov/kaggle-imaterialist) |
| Sartorius Cell Instance Seg (2021) | top gold | tascj | Instance seg | YOLOX + UPerNet/Swin, LIVECell pretrain | [repo](https://github.com/tascj/kaggle-sartorius-cell-instance-segmentation-solution) |
| UW-Madison GI Tract (2022) | 1st | CarnoZhao | MRI seg | 2.5D, mmseg+MONAI, U-Net/UPerNet ensemble | [repo](https://github.com/CarnoZhao/Kaggle-UWMGIT) |
| HuBMAP – Hacking the Kidney (2021) | 1st | Tom (tikutikutiku) | Histology seg | U-Net (effnet/resnet) + pseudo-label, tiling | [repo](https://github.com/tikutikutiku/kaggle-hubmap) |
| HuBMAP+HPA Human Body (2022) | 3rd | Human Torus Team | Histology seg | Stain norm, scale, pseudo-label ensemble | [repo](https://github.com/VSydorskyy/hubmap_2022_htt_solution) |
| HuBMAP Human Vasculature (2023) | 3rd | Nischaydnk | Instance seg | Mask R-CNN/Cascade, multi-dataset, MMDet | [repo](https://github.com/Nischaydnk/HubMap-2023-3rd-Place-Solution) |
| PANDA Prostate Grade (2020) | 1st | kentaroy47 et al. | Histology cls | EffNet-B0/B1 tile-pooling, label noise handling | [repo](https://github.com/kentaroy47/Kaggle-PANDA-1st-place-solution) |
| SIIM-ISIC Melanoma (2020) | 1st | All Data Are Ext | Med cls | EffNet B3–B7 + meta, all-external data | [repo](https://github.com/haqishen/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution) |
| RANZCR CLiP Lines (2021) | 1st | (haqishen et al.) | Med multi-label | Seg-guided aux branch, ResNet200D/EffNet | [kernel](https://www.kaggle.com/code/haqishen/ranzcr-1st-place-soluiton-seg-model-small-ver) |
| RSNA Intracranial Hemorrhage (2019) | 1st | SeuTao | 3D CT seq | 2D CNN + sequence model (LSTM) | [repo](https://github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection) |
| RSNA Pneumonia Detection (2018) | 1st | i-pan (Ian Pan) | Detection | RetinaNet/Deformable + ensemble | [repo](https://github.com/i-pan/kaggle-rsna18) |
| RSNA Cervical Spine Fracture (2022) | 1st | — | 3D CT | 2.5D CNN + LSTM, seg-guided | [paper](https://pubs.rsna.org/doi/10.1148/ryai.230256) |
| RSNA Breast Cancer / Mammography (2023) | 1st | dangnh0611 | Med cls | ROI crop + ConvNeXt/EffNet, aux loss | [repo](https://github.com/dangnh0611/kaggle_rsna_breast_cancer) |
| RSNA Abdominal Trauma (2023) | 1st | Nischaydnk | 3D CT | 3D seg crop → 2D CNN+RNN per-organ | [repo](https://github.com/Nischaydnk/RSNA-2023-1st-place-solution) |
| TReNDS Neuroimaging (2020) | 1st | DESimakov | fMRI/sMRI reg | Feature eng + GBM/NN blend on 3D maps | [repo](https://github.com/DESimakov/TReNDS) |
| Global Wheat Detection (2020) | 1st | nvnn (dungnb1333) | Detection | EffDet + mosaic/mixup, pseudo-label, WBF | [repo](https://github.com/dungnb1333/global-wheat-dection-2020) |
| Vesuvius Ink Detection (2023) | 1st | ryches/ainatersol et al. | 3D seg | 3D-to-2D segformer ensemble, tiling | [repo](https://github.com/ainatersol/Vesuvius-InkDetection) |
| Google Research Contrails (2023) | 1st | Jun Koda (junkoda) | Satellite seg | U-Net-1024 + ViT ensemble, pixel-shift fix | [repo](https://github.com/junkoda/kaggle_contrails_solution) |
| Happywhale (2022) | 1st | Preferred Dolphin | Metric learning | ArcFace + EffNet/ConvNeXt, pseudo-label | [repo](https://github.com/knshnb/kaggle-happywhale-1st-place) |
| Google Landmark Retrieval (2019) | 1st | lyakaap (smlyaka) | Retrieval | Two-stage discriminative re-ranking, ArcFace | [repo](https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution) |
| Google Landmark Recognition (2020) | 1st | psinger/H2O | Retrieval/cls | ArcFace + sub-center, GLDv2, big-image FT | [repo](https://github.com/psinger/kaggle-landmark-recognition-2020-1st-place) |
| Google Landmark Recognition (2021) | 1st | ChristofHenkel | Retrieval/cls | Hybrid-Swin-Transformer + DOLG, ArcFace | [repo](https://github.com/ChristofHenkel/kaggle-landmark-2021-1st-place) |
| BirdCLEF 2023 | 1st | Volodymyr (Cailloux) | Audio→spectrogram | "Correct data" + CNN spectrogram ensemble | [repo](https://github.com/VSydorskyy/BirdCLEF_2023_1st_place) |

(Full detail per entry below.)

---

## 1. Image classification

### Cassava Leaf Disease Classification (2021) — 1st
- **Rank/Team:** 1st of ~3,900 teams.
- **Techniques:** Prioritized **architecture diversity over per-model tuning**. Ensemble across ResNeXt, **ViT/DeiT transformers**, and **EfficientNet** (B4) plus other CNNs. Noisy-label robustness (bi-tempered / label smoothing common in this comp); soft-voting; TTA. Reported private-LB ~0.913 from the ensemble vs ~0.895 single best.
- **Link:** [Kaggle 1st place discussion](https://www.kaggle.com/competitions/cassava-leaf-disease-classification/discussion/221957) · [top-1% reference repo (kozodoi)](https://github.com/kozodoi/Kaggle_Leaf_Disease_Classification)

### Human Protein Atlas Image Classification (2019) — 1st
- **Rank/Team:** 1st — **bestfitting**.
- **Techniques:** Multi-label subcellular localization; **4-channel (RGBY)** input; DenseNet/ResNet/Inception backbones; combined **focal + metric (macro-F1 oriented) loss**; external HPAv18 data; class-balance handling, TTA, ensemble.
- **Link:** [Kaggle 1st place discussion](https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/78109)

### Planet: Understanding the Amazon from Space (2017) — 1st
- **Rank/Team:** 1st — **bestfitting**.
- **Techniques:** Satellite multi-label tagging; **ensemble of 11 fine-tuned CNNs**; **soft-F2 loss**; **haze-removal preprocessing**; explicit **label-correlation model**; **two-level ridge regression** stacking; trust-local-CV under label noise.
- **Link:** [Kaggle 1st place winner's interview](https://medium.com/kaggle-blog/planet-understanding-the-amazon-from-space-1st-place-winners-interview-bf66fb444bc2) · [discussion](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/36809)

### PetFinder.my – Pawpularity Contest (2021) — 1st
- **Rank/Team:** 1st — giba & RAPIDS collaborators.
- **Techniques:** Image-quality regression; **Swin Transformer embeddings → RAPIDS SVR head** (the "SVR magic"); blending of CNN/transformer features with metadata; dense/binned-target trick.
- **Link:** [Kaggle 1st place writeup](https://www.kaggle.com/competitions/petfinder-pawpularity-score/writeups/giba-rapids-svr-magic-1st-place-winning-solution-p)

---

## 2. Semantic & instance segmentation

### Carvana Image Masking Challenge (2017) — 1st
- **Rank/Team:** 1st of 735 — team **"Best[over]fitting"** (Artsiom Sanakoyeu, Alexander Buslaev, Vladimir Iglovikov).
- **Techniques:** High-res car-mask binary segmentation; **ensemble of LinkNet + U-Net variants with VGG11 (TernausNet) and custom encoders**; boundary refinement; test-time refinement. Spawned **TernausNet** (U-Net + ImageNet VGG11 encoder).
- **Link:** [Official ensemble repo (asanakoy)](https://github.com/asanakoy/kaggle_carvana_segmentation) · [TernausNet](https://github.com/ternaus/TernausNet) · [winner's interview](https://medium.com/kaggle-blog/carvana-image-masking-challenge-1st-place-winners-interview-78fcc5c887a8)

### TGS Salt Identification Challenge (2018) — 1st
- **Rank/Team:** 1st — **b.e.s. & phalanx** (Yauhen Babakhin, Yuval).
- **Techniques:** Seismic salt-body segmentation; **U-Net + scSE blocks, hypercolumns**, deep supervision; **jigsaw mosaic** reconstruction of tiles; **semi-supervised pseudo-labeling**; cyclic LR / SWA; large ensemble. Published at GCPR 2019.
- **Link:** [Official repo (ybabakhin)](https://github.com/ybabakhin/kaggle_salt_bes_phalanx) · [Kaggle writeup with code](https://www.kaggle.com/competitions/tgs-salt-identification-challenge/writeups/b-e-s-phalanx-1st-place-solution-with-code)

### Severstal: Steel Defect Detection (2019) — 1st
- **Rank/Team:** 1st — bibimorph.
- **Techniques:** Multi-class defect segmentation; **classifier gate** (predict defect presence, zero-out negatives) feeding **segmentation ensemble (U-Net / FPN with se-resnext, efficientnet)**; threshold/min-area post-processing; pseudo-labeling; TTA.
- **Link:** [Kaggle 1st place writeup](https://www.kaggle.com/competitions/severstal-steel-defect-detection/writeups/1st-place-solution)

### 2018 Data Science Bowl — Nuclei Segmentation (2018) — 1st
- **Rank/Team:** 1st of 739 teams — **[ods.ai] topcoders** (Selim Seferbekov, Victor Durnov, Alexander Buslaev).
- **Techniques:** Nucleus instance segmentation across diverse microscopy; **watershed-energy U-Net ensemble** (resnet/densenet/inception encoders) predicting body + border masks; **32-network ensemble**; **style-transfer augmentation** to generalize to unseen experiments; heavy post-processing to separate touching nuclei. Published in *Nature Methods*.
- **Link:** [Official repo (selimsef)](https://github.com/selimsef/dsb2018_topcoders) · [1st place discussion](https://www.kaggle.com/c/data-science-bowl-2018/discussion/54741)

### SIIM-ACR Pneumothorax Segmentation (2019) — 1st
- **Rank/Team:** 1st — **sneddy** (Anuar Aimoldin).
- **Techniques:** Chest X-ray pneumothorax mask segmentation; **U-Net with se-resnext50/101 and (se)resnet encoders**; **triple-threshold decision rule** (top_score / min_contour_area / bottom_score) to suppress false positives; **adaptive positive/negative sampling** schedule (0.8→0.4); TTA + ensemble.
- **Link:** [Official repo (sneddy)](https://github.com/sneddy/pneumothorax-segmentation) · [1st place writeup](https://www.kaggle.com/competitions/siim-acr-pneumothorax-segmentation/writeups/dsmlkz-aimoldin-anuar-1st-place-solution-with-code)

### iMaterialist (Fashion) 2019 at FGVC6 — 1st
- **Rank/Team:** 1st — **amirassov** (Miras Amir).
- **Techniques:** Fine-grained clothing instance segmentation + attributes; **Hybrid Task Cascade (HTC) Mask R-CNN** (mmdetection) with strong backbone; class-balanced training; mask + attribute classification; multi-scale TTA.
- **Link:** [Official repo (amirassov)](https://github.com/amirassov/kaggle-imaterialist)

### Sartorius – Cell Instance Segmentation (2021) — gold (tascj)
- **Rank/Team:** Top gold — **tascj**.
- **Techniques:** Neuronal cell instance segmentation; **YOLOX-X detector + UPerNet/Swin segmentation**, both **pretrained on LIVECell** then fine-tuned on competition data; mask post-processing; strong external-data leverage.
- **Link:** [Solution repo (tascj)](https://github.com/tascj/kaggle-sartorius-cell-instance-segmentation-solution)

### UW-Madison GI Tract Image Segmentation (2022) — 1st
- **Rank/Team:** 1st — **CarnoZhao**.
- **Techniques:** Multi-organ MRI (stomach/large+small bowel) segmentation; **2.5D stacked slices**; mmsegmentation + MONAI; backbones (efficientnet/convnext) with U-Net/UPerNet decoders; 3D context via adjacent-slice channels; classification gate; TTA + ensemble.
- **Link:** [1st place repo (CarnoZhao)](https://github.com/CarnoZhao/Kaggle-UWMGIT)

---

## 3. Medical imaging (CT / MRI / X-ray / histology)

### HuBMAP – Hacking the Kidney (2021) — 1st
- **Rank/Team:** 1st — **Tom (tikutikutiku)**.
- **Techniques:** Glomeruli FTU segmentation on giga-pixel WSIs; **tiled U-Net (efficientnet/resnet encoders)**; **pseudo-labeling external kidney data**; stain/scale handling; heavy aug; sliding-window inference; ensemble.
- **Link:** [1st place repo (tikutikutiku)](https://github.com/tikutikutiku/kaggle-hubmap) · [discussion](https://www.kaggle.com/c/hubmap-kidney-segmentation/discussion/238198)

### HuBMAP + HPA – Hacking the Human Body (2022) — 3rd (Human Torus Team)
- **Rank/Team:** 3rd — **Human Torus Team** (V. Sydorskyy et al.).
- **Techniques:** FTU segmentation across 5 organs; **stain normalization, multi-scale**; **pseudo-labeling**; encoder-decoder ensemble (coat/convnext/efficientnet); domain-shift handling between HPA and HuBMAP.
- **Link:** [3rd place repo (VSydorskyy)](https://github.com/VSydorskyy/hubmap_2022_htt_solution)

### HuBMAP – Hacking the Human Vasculature (2023) — 3rd
- **Rank/Team:** 3rd — **Nischaydnk**. (See also tascj's public solution.)
- **Techniques:** Microvasculature **instance segmentation** (blood vessels) on kidney PAS tiles; **Mask R-CNN / Cascade R-CNN** (MMDetection) ensembles; multi-dataset training across annotation tiers; mask NMS / WBF-style fusion; TTA.
- **Link:** [3rd place repo (Nischaydnk)](https://github.com/Nischaydnk/HubMap-2023-3rd-Place-Solution) · [tascj solution repo](https://github.com/tascj/kaggle-hubmap-hacking-the-human-vasculature)

### PANDA – Prostate cANcer graDe Assessment (2020) — 1st
- **Rank/Team:** 1st of ~1,000 teams — **kentaroy47** & team.
- **Techniques:** Gleason-grade (ISUP) classification on prostate biopsy WSIs; **tile/patch extraction → tile-pooling EfficientNet-B0/B1** ("concat-tile pooling"); explicit **noisy-label handling** (Radboud vs Karolinska providers); QWK-oriented training; ensemble. Jumped from 22nd public (~0.910) to 1st private (~0.940), illustrating CV-over-LB trust.
- **Link:** [1st place repo (kentaroy47)](https://github.com/kentaroy47/Kaggle-PANDA-1st-place-solution)

### SIIM-ISIC Melanoma Classification (2020) — 1st
- **Rank/Team:** 1st — team **"All Data Are Ext"** (@boliu0, @haqishen, @garybios).
- **Techniques:** **EfficientNet B3–B7** at multiple resolutions (Chris Deotte's resized 2017–2020 ISIC data); **image + tabular metadata** dual-head; heavy aug; per-model 5-fold; large external-data emphasis ("all data are external"); careful CV vs LB; ensemble + TTA.
- **Link:** [Official repo (haqishen)](https://github.com/haqishen/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution) · [Kaggle writeup](https://www.kaggle.com/competitions/siim-isic-melanoma-classification/writeups/all-data-are-ext-1st-place-solution)

### RANZCR CLiP – Catheter & Line Position (2021) — 1st (public artifacts)
- **Rank/Team:** 1st place team; solution models shared publicly.
- **Techniques:** Multi-label catheter/line classification on CXR; **segmentation-guided** approach using line annotations to add an auxiliary mask branch; **ResNet200D / EfficientNet** classifiers; multi-stage training; TTA + ensemble.
- **Link:** [1st place seg-model kernel (haqishen)](https://www.kaggle.com/code/haqishen/ranzcr-1st-place-soluiton-seg-model-small-ver) · [1st place cls-model kernel](https://www.kaggle.com/tt195361/ranzcr-1st-place-solution-by-tf-4-cls-model)

### RSNA Intracranial Hemorrhage Detection (2019) — 1st
- **Rank/Team:** 1st — **SeuTao** (Tao Shen et al.).
- **Techniques:** **Two-stage 2D CNN (DenseNet121/169, SE-ResNeXt) → sequence model (LSTM/attention)** over CT slices to use inter-slice context; multiple windowing channels; multi-label (5 subtypes + any); TTA + ensemble.
- **Link:** [1st place repo (SeuTao)](https://github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection)

### RSNA Pneumonia Detection Challenge (2018) — 1st
- **Rank/Team:** 1st — **Ian Pan (i-pan)** & collaborators.
- **Techniques:** Chest X-ray opacity **object detection**; **RetinaNet / Deformable R-FCN** ensemble; bounding-box ensembling/shrinkage; classifier to suppress no-lung-opacity cases.
- **Link:** [1st place repo (i-pan)](https://github.com/i-pan/kaggle-rsna18) · [discussion](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/discussion/70421)

### RSNA 2022 Cervical Spine Fracture Detection — 1st
- **Rank/Team:** 1st of 883 teams. Winning algorithms documented in *Radiology: AI*.
- **Techniques:** **2.5D CNN + LSTM** for vertebra-level classification — z-axis slices per vertebra concatenated with neighbors and **segmentation masks** as multi-channel 2D input; **EfficientNet-V2 / ConvNeXt** backbones; per-vertebra LSTM head + patient-level aggregation; ensemble across backbones/folds.
- **Link:** [Winning-algorithms paper (Radiology: AI)](https://pubs.rsna.org/doi/10.1148/ryai.230256) · [5th-place open repo (Pfeiffer)](https://github.com/pascal-pfeiffer/kaggle-rsna-2022-5th-place)

### RSNA Screening Mammography Breast Cancer Detection (2023) — 1st
- **Rank/Team:** 1st of 1,687 teams — **dangnh0611**.
- **Techniques:** **YOLOX/region-based breast ROI crop** → high-res classifier (**ConvNeXt-small / EfficientNet**); aux losses (BiRADS, density, site); pos-class oversampling under heavy imbalance (pF1 metric); windowed DICOM preprocessing (dicomsdl/nvjpeg2k); TTA + ensemble.
- **Link:** [1st place repo (dangnh0611)](https://github.com/dangnh0611/kaggle_rsna_breast_cancer)

### RSNA 2023 Abdominal Trauma Detection — 1st
- **Rank/Team:** 1st — **Nischaydnk**.
- **Techniques:** Multi-organ injury grading on abdominal CT; **3-stage pipeline**: (1) **3D segmentation** (TotalSegmentator-style) to localize organs and generate crops; (2) **2D CNN + RNN** per-organ classifiers (kidney/liver/spleen/bowel) over slice sequences; (3) separate **bowel + extravasation** CNN+RNN; weighted log-loss aggregation; ensemble.
- **Link:** [1st place repo (Nischaydnk)](https://github.com/Nischaydnk/RSNA-2023-1st-place-solution)

### TReNDS Neuroimaging (2020) — 1st
- **Rank/Team:** 1st — **DESimakov** (Dmitry Simakov et al.).
- **Techniques:** Multi-target regression (age + assessment scores) from **3D sMRI/fMRI** brain-map features and provided FNC/loading features; heavy **feature engineering + GBM/linear/NN blending**; site-effect (domain) handling; weighted-NAE metric optimization.
- **Link:** [1st place repo (DESimakov)](https://github.com/DESimakov/TReNDS) · [discussion](https://www.kaggle.com/c/trends-assessment-prediction/discussion/162934)

---

## 4. Object detection

### Global Wheat Detection (2020) — 1st
- **Rank/Team:** 1st — **nvnn (dungnb1333)**.
- **Techniques:** Wheat-head detection across domains; **EfficientDet** (+ Faster/Cascade variants); **mosaic + mixup**, extensive Albumentations (CLAHE, blur, noise, color); external wheat datasets; **multi-round pseudo-labeling**; **Weighted Box Fusion (WBF)** + multi-scale TTA.
- **Link:** [1st place repo (dungnb1333)](https://github.com/dungnb1333/global-wheat-dection-2020)

---

## 5. Satellite / remote sensing & scientific imaging

### Vesuvius Challenge – Ink Detection (2023) — 1st
- **Rank/Team:** 1st — **ryches / ainatersol et al.** (Ryan Chesler, Ted Kyi, Alexander Loftus, Aina Tersol).
- **Techniques:** Detect ink in **3D X-ray micro-CT** volumes of carbonized Herculaneum papyri; treat z-layers as channels; **3D-aware → 2D segmentation (SegFormer / U-Net) ensemble** (~9 models); tiling + heavy aug; careful fragment-level CV; threshold tuning.
- **Link:** [1st place repo (ainatersol)](https://github.com/ainatersol/Vesuvius-InkDetection) · [1st place discussion](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/417496)

### Google Research – Identify Contrails to Reduce Global Warming (2023) — 1st
- **Rank/Team:** 1st — **Jun Koda (junkoda)**.
- **Techniques:** Aircraft-contrail **semantic segmentation** on GOES-16 infrared bands; **U-Net-1024 + ViT ensemble**; key insight on the **label/pixel half-shift** alignment issue; temporal frames as input; soft-label handling; TTA. Private LB ~0.724.
- **Link:** [1st place repo (junkoda)](https://github.com/junkoda/kaggle_contrails_solution) · [1st place writeup](https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming/writeups/jun-koda-1st-place-solution)

---

## 6. Metric learning, retrieval & re-identification

### Happywhale – Whale and Dolphin Identification (2022) — 1st
- **Rank/Team:** 1st — **Preferred Dolphin** (knshnb + charmq / Preferred Networks).
- **Techniques:** Individual re-ID via **ArcFace metric learning**; **EfficientNet-B6/B7 + ConvNeXt** backbones; fin/body **detector crop**; **multi-round pseudo-labeling (2 rounds)**; large embedding ensemble; kNN + new-individual threshold.
- **Link:** [1st place repo — knshnb](https://github.com/knshnb/kaggle-happywhale-1st-place) · [1st place repo — charmq](https://github.com/tyamaguchi17/kaggle-happywhale-1st-place-solution-charmq)

### Google Landmark Retrieval 2019 — 1st
- **Rank/Team:** 1st — **lyakaap** (smlyaka; also 3rd in the Recognition track).
- **Techniques:** **Two-stage discriminative re-ranking** for large-scale landmark retrieval under a noisy/diverse dataset; **ArcFace / cosine-softmax** global descriptors (FishNet/ResNet/SE backbones); DBA + query expansion; database-side augmentation. Paper: "Two-stage Discriminative Re-ranking for Large-scale Landmark Retrieval."
- **Link:** [1st & 3rd place repo (lyakaap)](https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution) · [paper (arXiv 2003.11211)](https://arxiv.org/abs/2003.11211)

### Google Landmark Recognition 2020 — 1st
- **Rank/Team:** 1st — H2O.ai (Philipp Singer/psinger et al.).
- **Techniques:** **ArcFace / sub-center ArcFace** on **GLDv2-clean**; backbone ensemble (EfficientNet/ResNeXt); **progressive fine-tuning on larger images**; loss-weight up-weighting of clean samples; global descriptor + re-ranking; out-of-domain (non-landmark) suppression.
- **Link:** [1st place repo (psinger)](https://github.com/psinger/kaggle-landmark-recognition-2020-1st-place) · [paper (arXiv 2010.01650)](https://arxiv.org/abs/2010.01650)

### Google Landmark Recognition 2021 — 1st
- **Rank/Team:** 1st (Christof Henkel / DSB team).
- **Techniques:** **Hybrid-Swin-Transformer** global descriptors with **DOLG (deep orthogonal local-global)**; **ArcFace** head over 81K+ classes; GLDv2; 8×V100 DDP; ensemble + re-ranking. Paper: "Efficient large-scale image retrieval with deep feature orthogonality and Hybrid-Swin-Transformers."
- **Link:** [1st place repo (ChristofHenkel)](https://github.com/ChristofHenkel/kaggle-landmark-2021-1st-place) · [paper (arXiv 2110.03786)](https://arxiv.org/abs/2110.03786)

---

## 7. Audio-vision (spectrogram CV)

### BirdCLEF 2023 — 1st
- **Rank/Team:** 1st — **Volodymyr (Cailloux team)**; thesis: *"Correct Data is All You Need."*
- **Techniques:** Bird-call classification framed as **CV on mel-spectrograms**; CNN backbones (EfficientNet/SED-style) on spectrogram images; **rigorous data cleaning / labeling**, background mixing, focal loss; heavy aug (mixup, time/freq masking); ensemble + TTA; ONNX inference under CPU time limits.
- **Link:** [1st place repo (VSydorskyy)](https://github.com/VSydorskyy/BirdCLEF_2023_1st_place) · [writeup "Correct Data is All You Need"](https://www.kaggle.com/competitions/birdclef-2023/writeups/volodymyr-1st-place-solution-correct-data-is-all-y)

---

## Cross-cutting patterns (the winning recipe)

- **CV scheme first.** bestfitting (Planet, HPA), PANDA (22nd public → 1st private), and SIIM melanoma winners all stress trusting a robust local CV over the public LB, especially under label/site noise.
- **Diverse-backbone ensembles** beat single-model tuning (Cassava, Carvana, Severstal, Landmark, DSB2018's 32-net). Mix CNN + transformer (ViT/Swin/ConvNeXt) where possible.
- **Pseudo-labeling on external data** is near-ubiquitous in golds: HuBMAP, Global Wheat, Happywhale, TGS Salt, Sartorius (LIVECell pretrain), melanoma (ISIC 2017–2019).
- **2.5D + sequence (CNN→RNN) models** dominate volumetric medical CT/MRI (RSNA ICH, RSNA C-Spine, RSNA Abdominal Trauma, UW-Madison GI).
- **Detector-then-crop ROI** boosts fine-grained medical classification (RSNA Breast, PANDA tiles, Happywhale fins) and trauma (organ crops).
- **Metric learning (ArcFace + variants)** is the standard for re-ID / landmark retrieval (Landmark 2019/2020/2021, Happywhale), usually with re-ranking / DBA on top.
- **Tiling + sliding-window** is the default for giga-pixel WSIs and 3D volumes (HuBMAP, Vesuvius, Carvana high-res).
- **TTA + box/mask fusion (WBF, multi-scale)** and threshold/post-processing tuning (Severstal, Pneumothorax triple-threshold) recover the last decimals.

---

## Sources

- Cassava Leaf Disease — https://www.kaggle.com/competitions/cassava-leaf-disease-classification/discussion/221957 ; https://github.com/kozodoi/Kaggle_Leaf_Disease_Classification
- Human Protein Atlas — https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/78109
- Planet Amazon — https://medium.com/kaggle-blog/planet-understanding-the-amazon-from-space-1st-place-winners-interview-bf66fb444bc2 ; https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/36809
- PetFinder Pawpularity — https://www.kaggle.com/competitions/petfinder-pawpularity-score/writeups/giba-rapids-svr-magic-1st-place-winning-solution-p
- Carvana — https://github.com/asanakoy/kaggle_carvana_segmentation ; https://github.com/ternaus/TernausNet ; https://medium.com/kaggle-blog/carvana-image-masking-challenge-1st-place-winners-interview-78fcc5c887a8
- TGS Salt — https://github.com/ybabakhin/kaggle_salt_bes_phalanx ; https://www.kaggle.com/competitions/tgs-salt-identification-challenge/writeups/b-e-s-phalanx-1st-place-solution-with-code
- Severstal Steel — https://www.kaggle.com/competitions/severstal-steel-defect-detection/writeups/1st-place-solution
- Data Science Bowl 2018 — https://github.com/selimsef/dsb2018_topcoders ; https://www.kaggle.com/c/data-science-bowl-2018/discussion/54741
- SIIM-ACR Pneumothorax — https://github.com/sneddy/pneumothorax-segmentation ; https://www.kaggle.com/competitions/siim-acr-pneumothorax-segmentation/writeups/dsmlkz-aimoldin-anuar-1st-place-solution-with-code
- iMaterialist Fashion 2019 — https://github.com/amirassov/kaggle-imaterialist
- Sartorius Cell — https://github.com/tascj/kaggle-sartorius-cell-instance-segmentation-solution
- UW-Madison GI Tract — https://github.com/CarnoZhao/Kaggle-UWMGIT
- HuBMAP Hacking the Kidney — https://github.com/tikutikutiku/kaggle-hubmap ; https://www.kaggle.com/c/hubmap-kidney-segmentation/discussion/238198
- HuBMAP+HPA Hacking the Human Body — https://github.com/VSydorskyy/hubmap_2022_htt_solution
- HuBMAP Hacking the Human Vasculature — https://github.com/Nischaydnk/HubMap-2023-3rd-Place-Solution ; https://github.com/tascj/kaggle-hubmap-hacking-the-human-vasculature
- PANDA Prostate — https://github.com/kentaroy47/Kaggle-PANDA-1st-place-solution
- SIIM-ISIC Melanoma — https://github.com/haqishen/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution ; https://www.kaggle.com/competitions/siim-isic-melanoma-classification/writeups/all-data-are-ext-1st-place-solution
- RANZCR CLiP — https://www.kaggle.com/code/haqishen/ranzcr-1st-place-soluiton-seg-model-small-ver ; https://www.kaggle.com/tt195361/ranzcr-1st-place-solution-by-tf-4-cls-model
- RSNA Intracranial Hemorrhage — https://github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection
- RSNA Pneumonia — https://github.com/i-pan/kaggle-rsna18 ; https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/discussion/70421
- RSNA 2022 Cervical Spine — https://pubs.rsna.org/doi/10.1148/ryai.230256 ; https://github.com/pascal-pfeiffer/kaggle-rsna-2022-5th-place
- RSNA Breast / Mammography — https://github.com/dangnh0611/kaggle_rsna_breast_cancer
- RSNA 2023 Abdominal Trauma — https://github.com/Nischaydnk/RSNA-2023-1st-place-solution
- TReNDS Neuroimaging — https://github.com/DESimakov/TReNDS ; https://www.kaggle.com/c/trends-assessment-prediction/discussion/162934
- Global Wheat Detection — https://github.com/dungnb1333/global-wheat-dection-2020
- Vesuvius Ink Detection — https://github.com/ainatersol/Vesuvius-InkDetection ; https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/417496
- Google Research Contrails — https://github.com/junkoda/kaggle_contrails_solution ; https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming/writeups/jun-koda-1st-place-solution
- Happywhale — https://github.com/knshnb/kaggle-happywhale-1st-place ; https://github.com/tyamaguchi17/kaggle-happywhale-1st-place-solution-charmq
- Google Landmark Retrieval 2019 — https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution ; https://arxiv.org/abs/2003.11211
- Google Landmark Recognition 2020 — https://github.com/psinger/kaggle-landmark-recognition-2020-1st-place ; https://arxiv.org/abs/2010.01650
- Google Landmark Recognition 2021 — https://github.com/ChristofHenkel/kaggle-landmark-2021-1st-place ; https://arxiv.org/abs/2110.03786
- BirdCLEF 2023 — https://github.com/VSydorskyy/BirdCLEF_2023_1st_place ; https://www.kaggle.com/competitions/birdclef-2023/writeups/volodymyr-1st-place-solution-correct-data-is-all-y

---
_Curated via public-source research (Kaggle Discussions, official solution repos, arXiv, blogs). Verify any specific link before relying on it; gold write-ups live in each competition's Discussion tab._
