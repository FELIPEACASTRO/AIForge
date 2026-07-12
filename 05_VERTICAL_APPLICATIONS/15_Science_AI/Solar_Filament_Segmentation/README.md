# ☀️ Solar Filament Segmentation — Research Compendium

> The most complete, fact-checked index for **solar filament segmentation** and the techniques a winning solution needs — solar physics / H-alpha filament detection, **fine / curvilinear segmentation**, **instance separation** of touching thin structures, domain data, and evaluation metrics. Anchored to the **[IEEE Big Data Cup 2026 — Pixel-Precise Segmentation of Solar Filaments](https://bigdataieee.org/BigData2026/cup/)** (hosted on Kaggle, MAGFiLO dataset). Built from a multi-agent research sweep of **500+ live-verified sources** across arXiv, NASA ADS, Papers with Code, Semantic Scholar, OpenAlex, PubMed/Europe PMC, bioRxiv, DBLP and more.

## 📚 Pages

| Page | What's inside |
|---|---|
| [Competition, Metric & Domain Data](./Competition_Metric_and_Domain_Data.md) | The IEEE Big Data Cup + **MAGFiLO** (GONG Hα, 10,244 filaments), the scoring metrics (IoU, AP@IoU, hit/miss, Multi-scale IoU), and **all H-alpha / 304Å data archives** (GONG, BBSO, Kanzelhöhe, SDO/SDOML, CHASE, Meudon/BASS2000, MLSO, prominence catalogs, SunPy). |
| [Solar Filament Segmentation Methods](./Solar_Filament_Segmentation_Methods.md) | Solar-specific methods: **Flat U-Net**, **EdgeAttNet** (barb-aware), chirality classification, improved U-Nets, DETR + U-Net + tracking, semi-supervised (YOLOv5+U-Net), classical AAFDCC, 304Å prominence detection. |
| [Curvilinear / Fine Segmentation & Topology Losses](./Curvilinear_Fine_Segmentation_and_Losses.md) | Cross-domain thin-structure architectures (vessels, roads, cracks, neurons) + **topology-preserving losses**: clDice, cbDice, Skeleton Recall, centerline CE, persistent-homology / DMT / Betti-matching. |
| [Instance Separation Methods](./Instance_Separation_Methods.md) | Separating touching/overlapping/crossing thin structures: Deep Watershed, StarDist, Cellpose, orientation-aware + terminus pairing, CurvSegFlow, connectivity-preserving losses, Frenet-Serret. |
| [Metrics, Tools & Benchmarks](./Metrics_Tools_and_Benchmarks.md) | Metrics (IoU/mIoU, clDice, **Betti number & matching errors**, ARI/VOI, panoptic quality, HD95, + pitfalls) · tools/libraries (MONAI, GUDHI/Ripser, scikit-image/skan, SunPy) · cross-domain benchmarks (DRIVE/STARE/CHASE, roads, cracks, neurons). |
| [Surveys & Reviews](./Surveys_and_Reviews.md) | Review/survey articles across all axes — curvilinear/vessel segmentation, instance segmentation, topology-aware segmentation, and deep learning in solar physics. |
| [Where to Search — Scholarly Platforms](./Where_to_Search_Scholarly_Platforms.md) | A guide to *where* the subject lives across scholarly/preprint platforms (NASA ADS, arXiv, Papers with Code, Semantic Scholar, OpenAlex, PubMed/Europe PMC, bioRxiv, DBLP…), with representative verified hits per platform. |

## ☀️ Why solar filaments matter

Solar filaments (prominences seen on the disk) are cool, dense chromospheric structures suspended by magnetic fields. Their **eruption drives Coronal Mass Ejections**, solar flares, and Solar Energetic Particle storms — space-weather events that disrupt power grids, GPS, aviation, and satellites. Automatically and precisely segmenting them (including faint barbs and their magnetic chirality) at scale is a key input to space-weather forecasting — which is exactly what the IEEE Big Data Cup 2026 targets, using the MAGFiLO ground-truth built on the GONG H-alpha network.

## Related in AIForge
- Parent: [`../`](../) (Science AI) · Astronomy context: [`../Astrophysics_Prompts.md`](../Astrophysics_Prompts.md)
- Cross-domain technique twin (same curvilinear/instance/metric toolkit): [`../../20_AI_Project_Showcases/Kaggle/Biohub_Cell_Tracking/`](../../20_AI_Project_Showcases/Kaggle/Biohub_Cell_Tracking/)
- Fundamentals: [`../../../01_AI_FUNDAMENTALS_AND_THEORY/Computer_Vision/`](../../../01_AI_FUNDAMENTALS_AND_THEORY/Computer_Vision/)

**Keywords:** solar filament segmentation, H-alpha filament detection, prominence segmentation, MAGFiLO, GONG, IEEE Big Data Cup 2026, curvilinear segmentation, tubular structure segmentation, topology-preserving loss, clDice, cbDice, Betti error, instance separation, deep watershed, vessel segmentation, crack detection, road extraction, solar physics deep learning, space weather, chirality, spine barb detection, segmentation metrics.
