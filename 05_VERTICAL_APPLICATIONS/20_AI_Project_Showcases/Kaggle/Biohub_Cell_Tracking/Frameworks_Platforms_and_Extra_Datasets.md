# Frameworks, Platforms & Extra Datasets

> The tooling layer and additional public datasets for cell tracking during development — where you build pipelines, train models without heavy coding, view/annotate 3D+t data, and find more labeled data to pretrain on. All links verified live (July 2026).

## Analysis platforms & viewers

| Tool | Role | Link |
|---|---|---|
| **napari** | Fast n-dimensional image viewer; the hub for Python bioimage plugins (Cellpose, StarDist, btrack, Ultrack, tracking layers) | [napari.org](https://napari.org/) |
| **Fiji / ImageJ** | The standard bioimage platform; hosts TrackMate, Mastodon, MaMuT, Labkit | [fiji.sc](https://fiji.sc/) |
| **ilastik** | Interactive ML for pixel/object classification, segmentation & tracking | [ilastik.org](https://www.ilastik.org/) |
| **inTRACKtive** | Web-based interactive 3D track viewer (no install); virtual fate-mapping | [Nature Methods 2025](https://www.nature.com/articles/s41592-025-02777-1) · [github royerlab/inTRACKtive](https://github.com/royerlab/inTRACKtive) |
| **BIII** | Searchable registry of bioimage-analysis software (find any tool by task) | [biii.eu](https://biii.eu/) |

## Train models without deep coding

| Framework | What it gives you | Link |
|---|---|---|
| **ZeroCostDL4Mic** | Free Colab notebooks to train/apply U-Net, StarDist, CARE, Noise2Void, YOLO — no coding | [Nature Communications 2021](https://www.nature.com/articles/s41467-021-22518-0) · [github HenriquesLab/ZeroCostDL4Mic](https://github.com/HenriquesLab/ZeroCostDL4Mic) |
| **BiaPy** | Unified DL framework for 2D/3D bioimage tasks incl. instance segmentation & detection | [github BiaPyX/BiaPy](https://github.com/BiaPyX/BiaPy) |
| **DeepImageJ** | Run pretrained DL models inside ImageJ (no training) | [deepimagej.github.io](https://deepimagej.github.io/) |
| **BioImage Model Zoo** | Standard registry of shareable, runnable pretrained bioimage models | [bioimage.io](https://bioimage.io/) |

## Data engineering for 3D+t microscopy

- Images in this domain are typically **OME-Zarr / OME-NGFF** (chunked, cloud-friendly n-D arrays) read with **zarr** + **dask** for out-of-core processing. The Biohub competition ships OME-Zarr `(T,Z,Y,X)` stacks and a sparse `tracksdata` graph — see [Competition Overview & Scoring](./Competition_Overview_and_Scoring.md).

## Extra public datasets (pretrain / augment)

Beyond the on-distribution data in [Datasets, Metrics & Notebooks](./Datasets_Metrics_and_Notebooks.md):

| Dataset | Content | Link |
|---|---|---|
| **LIVECell** | 1.6M+ manually annotated cells, label-free phase-contrast — huge 2D segmentation training set | [Nature Methods 2021](https://www.nature.com/articles/s41592-021-01249-6) |
| **EVICAN** | 4,600 images / 26,000 cells across 30 cell types, multiple microscopes/modalities | [Bioinformatics 2020](https://academic.oup.com/bioinformatics/article/36/12/3863/5814923) |
| **Image Data Resource (IDR)** | Public repository of reference imaging studies (incl. segmentation/tracking datasets) via OMERO | [idr.openmicroscopy.org](https://idr.openmicroscopy.org/) |
| **Cell Tracking Challenge — benchmark results** | Latest leaderboard + linked training/challenge datasets (2D & 3D+t) | [celltrackingchallenge.net/latest-ctb-results](https://celltrackingchallenge.net/latest-ctb-results/) |

## Related

- [Segmentation Methods Compendium](./Segmentation_Methods_Compendium.md) · [Tracking Methods Compendium](./Tracking_Methods_Compendium.md) · [Surveys, Benchmarks & Key Papers](./Surveys_Benchmarks_and_Key_Papers.md) · [Datasets, Metrics & Notebooks](./Datasets_Metrics_and_Notebooks.md) · [Playbook README](./README.md)
- Parent: [`../`](../) (Kaggle)

**Sources:** [napari](https://napari.org/) · [Fiji](https://fiji.sc/) · [ilastik](https://www.ilastik.org/) · [inTRACKtive](https://github.com/royerlab/inTRACKtive) · [BIII](https://biii.eu/) · [ZeroCostDL4Mic](https://github.com/HenriquesLab/ZeroCostDL4Mic) · [BiaPy](https://github.com/BiaPyX/BiaPy) · [DeepImageJ](https://deepimagej.github.io/) · [BioImage Model Zoo](https://bioimage.io/) · [LIVECell](https://www.nature.com/articles/s41592-021-01249-6) · [EVICAN](https://academic.oup.com/bioinformatics/article/36/12/3863/5814923) · [IDR](https://idr.openmicroscopy.org/) · [CTC results](https://celltrackingchallenge.net/latest-ctb-results/)

**Keywords:** napari, Fiji ImageJ, ilastik, inTRACKtive, BIII, ZeroCostDL4Mic, BiaPy, DeepImageJ, BioImage Model Zoo, OME-Zarr, OME-NGFF, zarr dask, LIVECell, EVICAN, Image Data Resource IDR, OMERO, Cell Tracking Challenge datasets, bioimage analysis, no-code deep learning microscopy.
