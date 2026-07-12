# Competition, Metric & Domain Data

> The anchor challenge (IEEE Big Data Cup 2026), its ground-truth (MAGFiLO), the metrics that score it, and every H-alpha / 304Å data archive you can pull from to build, pretrain, and stress-test a solar-filament segmentation model.

## The competition

The IEEE Big Data Cup 2026 task **"Pixel-Precise Segmentation of Solar Filaments"** (hosted on Kaggle, built on MAGFiLO) asks for three things at once: recover **fine barbs**, separate **faint filament material from ground/instrument noise**, and return **each filament as one coherent instance** rather than a smear of fragments.

| Resource | What / why |
|---|---|
| [IEEE Big Data Cup 2026 — Pixel-Precise Segmentation of Solar Filaments (challenge page)](https://bigdataieee.org/BigData2026/cup/solar-filament-segmentation/) | Full task definition, metric panel, and timeline for the anchor challenge. |
| [IEEE Big Data Cup 2026 — Cup landing](https://bigdataieee.org/BigData2026/cup/) | Top-level Cup page; the Kaggle-hosted competition with the custom scoring metric. |
| [IEEE Big Data Cup — Call + prior editions](https://bigdataieee.org/BigData2026/calls/big-data-cup/) | Format/rules lineage; prior editions establish the leaderboard and evaluation conventions the 2026 solar-filament cup follows. |
| [SWAN-SF (Space Weather ANalytics for Solar Flares)](https://ui.adsabs.harvard.edu/abs/2020harv.data..102A/abstract) | Not filaments — a model of a rigorous, reproducible solar-ML challenge dataset + evaluation protocol (SDO/HMI SHARP, Cycle 24) worth copying for splits and leakage control. |

## MAGFiLO — the challenge ground truth

MAGFiLO (**M**anually **A**nnotated **G**ONG **Fi**laments in H-a**L**pha **O**bservations): **10,244** manually annotated filaments across GONG H-alpha full-disk images (2011–2022), each with polygon mask, spine, bounding box and chirality, in COCO-style JSON.

| Resource | What / why |
|---|---|
| [MAGFiLO dataset — Harvard Dataverse](https://doi.org/10.7910/DVN/J6JNVK) | THE competition dataset: the exact COCO-style annotations (polygon, spine, bbox, chirality) the Cup is built on. Start here. |
| [MAGFiLO data descriptor — *Scientific Data*](https://www.nature.com/articles/s41597-024-03876-y) | Peer-reviewed descriptor: annotation pipeline, double-blind review, inter-annotator agreement (Kappa ≈ 0.66) and label semantics. Read first to understand ground-truth structure and known label noise. |
| [MAGFiLO project hub (MLEco / mlecofi.net)](https://www.mlecofi.net/magfilo) | Official landing page linking the Dataverse download, code repo, and viewer; part of the MLEco "Machine Learning Ecosystem" that produced both the dataset and the Cup. |
| [MAGFiLO data + code repository (Bitbucket)](https://bitbucket.org/dataresearchlab/mleco-magfilodatacode/) | Loader/utility code for parsing the COCO JSON, rendering masks/spines, and fetching per-filament GONG image URLs — a head start on the data pipeline. |
| [GONG H-Alpha Viewer (GSU DMLab)](https://dmlab.cs.gsu.edu/MLEco/GONGHAlphaViewer/) | MAGFiLO-team web tool to browse GONG H-alpha and inspect filament evolution day-by-day — handy for qualitative error analysis of segmentation output. |
| [Automatic Classification of Magnetic Chirality from H-alpha (Chalmers & Ahmadzadeh)](https://arxiv.org/abs/2509.18214) | First reproducible baseline built directly on MAGFiLO (chirality task); shows exactly how the labels are consumed and split, from the same lab as the Cup. |

## Other labeled filament datasets & catalogs

Independent labeled sets and long-baseline catalogs — for transfer, robustness, weak-label priors, and expected feature distributions beyond GONG's 2011–2022 window.

| Resource | What / why |
|---|---|
| [Automated Detection, Tracking & Analysis of CHASE Filaments via ML (Zhu et al.)](https://arxiv.org/abs/2402.14209) | U-Net + tracking on CHASE H-alpha with a **released annotated filament dataset** (via SSDC-NJU) and code — a second, independent labeled filament set for transfer/robustness. |
| [Statistical Analysis of Filament Features 1988–2013 (Hao, Fang & Chen — automated catalog)](https://arxiv.org/abs/1511.04692) | Nanjing-University automated BBSO/MLSO catalog spanning ~3 solar cycles: spine, area, length, tilt. Pre-deep-learning reference labels/method — good for weak-label priors. |
| [Statistical Analysis of Filament Features (published ApJS catalog)](https://iopscience.iop.org/article/10.1088/0067-0049/221/2/33) | Published version of the same large automated catalog — reference for which derived attributes to compute and their expected distributions (area, length, spine, tilt, hemispheric stats). |
| [A New Comprehensive Data Set of Solar Filaments of 100-yr Interval. I](https://arxiv.org/abs/2006.09082) | Century-scale filament catalog from historical H-alpha/spectroheliograph synoptic charts — long-baseline context and hemispheric/chirality statistics. |

## Prominence & eruption catalogs

On-disk filaments and their limb/eruption counterparts — machine-readable event catalogs for cross-referencing, weak labels, and downstream space-weather targets.

| Resource | What / why |
|---|---|
| [Catalog of prominence eruptions auto-detected in SDO/AIA 304Å (CDAW / Yang & Chen)](https://cdaw.gsfc.nasa.gov/CME_list/autope/) | Live, machine-readable 304Å prominence-eruption catalog (2-min cadence since 2010): date/time, position angle, latitude, width, movie links. |
| [A Catalog of Prominence Eruptions Detected Automatically in SDO/AIA 304Å (Yashiro et al.)](https://arxiv.org/abs/2005.11363) | Method paper behind the CDAW autoPE catalog (polar transform + intensity ratio + monotonic height rise) — a template for turning detections into eruption alerts. |
| [Prominence and Filament Eruptions Observed by SDO — Statistical Properties + Online Catalog (McCauley et al.)](https://arxiv.org/abs/1505.02090) | 904-event eruption catalog (type, symmetry, twist/writhe, kinematics, flare+CME association) from SDO/AIA via HEK — labeled eruption outcomes for downstream prediction. |
| [SMART/SDDI Filament Disappearance Catalogue (Seki et al.)](https://arxiv.org/abs/2003.03454) | Near-complete catalog of filament **disappearances** since 2016 from Hida SDDI 73-wavelength H-alpha, linked to flare, CME, 3D trajectory and Dst — direct disappearance↔space-weather labels. |

## H-alpha data archives (raw imagery)

Ground-based full-disk H-alpha sources. GONG is the MAGFiLO domain; the rest are independent instruments/eras for extra unlabeled pretraining data and cross-instrument generalization tests.

| Resource | What / why |
|---|---|
| [GONG H-alpha full-disk archive (NSO / NISP)](https://nso.edu/data/nisp-data/h-alpha/) | Primary source of the imagery underlying MAGFiLO: 2048×2048 full-disk 6562.8Å, ~1 image/20s network-wide from six sites. Pull extra unlabeled H-alpha for pretraining/augmentation. |
| [GONG H-alpha archive & query/FTP (gong2.nso.edu)](https://gong2.nso.edu/archive/patch.pl?menutype=hAlpha) | The searchable archive + FTP tree of science-grade fpack FITS and JPGs. Download by date/site to reconstruct or extend the MAGFiLO images. |
| [BBSO Full-Disk H-alpha Telescope archive (NJIT)](https://www.bbso.njit.edu/Research/FDHA/) | Contrast-enhanced full-disk H-alpha (~2″, up to 3 frames/min) — the source many classic filament catalogs trained on; a complementary domain for robustness testing. |
| [Kanzelhöhe Observatory (KSO) online H-alpha archive](http://cesar.kso.ac.at/halpha3a) | High-cadence (10/min) full-disk H-alpha FITS/JPEG plus scanned film back to 1973 — a large independent H-alpha domain for cross-instrument generalization and extra training data. |
| [Kanzelhöhe: instruments, data processing and products (Pötzi et al.)](https://arxiv.org/abs/2111.03176) | Reference for KSO products, processing, and archive layout; explains the FITS headers/calibration needed to align KSO with GONG. |
| [BASS2000 Solar Survey Archive — Meudon spectroheliograph (Obs. de Paris)](https://bass2000.obspm.fr/) | 100,000+ full-disk CaII K and H-alpha spectroheliograms over 10+ solar cycles (1536×1024 FITS) — a very-long-baseline, very-different-instrument H-alpha domain. |
| [CHASE / Solar Science Data Center of Nanjing University (SSDC-NJU)](https://ssdc.nju.edu.cn) | Level-1 full-disk H-alpha spectroscopy from the Chinese H-alpha Solar Explorer (HIS) — a modern space-based, ground-noise-free H-alpha domain, strong for domain-shift experiments. |
| [The Chinese Hα Solar Explorer (CHASE) mission: an overview (Li et al.)](https://arxiv.org/abs/2205.05962) | Describes CHASE/HIS data products and the SSDC-NJU access model — the reference for using CHASE H-alpha as an auxiliary filament dataset. |
| [Mauna Loa Solar Observatory (MLSO) archive — HAO/NCAR](https://mlso.hao.ucar.edu/) | SQL-searchable, VSO-registered archive including H-alpha disk/prominence monitors (PICS); another ground-based H-alpha domain used by classic MLSO filament studies. |

## SDO / AIA 304Å data (prominence-on-disk & multi-wavelength context)

Space-based 304Å shows filaments as bright prominences on the limb and dark absorption on the disk — plus HMI magnetograms for chirality/PIL context around each filament.

| Resource | What / why |
|---|---|
| [SDOML — ML-ready SDO dataset (Registry of Open Data on AWS)](https://registry.opendata.aws/sdoml-fdl/) | Curated AIA (all 10 bands incl. 304Å) + HMI, 512×512, degradation/exposure-corrected — the go-to source for 304Å prominence context to complement H-alpha filaments. |
| [SDOMLv2 preprocessing pipeline (GitHub)](https://github.com/SDOML/SDOMLv2) | Scripts to fetch/process SDO AIA/HMI/EVE into cloud-friendly `.zarr` (calibration-corrected). Reuse to build 304Å or HMI-magnetogram companion cubes aligned to GONG timestamps. |
| [SDO ML Dataset (Galvez et al. 2019)](https://arxiv.org/abs/1903.04538) | The descriptor paper for SDOML: ML-ready AIA/HMI/EVE imagery, 2010–2018 — the standard solar-imaging pretrain corpus and design reference. |
| [JSOC — Joint Science Operations Center (Stanford) data access](http://jsoc.stanford.edu/How_toget_data.html) | Authoritative SDO/HMI+AIA series and export system; documents series names and cutout/export options for building 304Å or magnetogram companions. |

## Federated search, event registries & viewers

One-stop retrieval across providers, event-level (often noisy) filament labels, and viewers for quick QA.

| Resource | What / why |
|---|---|
| [Virtual Solar Observatory (VSO)](https://docs.virtualsolar.org/) | Federated search/download across many providers (GONG H-alpha, MLSO, AIA/HMI, …) via one API (backs SunPy's Fido) — assemble multi-source H-alpha and context data. |
| [Heliophysics Event Knowledgebase (HEK) — LMSAL](https://www.lmsal.com/hek/api.html) | REST event registry with a filament (`FI`) event class (spines/bounding boxes from feature-finding modules) and filament eruptions — extra (noisy) labels and event cross-referencing. |
| [Helioviewer Project API (JPEG2000)](https://api.helioviewer.org/docs/v2/) | Fast tiled access to GONG H-alpha and SDO/AIA layers plus HEK overlays — great for screenshots and qualitative review dashboards of segmentation results. |
| [JHelioviewer — desktop viewer (ESA/NASA)](https://www.jhelioviewer.org/) | Open-source viewer over the same JPEG2000 layers; time-dependent browsing with event overlays for manual QA of detections across a filament's disk passage. |

## Data-access toolchain (Python)

The standard toolchain for loading MAGFiLO imagery and aligning multi-instrument data.

| Resource | What / why |
|---|---|
| [SunPy — Python for Solar Physics](https://github.com/sunpy/sunpy) | Core library: read H-alpha/AIA FITS (`Map`), unified search/download (`Fido`), coordinate frames and reprojection — the standard way to load and align solar images. |
| [drms — Python access to JSOC DRMS](https://github.com/sunpy/drms) | Query metadata and export/download HMI/AIA (incl. 304Å) directly from Stanford JSOC — pull magnetograms for chirality/PIL context around each filament. |
| [astropy](https://github.com/astropy/astropy) | The astronomy core under SunPy: `astropy.io.fits` for H-alpha FITS data/headers, `astropy.wcs` for world coordinates, units and frames — essential for correct ingest. |
| [aiapy](https://github.com/LM-SAL/aiapy) | SunPy-affiliated SDO/AIA tools: calibration, PSF deconvolution, degradation correction — for normalizing space-based imagery alongside ground-based H-alpha. |

## Scoring metrics (how the Cup grades submissions)

The Cup scores with **IoU**, per-pixel **precision/recall**, **AP@IoU** and **hit/miss rate** (each filament counted as a coherent detected instance), plus the organizers' custom **Multi-scale IoU** (defined on the challenge page above — no separate citation). The references below define that family; because pixel IoU is famously blind to thin-structure boundary and connectivity errors, treat boundary- and topology-aware scores as companions.

| Metric | What / why |
|---|---|
| [IoU / Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index) | The primary per-pixel overlap score (intersection/union); mIoU and the organizers' Multi-scale IoU are averaged variants. Central to the Cup but insensitive to thin-structure boundary/topology errors. |
| [Sørensen–Dice coefficient / Dice-F1](https://doi.org/10.2307/1932409) | Dice = 2·overlap / (|A|+|B|), equal to F1 on pixels and monotonic with IoU — the standard region-overlap score and the pixel-level precision/recall summary. |
| [COCO — Average Precision AP@IoU / mAP (Lin et al.)](https://arxiv.org/abs/1405.0312) | Defines AP-at-IoU-threshold and mAP over IoU 0.50:0.05:0.95 — exactly the AP@IoU / hit-rate / miss-rate family used to score each filament as a coherent detected instance. |
| [Panoptic Quality (PQ = SQ × RQ) (Kirillov et al.)](https://arxiv.org/abs/1801.00868) | The cleanest single number for "separate every touching/overlapping filament AND segment it well": Segmentation Quality (matched-instance IoU) × Recognition Quality (detection F1). |
| [Boundary IoU (Cheng et al.)](https://arxiv.org/abs/2103.16562) | Scores only a thin band around the boundary — far more sensitive to fine spine/barb contour errors than mask IoU without over-penalizing small objects. |

**Choosing & reporting the metric panel** (why the scoring choices matter for faint, thin filaments):

| Reference | What / why |
|---|---|
| [Metrics Reloaded — recommendations for image analysis validation (Maier-Hein, Reinke et al.)](https://www.nature.com/articles/s41592-023-02151-z) | Problem-fingerprint framework for choosing the *right* metric per task; use to justify the Cup's instance + semantic metric panel. |
| [Common Limitations of Image Processing Metrics: A Picture Story (Reinke et al.)](https://arxiv.org/abs/2104.05642) | Illustrated catalog of how Dice/IoU/HD mislead on small/thin targets, class imbalance, and boundary noise — a fast sanity-check for filament scoring. |
| [Pitfalls of Topology-Aware Image Segmentation (Berger et al.)](https://arxiv.org/abs/2412.14619) | Cautions on connectivity choices (4- vs 8-conn), topological artifacts in ground truth, and metrics that lack expressive power — actionable for fair connectivity-aware evaluation. |

> For the full evaluation toolkit — ARI/VOI, CREMI/Adapted-Rand, Betti number & matching errors, HD95, surface Dice, APLS, and their reference implementations (MONAI, TorchMetrics, GUDHI/Ripser, scikit-image) — see [Metrics, Tools & Benchmarks](./Metrics_Tools_and_Benchmarks.md).

## Related

- Sibling pages: [Solar Filament Segmentation Methods](./Solar_Filament_Segmentation_Methods.md) · [Curvilinear / Fine Segmentation & Topology Losses](./Curvilinear_Fine_Segmentation_and_Losses.md) · [Instance Separation Methods](./Instance_Separation_Methods.md) · [Metrics, Tools & Benchmarks](./Metrics_Tools_and_Benchmarks.md) · [Surveys & Reviews](./Surveys_and_Reviews.md) · [Where to Search — Scholarly Platforms](./Where_to_Search_Scholarly_Platforms.md) · Parent: [Solar Filament Segmentation](./README.md)

**Sources:** IEEE Big Data / bigdataieee.org · Harvard Dataverse · Nature *Scientific Data* · MLEco / mlecofi.net · NSO/GONG · BBSO/NJIT · Kanzelhöhe (KSO) · BASS2000 (Obs. de Paris) · SSDC-NJU / CHASE · MLSO (HAO/NCAR) · SDOML (AWS Open Data) · JSOC (Stanford) · VSO · HEK (LMSAL) · Helioviewer / JHelioviewer · SunPy / astropy / aiapy / drms · CDAW (NASA GSFC) · arXiv · IOPscience · *Nature Methods*.

**Keywords:** IEEE Big Data Cup 2026, solar filament segmentation competition, MAGFiLO dataset, GONG H-alpha, H-alpha full-disk archive, BBSO, Kanzelhöhe KSO, BASS2000 Meudon spectroheliograph, CHASE SSDC-NJU, MLSO, SDO AIA 304 dataset, SDOML, JSOC, Virtual Solar Observatory, HEK filament event, Helioviewer API, SunPy, drms, aiapy, prominence eruption catalog, filament disappearance catalogue, IoU Jaccard, Dice F1, AP@IoU mAP, panoptic quality, boundary IoU, multi-scale IoU, segmentation scoring metrics, space weather data.
