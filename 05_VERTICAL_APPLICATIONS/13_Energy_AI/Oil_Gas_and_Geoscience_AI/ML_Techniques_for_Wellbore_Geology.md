# ML Techniques for Wellbore Geology Prediction

Methods used to predict lithology, formation tops, and geology along a wellbore — from classical ML to modern deep learning and reinforcement learning for geosteering.

## 1. Classical ML (strong baselines)

- **Gradient-Boosted Trees** — XGBoost, **LightGBM**, CatBoost dominate tabular log data and most competitions; robust to mixed log suites and missing curves.
- **Random Forests / SVM** — solid lithofacies classifiers on conventional logs.
- **K-Means / GMM clustering** — unsupervised facies grouping ("electrofacies") when labels are scarce.
- **Feature engineering for logs:** depth/distance signals, rolling-window statistics, curve ratios (e.g., GR normalization), typewell-to-horizontal alignment, and **Discrete Wavelet Transform (DWT)** features for denoising/multi-scale structure.

## 2. Deep learning architectures

Logs are **depth sequences**, so sequence/convolutional models excel:

| Architecture | Why it's used | Note |
|---|---|---|
| **CNN / ResNet (1D & 2D)** | Capture local log patterns / scale-dependent geological texture. | Logs can be turned into images for 2D CNNs. |
| **LSTM / BiLSTM, GRU / BiGRU** | Capture **stratigraphic context** (a depth's facies depends on neighbors). | Standard for sequential logs. |
| **CNN-LSTM / CNN-BiLSTM / CNN-BiGRU** | Local features + long-range context. | Common winning combo. |
| **Transformers** | Long-range dependencies across the log; attention over depth & curves. | Increasingly used. |
| **Attention-hybrid (e.g., CA-HybridNet)** | 2D ResNet + BiLSTM + channel-spatial attention to focus on the most discriminative logs/intervals. | Geology-guided design. |

Reported results in the literature reach ~95% accuracy for common lithologies (claystone/marl/sandstone) from drilling parameters, and ~80–88% for harder/carbonaceous classes — depending on basin, label quality, and log suite.

## 3. Ahead-of-bit / look-ahead prediction

Predicting complex geology **ahead of the drill bit** combines:
- **GANs** to represent the earth model (geological uncertainty),
- **Forward Deep Neural Networks (FDNN)** as fast simulators,
- yielding **real-time estimates of geological uncertainty** to steer proactively. ([arXiv 2104.02550](https://arxiv.org/pdf/2104.02550))

## 4. Geosteering optimization with (Deep) Reinforcement Learning

- **Deep RL** dynamically adjusts the well trajectory in real time to **maximize reservoir contact and productivity**, treating geosteering as a sequential decision problem under uncertainty.
- Combine a learned **earth model** (perception) with an **RL policy** (control) for autonomous steering.

## 5. Practical modeling notes

- **Cross-validation by well / group** (GroupKFold by well) — avoid leaking the same well across folds.
- **Handle missing curves** — different wells log different suites; impute or train suite-robust models.
- **Normalize logs across wells** (especially GR) — tool/borehole effects cause shifts.
- **Class imbalance** — rare facies need weighting/resampling.
- **Depth alignment & leakage** — align horizontal wells to typewells carefully; "target-free" alignment avoids leaking the label.

## Sources
- [MDPI Eng — Real-Time Formation Lithology & Tops from Drilling Parameters](https://www.mdpi.com/2673-4117/4/3/139)
- [arXiv 2104.02550 — Deep Learning for Prediction of Complex Geology Ahead of Drilling](https://arxiv.org/pdf/2104.02550)
- [Frontiers — Geological-information-driven DL for lithology ID from well logs](https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2025.1662760/full)
- [Applied Sciences — Lithofacies Prediction from Well Logs (DL case study)](https://www.mdpi.com/2076-3417/14/18/8195)
- [Springer — Automated real-time formation tops, Norwegian Continental Shelf](https://link.springer.com/article/10.1007/s13202-024-01789-5)
- [Springer — Transforming petrophysical well-logs to images for lithology recognition](https://link.springer.com/article/10.1007/s12145-025-01953-3)
