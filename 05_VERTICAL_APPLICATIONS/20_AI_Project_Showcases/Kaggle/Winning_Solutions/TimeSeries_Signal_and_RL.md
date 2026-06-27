# Kaggle Winning Solutions — Time Series, Signal, Audio & Simulation/RL

> Curated index of verified, publicly-documented winning (and select top-3) Kaggle solutions across forecasting, signal/sensor classification, audio bioacoustics, and simulation/RL/optimization. Every entry links to a real public source (Kaggle discussion/write-up, official solution repo, arXiv, or author blog). Facts grounded via web research, 2026-06.

---

## 1. Demand & Sales Forecasting (tabular time series)

| Competition (year) | Rank / Team | Core approach | Link |
|---|---|---|---|
| **M5 Forecasting — Accuracy** (2020) | 1st — YeonJun Im (`YJ_STU`) | Equal-weighted (arithmetic mean) ensemble of many **LightGBM** models trained per-store, recursive multi-step; basic calendar + price features; non-recursive tweedie objective | [Results paper (PDF)](https://statmodeling.stat.columbia.edu/wp-content/uploads/2021/10/M5_accuracy_competition.pdf) · [Comp](https://www.kaggle.com/competitions/m5-forecasting-accuracy) |
| **M5 Forecasting — Accuracy** (2020) | 2nd | 50 models = 10 per store × 5 multipliers; **LightGBM** (calendar/price feats) blended with **N-BEATS** (sales-only) | [Kaggle forecasting paper, arXiv:2009.07701](https://arxiv.org/pdf/2009.07701) |
| **M5 Forecasting — Uncertainty** (2020) | 4th — marisakamozz | Quantile prediction with neural net; published code | [GitHub](https://github.com/marisakamozz/m5) · [Comp](https://www.kaggle.com/competitions/m5-forecasting-uncertainty) |
| **Web Traffic Time Series Forecasting** (2017) | 1st — Arthur Suilin (`Arturus`) | **seq2seq RNN (GRU encoder-decoder)** with attention-free conditioning; lagged (year-ago, quarter) features; SMAC-style cocob optimizer; ensemble of checkpoints/seeds; trained on TF | [GitHub (official)](https://github.com/Arturus/kaggle-web-traffic) · [Write-up](https://www.kaggle.com/competitions/web-traffic-time-series-forecasting/writeups/arthur-suilin-1st-place-solution) |
| **Rossmann Store Sales** (2015) | 1st — Gert Jacobusse | Ensemble of **20+ XGBoost** models adapted to time series; heavy temporal feature engineering on exogenous vars; **ridge-regression trend adjustment** | [Winner interview](https://medium.com/kaggle-blog/rossmann-store-sales-winners-interview-1st-place-gert-jacobusse-a14b271659b) |
| **Rossmann Store Sales** (2015) | 3rd — Neokami (Cheng Guo) | **Entity embeddings** of categorical vars + neural net — first top-3 NN in the comp; spawned the canonical embeddings paper | [arXiv:1604.06737](https://arxiv.org/pdf/1604.06737) · [GitHub](https://github.com/entron/entity-embedding-rossmann) |
| **Corporación Favorita Grocery Sales** (2018) | 1st — Sean Vasquez | Complex **LSTM seq2seq (encoder-decoder)** ensemble blended with gradient boosting; scheduled sampling decoder variants | [Discussion](https://www.kaggle.com/c/favorita-grocery-sales-forecasting/discussion/47582) |
| **Corporación Favorita Grocery Sales** (2018) | 2nd | **Dilated causal CNN (WaveNet-style)** for time series forecasting | [arXiv:1803.04037](https://arxiv.org/pdf/1803.04037) |
| **Corporación Favorita Grocery Sales** (2018) | 5th — LenzDu | 3 models: **GBM + CNN+DNN + seq2seq RNN**, weighted average | [GitHub](https://github.com/LenzDu/Kaggle-Competition-Favorita) |

**Forecasting takeaways**
- LightGBM (recursive / direct multi-horizon) is the default tabular forecaster; deep seq models (seq2seq RNN, N-BEATS, WaveNet/dilated CNN) win when raw history signal is rich.
- Multipliers / post-hoc trend correction (ridge, hand-tuned scalars) repeatedly add value at the top.
- Ensembling across seeds, stores, and model families is near-universal among winners.

---

## 2. Sequence / Physical-Process Time Series

| Competition (year) | Rank / Team | Core approach | Link |
|---|---|---|---|
| **Google Brain — Ventilator Pressure Prediction** (2021) | 1st (of 2,650) — Shujun He et al. | **Stacked LSTM + 1D-Conv + Transformer**; custom **ResidualLSTM** module (FFN + residual) to fix gradient issues; lag/diff feats, cumsum of `u_in`, one-hot R&C; + **PID matching** post-processing | [Winner blog](https://medium.com/data-science/winning-the-kaggle-google-brain-ventilator-pressure-prediction-2d4c90d831ec) · [GitHub](https://github.com/Shujun-He/Google-Brain-Ventilator) · [Notebook](https://www.kaggle.com/code/shujun717/1-solution-lstm-cnn-transformer-1-fold) |
| **Google Brain — Ventilator Pressure** (2021) | 1st team repo (Vandewiele) | Same team, alternate code release | [GitHub](https://github.com/GillesVandewiele/google-brain-ventilator) |

**Takeaway:** strongly autoregressive physical sequences reward LSTM backbones; transformers help global context but need conv/LSTM for local interactions. PID-aware post-processing exploited the deterministic simulator.

---

## 3. Signal / Sensor Classification

| Competition (year) | Rank / Team | Core approach | Link |
|---|---|---|---|
| **VSB Power Line Fault Detection** (2019) | 1st — Mark4h | **Bi-LSTM + Attention** on chunked statistical features of 3-phase PD signals; DWT-style denoising; metric MCC; notorious public→private shake | [1st-place notebook](https://www.kaggle.com/code/mark4h/vsb-1st-place-solution) · [Comp](https://www.kaggle.com/c/vsb-power-line-fault-detection) |
| **VSB Power Line Fault Detection** (2019) | top — braquino | Public **LSTM + Attention** baseline widely reused | [Notebook](https://www.kaggle.com/braquino/vsb-power-lstm-attention) |
| **LANL Earthquake Prediction** (2019) | 1st — Philipp Singer, Dmitry Gordeev et al. | Hillclimber blend of **LightGBM (fair loss) + SVR + multi-task NN**; only ~4 features (denoised peak counts, rolling-std percentiles, MFCC); KS-test distribution alignment of train→test; subsample 10/17 quake cycles | [1st-place write-up](https://medium.com/@ph_singer/1st-place-in-kaggle-lanl-earthquake-prediction-competition-15a1137c2457) |
| **LANL Earthquake Prediction** (2019) | 31st (ref) — viktorsapozhok | **DEAP genetic algorithm** feature selection + CatBoost (public reference repo) | [GitHub](https://github.com/viktorsapozhok/earthquake-prediction) |
| **G2Net Gravitational Wave Detection** (transient, 2021) | 1st — team KDL | **CWT/CQT spectrogram → 2D-CNN** ensemble + 1D-CNN on raw strain; signal whitening; ~0.88 AUC at top | [Comp](https://www.kaggle.com/competitions/g2net-gravitational-wave-detection) · [GWNET (top-6) arXiv ref](https://www.researchgate.net/publication/359051366_GWNET_Detecting_Gravitational_Waves_using_Hierarchical_and_Residual_Learning_based_1D_CNNs) |
| **G2Net Gravitational Wave Detection** (transient) | 6th/4th — GWNET (Dunlap et al.) | Hierarchical/residual **1D-CNN** on raw time series, blended with 2D-CNN spectrogram models | [GitHub](https://github.com/mddunlap924/G2Net_1D-CNN) |
| **G2Net Detecting Continuous Gravitational Waves** (2022–23) | 1st — Jun Koda | "**Summing the power with GPU**" — physics-driven GPU power-summing detection statistic (not pure CNN); $25k prize pool | [1st-place write-up](https://www.kaggle.com/competitions/g2net-detecting-continuous-gravitational-waves/writeups/jun-koda-1st-place-solution-summing-the-power-with) |
| **BHF Data Science Centre ECG Challenge** | active comp — Heartbeat/ECG signal classification | 12-lead ECG classification challenge (Kaggle-hosted) | [Comp](https://www.kaggle.com/competitions/bhf-data-science-centre-ecg-challenge/overview) |

**Signal takeaways**
- Two dominant paradigms: **(a) transform to spectrogram (CWT/CQT/STFT) → 2D-CNN**, and **(b) 1D-CNN / RNN on raw waveform**; top teams blend both.
- Distribution shift between train and test (LANL) is the real adversary — KS alignment + careful CV beat feature count.
- Domain physics (G2Net continuous GW power-summing) can outperform generic deep nets.

---

## 4. Audio Bioacoustics & Tagging (spectrogram + CNN/SED)

| Competition (year) | Rank / Team | Core approach | Link |
|---|---|---|---|
| **Cornell Birdcall Identification** (2020) | 1st — Ryan Wong | **PANNs-style Sound Event Detection (SED)** with attention; **DenseNet121** feature extractor on mel-spectrograms; weakly-supervised training, noisy-label losses (lsoft/lq) | [Write-up](https://ryaniswong.com/post/kaggle-bsr/) · [GitHub (official)](https://github.com/ryanwongsa/kaggle-birdsong-recognition) |
| **BirdCLEF 2023** | 1st — Volodymyr Sydorskyy ("Cailloux") — "Correct Data is All You Need" | Mel-spectrogram **CNN ensemble (EfficientNet / SED)**; heavy data curation, background-noise aug, pseudo-labeling | [Write-up](https://www.kaggle.com/competitions/birdclef-2023/writeups/volodymyr-1st-place-solution-correct-data-is-all-y) · [GitHub (official)](https://github.com/VSydorskyy/BirdCLEF_2023_1st_place) |
| **BirdCLEF 2022** | 1st — Henkel AI Team | Mel-spectrogram **CNN** (tf_efficientnet_b3_ns and friends); background-noise augmentation (became standard reference for later years) | [Discussion](https://www.kaggle.com/competitions/birdclef-2022/discussion/327108) |
| **Rainforest Connection Species Audio Detection (RFCX)** (2021) | top-2 (public repo) | **EfficientNet + ReXNet** ensemble on mel-spectrograms; 24-species multilabel | [Comp](https://www.kaggle.com/competitions/rfcx-species-audio-detection) · [2nd-place reimpl](https://github.com/WNoxchi/rfcx_species_audio_detection) |
| **Freesound Audio Tagging 2019** | 1st — Roman Solovyev (`lRomul`) | **Log-mel-spectrogram CNN** (Argus/PyTorch); 80-class multilabel with curated + noisy web labels; SR=44100, n_mels=128 | [GitHub (official)](https://github.com/lRomul/argus-freesound) · [README](https://github.com/lRomul/argus-freesound/blob/master/README.md) |

**Audio takeaways**
- The winning recipe is remarkably stable: **log-mel spectrogram → EfficientNet/DenseNet CNN**, often in an **SED framework** with attention for weak/clip-level labels.
- Augmentation (background noise from prior winners, mixup, SpecAugment) and **data cleaning/pseudo-labeling** decide the gold zone ("correct data is all you need").
- CQT is an occasional alternative to mel for high-frequency detail.

---

## 5. Simulation / Reinforcement Learning / Optimization

| Competition (year) | Rank / Team | Core approach | Link |
|---|---|---|---|
| **Lux AI Challenge — Season 1** (2021) | 1st — Toad Brigade | **Deep RL via IMPALA**; **ResNet + squeeze-excitation** policy over a 32×32 grid (per-unit action heads + win-probability head); led wire-to-wire; other teams imitation-learned its behavior | [Write-up](https://www.kaggle.com/competitions/lux-ai-2021/writeups/toad-brigade-toad-brigade-s-approach-deep-reinforc) · [Top-RL reference repo](https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021) |
| **Halite by Two Sigma** (2020) | 1st (of 1,143) — Tom Van de Wiele | **Deep RL with self-play**; spatial CNN policy over the board; segmentation-style per-cell action selection | [Discussion](https://www.kaggle.com/c/halite/discussion/166560) · [Imitation/seg approach (Voanh Kha)](https://voanhkha.github.io/2020/09/15/halite/) |
| **ConnectX** (ongoing) | top — community | **AlphaZero (MCTS + policy/value net)** and **minimax/alpha-beta** agents; AlphaZero reported top-10 of ~225 | [AlphaZero+RL study, arXiv:2210.08263](https://arxiv.org/pdf/2210.08263) · [Minimax notebook](https://www.kaggle.com/vatch123/connectx-minimax) |
| **Santa's Workshop Tour 2019** | top (MIP) — community | **Mixed Integer Programming (Gurobi)** for family→day assignment + simulated annealing / local search to escape local minima | [MIP write-up (Vandewiele)](https://medium.com/data-science/helping-santa-plan-with-mixed-integer-programming-mip-1951386a6ba5) |
| **Santa 2020 — The Candy Cane Contest** | 1st | **Integer Linear Programming** via FICO Xpress (interactive), with R + C++ tooling | [Comp list](https://www.kaggle.com/competitions?tagIds=15002-Optimization) |
| **Santa's Stolen Sleigh** (2015–16) | winners (deepsense) | Large-scale **route optimization** (TSP-like) with custom heuristics / metaheuristics | [deepsense write-up](https://deepsense.ai/blog/santas-stolen-sleigh-kaggles-optimization-competition/) · [Kaggle winner story](https://medium.com/kaggle-blog/defending-champions-winners-story-helping-santa-s-helpers-5984cc68efbb) |

**RL / optimization takeaways**
- Grid-world strategy games (Lux, Halite) are won by **CNN/ResNet policies trained with scalable RL (IMPALA / self-play PPO)**, frequently combined with imitation learning to bootstrap.
- Board games (ConnectX) favor **AlphaZero-style MCTS + minimax** hybrids.
- "Santa" combinatorial puzzles are pure **OR**: MIP/ILP solvers (Gurobi, FICO Xpress) plus simulated annealing / local search heuristics — ML is usually secondary.

---

## Cross-Competition Pattern Summary

| Problem family | Winning architecture pattern | Recurring tricks |
|---|---|---|
| Tabular forecasting | LightGBM (recursive/direct) ± seq2seq RNN / N-BEATS | per-segment models, multipliers, trend correction, seed ensembling |
| Autoregressive sequences | LSTM (+Conv +Transformer) | lag/diff feats, residual LSTM, simulator-aware post-proc (PID) |
| 1D signals | spectrogram→2D-CNN **and** 1D-CNN/RNN blend | wavelet denoising, train↔test distribution alignment |
| Audio bioacoustics | log-mel → EfficientNet/DenseNet, SED+attention | noise aug, mixup, pseudo-labeling, data cleaning |
| Sim / RL games | ResNet/CNN policy + IMPALA / self-play / AlphaZero-MCTS | imitation learning bootstrap, win-prob head |
| Combinatorial opt | MIP/ILP solver | simulated annealing, local search |

---

## Sources

- M5 Accuracy results paper (Makridakis et al.): https://statmodeling.stat.columbia.edu/wp-content/uploads/2021/10/M5_accuracy_competition.pdf
- Kaggle forecasting competitions survey (arXiv:2009.07701): https://arxiv.org/pdf/2009.07701
- M5 Uncertainty 4th place: https://github.com/marisakamozz/m5
- Web Traffic 1st (Arturus repo): https://github.com/Arturus/kaggle-web-traffic
- Web Traffic 1st write-up: https://www.kaggle.com/competitions/web-traffic-time-series-forecasting/writeups/arthur-suilin-1st-place-solution
- Rossmann 1st (Gert Jacobusse interview): https://medium.com/kaggle-blog/rossmann-store-sales-winners-interview-1st-place-gert-jacobusse-a14b271659b
- Entity embeddings (Rossmann 3rd, arXiv:1604.06737): https://arxiv.org/pdf/1604.06737
- Favorita discussion: https://www.kaggle.com/c/favorita-grocery-sales-forecasting/discussion/47582
- Favorita WaveNet (arXiv:1803.04037): https://arxiv.org/pdf/1803.04037
- Favorita 5th place: https://github.com/LenzDu/Kaggle-Competition-Favorita
- Ventilator 1st blog: https://medium.com/data-science/winning-the-kaggle-google-brain-ventilator-pressure-prediction-2d4c90d831ec
- Ventilator 1st repo: https://github.com/Shujun-He/Google-Brain-Ventilator
- VSB 1st-place notebook: https://www.kaggle.com/code/mark4h/vsb-1st-place-solution
- LANL 1st write-up (Philipp Singer): https://medium.com/@ph_singer/1st-place-in-kaggle-lanl-earthquake-prediction-competition-15a1137c2457
- G2Net (transient) competition: https://www.kaggle.com/competitions/g2net-gravitational-wave-detection
- G2Net 1D-CNN (GWNET) repo: https://github.com/mddunlap924/G2Net_1D-CNN
- G2Net continuous GW 1st (Jun Koda): https://www.kaggle.com/competitions/g2net-detecting-continuous-gravitational-waves/writeups/jun-koda-1st-place-solution-summing-the-power-with
- BHF ECG Challenge: https://www.kaggle.com/competitions/bhf-data-science-centre-ecg-challenge/overview
- Cornell Birdcall 1st (Ryan Wong): https://ryaniswong.com/post/kaggle-bsr/ · https://github.com/ryanwongsa/kaggle-birdsong-recognition
- BirdCLEF 2023 1st write-up: https://www.kaggle.com/competitions/birdclef-2023/writeups/volodymyr-1st-place-solution-correct-data-is-all-y
- BirdCLEF 2023 1st repo: https://github.com/VSydorskyy/BirdCLEF_2023_1st_place
- BirdCLEF 2022 1st (Henkel) discussion: https://www.kaggle.com/competitions/birdclef-2022/discussion/327108
- RFCX competition + 2nd-place reimpl: https://www.kaggle.com/competitions/rfcx-species-audio-detection · https://github.com/WNoxchi/rfcx_species_audio_detection
- Freesound 2019 1st (lRomul argus-freesound): https://github.com/lRomul/argus-freesound
- Lux AI S1 1st (Toad Brigade write-up): https://www.kaggle.com/competitions/lux-ai-2021/writeups/toad-brigade-toad-brigade-s-approach-deep-reinforc
- Lux AI top-RL reference repo: https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021
- Halite discussion: https://www.kaggle.com/c/halite/discussion/166560
- ConnectX RL/AlphaZero study (arXiv:2210.08263): https://arxiv.org/pdf/2210.08263
- Santa 2019 MIP write-up: https://medium.com/data-science/helping-santa-plan-with-mixed-integer-programming-mip-1951386a6ba5
- Santa's Stolen Sleigh (deepsense): https://deepsense.ai/blog/santas-stolen-sleigh-kaggles-optimization-competition/

---
_Curated via public-source research (Kaggle Discussions, official solution repos, arXiv, blogs). Verify any specific link before relying on it; gold write-ups live in each competition's Discussion tab._
