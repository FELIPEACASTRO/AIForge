# Data Providers, Tracking Data & Computer Vision

> The professional **data-supply layer** beneath advanced football (soccer / futebol) prediction: who collects **event data** (dados de eventos), who collects **tracking data** (dados de rastreamento / posicionais), the **computer-vision** (visão computacional) pipelines that turn broadcast video into coordinates, the **open samples** researchers can actually download, and how spatial features (pitch control, xT, off-ball runs, pressing) become model inputs. Real URLs, free-vs-paid marks, worldwide coverage, current 2024–2026. **Research & education only.**

> ⚠️ **Not betting advice — research & education (pesquisa e educação).** Elite tracking/event data feeds the very models bookmakers and syndicates already use, so it is **priced into the market**: football closing lines are extremely efficient (Pinnacle closing odds correlated with observed outcomes at **r² ≈ 0.997 across 397,935 games** — [Trademate Sports analysis](https://tradematesports.medium.com/closing-line-the-most-important-metric-in-sports-trading-58e56cdb4458)). Better data does **not** guarantee an edge; most bettors lose money over time. Nothing here is a tip or a system. If gambling is a problem, get help: [BeGambleAware](https://www.begambleaware.org/) · [GamCare](https://www.gamcare.org.uk/) · [Gambling Therapy](https://www.gamblingtherapy.org/) · 🇧🇷 [Jogo Responsável (SPA/MF)](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas) · 🇧🇷 CVV **188**.

**Where this page sits:** datasets/APIs are indexed in [Global Datasets & Data APIs](./Global_Datasets_and_Data_APIs.md); OSS feature/modeling tools (socceraction, penaltyblog, soccerdata) in [Open-Source Tools](./Open_Source_Tools_and_Libraries.md); models in [Match Prediction Models](./Match_Prediction_Models_and_Techniques.md) and [Deep Learning](./Innovative_Models_and_Deep_Learning.md). **This page = the provider + tracking + CV layer that produces the raw material.**

**Price legend:** ✅ Free/open · 🆓 free sample or tier · ⬛ enterprise / no public price list.

---

## 1) The data pyramid — what each layer *is*

Prediction quality is bounded by input granularity. Four rungs, from cheapest/most-available to richest/most-restricted:

| Layer (EN / PT) | What one row/frame contains | Volume per match | Who has it |
|---|---|---|---|
| **Results + odds** (resultados + cotações) | final score, market prices | ~1 row | everyone (free CSVs) |
| **Aggregated stats** (estatísticas agregadas) | shots, xG, possession, PPDA | ~1 row/team | free scrapes (FBref/Understat/Sofascore) |
| **Event data** (dados de eventos) | every on-ball action w/ (x,y), player, outcome | **~1,600–3,400 events/game** | Opta, StatsBomb, Wyscout, PFF (paid) |
| **Tracking data** (dados de rastreamento) | x,y of **all 22 players + ball**, 10–25 Hz | **~2–4M position samples/game** | Second Spectrum, TRACAB, SkillCorner, Hawk-Eye, PFF (paid/optical) |

**Event data** answers *what happened & where*; **tracking data** adds *where everyone else was* — the off-ball context needed for pitch control, space/threat, pressing and physical load. **StatsBomb 360** (freeze-frames) and **Opta Vision** (event + real tracking) are hybrids. Almost all is enterprise-priced with **no public price list** — mark ⬛ below.

---

## 2) Event-data providers

Every on-ball action, hand-tagged or semi-automated from video, with (x,y), player, body part and outcome. This is the backbone of xG/xT/VAEP.

| Provider | What | Coverage | Price | Site |
|---|---|---|---|---|
| **Stats Perform / Opta** | Official event feed; **Opta Vision** = event **+ tracking** hybrid | 3,900+ competitions; official partner of EPL, Bundesliga, Serie A, La Liga, MLS | ⬛ | [statsperform.com/opta](https://www.statsperform.com/opta/) |
| **Hudl StatsBomb** | Event + **StatsBomb 360** freeze-frames; ~**3,400 events/match** | 190+ competitions; 360 in 40+ leagues | ⬛ (🆓 open-data tier) | [hudl.com/products/statsbomb](https://www.hudl.com/products/statsbomb) |
| **Hudl Wyscout** | Event + video scouting | 1,000+ competitions, wide global reach | ⬛ | [hudl.com/products/wyscout](https://www.hudl.com/products/wyscout) |
| **Hudl InStat** | Event + video (merged into Hudl 2022) | Global; football + other sports | ⬛ | [hudl.com](https://www.hudl.com/) |
| **PFF FC** | Event from **HD broadcast**; play-by-play grades | EPL + UCL (2024/25); EFL Championship/L1/L2 from 2025 | ⬛ (🆓 2022 WC sample) | [fc.pff.com](https://fc.pff.com/) |
| **Sofascore** | Aggregated stats, xG, ratings, momentum (**not** a raw event feed) | Global | Free site / unofficial API | [sofascore.com](https://www.sofascore.com/) |

> ⚠️ **StatsBomb 360 is not continuous tracking.** It stores a **freeze-frame** of visible player positions at the moment of each event — rich off-ball context, but no inter-event trajectories. For continuous 25 Hz movement you need true tracking (§3) or **Opta Vision** (event + tracking).

---

## 3) Tracking-data providers (all 22 players + ball, 10–25 Hz)

Two families: **in-stadium optical** (dedicated camera arrays, ground-truth accuracy) and **broadcast/monocular computer vision** (cheaper, wider coverage, off-camera players extrapolated).

| Provider | Method | Owner (2026) | Coverage | Price | Site |
|---|---|---|---|---|---|
| **Second Spectrum** | In-stadium optical, **25 Hz** | **Genius Sports** (acq. 2021) | Official tracking of EPL, MLS, NBA | ⬛ | [geniussports.com](https://www.geniussports.com/) |
| **TRACAB** | In-stadium optical (Gen5) + skeletal | **EA** (acq. 2025; spun out of ChyronHego 2021) | 300+ stadiums; Bundesliga, La Liga, ex-EPL | ⬛ | [tracab.com](https://tracab.com/) · [ea.com/tracab](https://www.ea.com/tracab) |
| **Hawk-Eye — SkeleTRACK** | Optical + **skeletal** (29 points); powers SAOT | **Sony** | FIFA/UEFA, many leagues & venues | ⬛ | [hawkeyeinnovations.com](https://www.hawkeyeinnovations.com/) |
| **SkillCorner** | **Broadcast / single-camera CV** (extrapolates off-camera) | Independent | 130+ competitions; Physical, Game Intelligence, XY | ⬛ (🆓 open sample) | [skillcorner.com](https://skillcorner.com/) |
| **PFF FC / Gradient Sports** | Broadcast tracking via **Sportlogiq** CV + PFF grades | PFF | 40+ competitions (2024/25) | ⬛ (🆓 2022 WC) | [fc.pff.com](https://fc.pff.com/) · [gradientsports.com](https://www.gradientsports.com/) |
| **Sportlogiq** | Broadcast CV (multi-sport), FIFA-certified | **Teamworks** (acq. Jan 2026) | Global football, hockey, American football | ⬛ | [sportlogiq.com](https://www.sportlogiq.com/) |
| **Signality** | Broadcast CV, 25 Hz | **Spiideo** (prev. IMG Arena) | Allsvenskan / Superettan (Sweden) | ⬛ | [spiideo.com](https://www.spiideo.com/) |
| **Metrica Sports** | Optical tracking + video analysis (PlayBase/Nexus) | Independent | Clubs worldwide | ⬛ (✅ 3-game open sample) | [metrica-sports.com](https://www.metrica-sports.com/) |

---

## 4) Open samples you can actually download (free teaching corpora)

The standard datasets for learning spatial modelling without an enterprise contract.

| Dataset | Type | Size / coverage | License | URL |
|---|---|---|---|---|
| **StatsBomb Open Data** | Event + **360** freeze-frames | Men's/Women's World Cups, UEFA Euro, UCL finals, La Liga (Messi years), FA WSL, Indian Super League | Free (attribution) | [github.com/statsbomb/open-data](https://github.com/statsbomb/open-data) · py [statsbombpy](https://github.com/statsbomb/statsbombpy) |
| **Metrica Sports sample** | **Tracking + event** | 3 anonymized full matches (CSV + FIFA-EPTS + JSON) | Free | [github.com/metrica-sports/sample-data](https://github.com/metrica-sports/sample-data) |
| **SkillCorner Open Data** | Broadcast **tracking** (CV) | 10 matches (Australian A-League 2024/25) + season aggregates | Free | [github.com/SkillCorner/opendata](https://github.com/SkillCorner/opendata) |
| **PFF FC 2022 World Cup** | **Broadcast tracking + event** | All **64** games + play-by-play grades | Free (form-gated) | [blog.fc.pff.com/…/enhanced-2022-world-cup-dataset](https://www.blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset) |
| **Bundesliga integrated dataset** (Nature 2025) | **Synced tracking (TRACAB) + event** | 7 Bundesliga 1./2. matches; frame-by-frame + hierarchical events | CC-BY 4.0 | [Nature s41597-025-04505-y](https://www.nature.com/articles/s41597-025-04505-y) · repo [spoho-datascience/idsse-data](https://github.com/spoho-datascience/idsse-data) · [Figshare](https://doi.org/10.6084/m9.figshare.28196177) |
| **Google Research Football** | RL **simulator** (physics 3D — not real-match data) | Football Benchmarks + Academy scenarios | Apache-2.0 | [github.com/google-research/football](https://github.com/google-research/football) |
| **SoccerNet** | **CV video** benchmark | 550+ broadcast games; **13 tasks** (see §5) | Research (NDA/EULA) | [soccer-net.org](https://www.soccer-net.org/) · [github.com/SoccerNet](https://github.com/SoccerNet) |

> Note: **Google Research Football** is a *reinforcement-learning environment* (Google Brain), **not** real tracking data — useful for multi-agent RL, not for predicting real fixtures. The **Bundesliga integrated dataset** (Bassek, Rein, Weber & Memmert, *Scientific Data* **12**:195, 2025) is the first open set of **official** synchronized position + event data.

---

## 5) SoccerNet — the computer-vision benchmark (13 tasks)

[SoccerNet](https://www.soccer-net.org/) is the reference academic benchmark for **broadcast-video understanding**, with an annual open challenge (CVPR CVSports workshop). The full family spans **13 distinct tasks** (each annual challenge uses a subset — e.g. 7 tasks in 2023, 4 in 2024):

1. **Action Spotting** · 2. **Ball Action Spotting** · 3. **Replay Grounding** · 4. **Camera Shot Segmentation/Boundary** · 5. **Dense Video Captioning** · 6. **Multi-View Foul Recognition** · 7. **Field (Pitch) Localization** · 8. **Camera Calibration** · 9. **Player/Ball Tracking** · 10. **Player Re-Identification** · 11. **Jersey Number Recognition** · 12. **Game State Reconstruction** (video → 2D minimap) · 13. **Monocular Depth Estimation**.

Task repos live under the [SoccerNet org](https://github.com/SoccerNet): [`sn-tracking`](https://github.com/SoccerNet/sn-tracking), [`sn-calibration`](https://github.com/SoccerNet/sn-calibration), [`sn-spotting`](https://github.com/SoccerNet/sn-spotting), [`sn-gamestate`](https://github.com/SoccerNet/sn-gamestate) (Game State Reconstruction, CVPRW'24).

---

## 6) The computer-vision pipeline & open libraries

Turning broadcast video into (x,y) coordinates is a chain: **detect → track → re-identify → calibrate/homography → assign teams → project to pitch**. Free OSS you can build this with:

| Stage | Library | Notes | URL |
|---|---|---|---|
| End-to-end soccer CV example | **roboflow/sports** | YOLOv8 player/ball/pitch detection + team classification (SigLIP + UMAP + KMeans) + RADAR minimap; MIT | [github.com/roboflow/sports](https://github.com/roboflow/sports) |
| Annotate / post-process detections | **roboflow/supervision** | Reusable CV utilities: annotators, zones, tracking glue | [github.com/roboflow/supervision](https://github.com/roboflow/supervision) |
| Multi-object trackers | **roboflow/trackers** | Clean re-impls of **SORT, ByteTrack, OC-SORT, BoT-SORT, C-BIoU**; Apache-2.0; speaks `supervision.Detections` | [github.com/roboflow/trackers](https://github.com/roboflow/trackers) |
| Core tracker (ByteTrack) | **ByteTrack** | ECCV 2022 "associate every detection box"; MOT17/20 SOTA | [github.com/ifzhang/ByteTrack](https://github.com/ifzhang/ByteTrack) |
| MOT/SOT/VID/VIS toolbox | **open-mmlab/mmtracking** | OpenMMLab video-perception toolbox — ⚠️ **frozen/deprecated**; MOT now lives in **MMDetection 3.x** | [github.com/open-mmlab/mmtracking](https://github.com/open-mmlab/mmtracking) |
| Modular sports-tracking framework | **TrackLab** | Detection + re-id + tracking pipeline; **backbone of SoccerNet Game State Reconstruction** | [github.com/TrackingLaboratory/tracklab](https://github.com/TrackingLaboratory/tracklab) |
| Standardize provider output | **kloppy** · **floodlight** | Vendor-independent load of StatsBomb/Opta/Wyscout/Sportec/Metrica/TRACAB/SecondSpectrum/SkillCorner/PFF; space/pitch-control primitives | [PySport/kloppy](https://github.com/PySport/kloppy) · [floodlight-sports/floodlight](https://github.com/floodlight-sports/floodlight) |

> **Hardest sub-problems** (per roboflow/sports): tiny fast **ball** detection, **jersey-number** OCR under blur/occlusion, and **identity persistence** through occlusions. Camera calibration/homography is what lets you project image coordinates onto a metric pitch — the step that makes the data comparable across broadcasts.

---

## 7) From tracking/CV to prediction features

Tracking + CV do not predict scorelines directly — they generate **spatial features** that feed the models on the sibling pages. The canonical ones:

| Feature (EN / PT) | What it measures | Reference |
|---|---|---|
| **Pitch control** (controle de campo) | P(a team controls the ball at each pitch location) from positions + velocities | Spearman, *Beyond Expected Goals* (MIT Sloan 2018) — [ResearchGate](https://www.researchgate.net/publication/327139841_Beyond_Expected_Goals) · explainer [Get Goalside](https://www.getgoalsideanalytics.com/everything-you-need-to-know-about-pitch-control/) |
| **Expected Threat (xT)** (ameaça esperada) | Value of possessing the ball in each zone via a Markov move/shoot model | Karun Singh (2018) — [karun.in/blog/expected-threat.html](https://karun.in/blog/expected-threat.html) |
| **OBSO** — Off-Ball Scoring Opportunity | P(scoring from the next on-ball action) incl. off-ball players; combines pitch control × ball-transition × score models | Spearman et al. (2017–18) |
| **Off-ball runs** (corridas sem bola) | Timing/type/threat of runs made *without* the ball | [SkillCorner Game Intelligence](https://skillcorner.com/) |
| **Pressing / PPDA** | Defensive intensity: passes allowed per defensive action, pressure events | derived from event+tracking |
| **Physical load** (carga física) | Distance, sprints, accel/decel by speed zone | tracking-native (SkillCorner Physical, TRACAB) |

**Reproducible teaching implementation:** Laurie Shaw's *Friends of Tracking* code ([`LaurieOnTracking`](https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking), MIT) builds velocities → **pitch control** → **EPV** directly on the Metrica open sample — the standard on-ramp. Standardize any provider first with **kloppy**; value on-ball actions (VAEP/xT) with **socceraction** (see [Open-Source Tools](./Open_Source_Tools_and_Libraries.md)).

**How it feeds prediction:** spatial features become inputs to (a) **shot/xG** models, (b) **action-value** models (VAEP/xT) for team-strength ratings, and (c) **possession/next-goal** and in-play win-probability models ([In-Play & ML Approaches](./In_Play_Advanced_and_ML_Approaches.md)). They sharpen *description and valuation*; they do **not** manufacture a betting edge — the market already prices this information (see reality check below).

---

## 8) Reality check & responsible use (uso responsável) — mandatory

- **This data is already in the price.** The providers above supply clubs, media, and the trading desks that set sharp lines. By kickoff, Pinnacle-type closing odds correlate with outcomes at **r² ≈ 0.997 across 397,935 games** ([Trademate](https://tradematesports.medium.com/closing-line-the-most-important-metric-in-sports-trading-58e56cdb4458)). Richer inputs raise the *ceiling* of your model; they do not lower the **margin/overround (vig)** you must overcome. The honest skill metric is **Closing Line Value (CLV)** — do your prices repeatedly beat the close? ([Pinnacle: CLV](https://www.pinnacle.com/betting-resources/en/educational/what-is-closing-line-value-clv-in-sports-betting)).
- **Most bettors lose over time.** Treat every dataset here as **modelling practice / ML education**, not income.

| Region | Resource | Contact |
|---|---|---|
| 🌍 Global | [Gambling Therapy](https://www.gamblingtherapy.org/) | Online chat/forum, multilingual |
| 🇬🇧 UK | [BeGambleAware](https://www.begambleaware.org/) · [GamCare](https://www.gamcare.org.uk/) | National Gambling Helpline **0808 8020 133** (24/7) |
| 🇧🇷 Brazil | [Jogo Responsável — SPA/Ministério da Fazenda](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas) (Portaria SPA/MF nº 1.231/2024) | Apoio emocional **CVV 188** |

---

## Related in AIForge
- [Global Football Datasets & Data APIs](./Global_Datasets_and_Data_APIs.md) · [Open-Source Tools & Libraries](./Open_Source_Tools_and_Libraries.md) · [Features & Feature Engineering](./Features_and_Feature_Engineering.md) · [Match Prediction Models](./Match_Prediction_Models_and_Techniques.md) · [Innovative Models & Deep Learning](./Innovative_Models_and_Deep_Learning.md) · Parent vertical: [`../`](../) (Sports Analytics AI)

**Sources (all verified live, Jul 2026):** statsperform.com/opta · hudl.com/products/statsbomb · hudl.com/products/wyscout · fc.pff.com · gradientsports.com · sofascore.com · geniussports.com · tracab.com · ea.com/tracab · hawkeyeinnovations.com · skillcorner.com · sportlogiq.com · spiideo.com · metrica-sports.com · github.com/statsbomb/open-data · github.com/statsbomb/statsbombpy · github.com/metrica-sports/sample-data · github.com/SkillCorner/opendata · blog.fc.pff.com · nature.com/articles/s41597-025-04505-y · doi.org/10.1038/s41597-025-04505-y · github.com/spoho-datascience/idsse-data · doi.org/10.6084/m9.figshare.28196177 · github.com/google-research/football · soccer-net.org · github.com/SoccerNet (sn-tracking, sn-calibration, sn-spotting, sn-gamestate) · github.com/roboflow/sports · github.com/roboflow/supervision · github.com/roboflow/trackers · github.com/ifzhang/ByteTrack · github.com/open-mmlab/mmtracking · github.com/TrackingLaboratory/tracklab · github.com/PySport/kloppy · github.com/floodlight-sports/floodlight · github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking · researchgate.net/publication/327139841_Beyond_Expected_Goals (Spearman 2018) · karun.in/blog/expected-threat.html · getgoalsideanalytics.com · tradematesports.medium.com · pinnacle.com · begambleaware.org · gamcare.org.uk · gamblingtherapy.org · gov.br/fazenda (SPA)

**Keywords:** football data providers, tracking data, event data, computer vision soccer, dados de rastreamento, dados de eventos, visão computacional futebol, Opta Stats Perform, Opta Vision, StatsBomb 360, Wyscout, Second Spectrum, Genius Sports, TRACAB, EA Sports, Hawk-Eye SkeleTRACK, SkillCorner, PFF FC, Sportlogiq, Signality, Metrica Sports, SoccerNet, Google Research Football, Bundesliga integrated dataset, pitch control, expected threat xT, OBSO, off-ball runs, corridas sem bola, homografia, player tracking, ByteTrack, roboflow sports, TrackLab, kloppy, floodlight, previsão de futebol, apostas esportivas, jogo responsável, closing line value.
