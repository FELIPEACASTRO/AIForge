# 29 Sports Analytics AI

> AI applied to sport: player & ball tracking, pose estimation, possession-value and win-probability models, automated highlight/commentary generation, computer-vision officiating, and reinforcement-learning game simulators — built on rich public event and tracking datasets.

## Why it matters

Sport is one of the most data-rich verticals: every match produces synchronized video, event streams, and high-frequency positional tracking. AI turns this into decision support for clubs (recruitment, tactics, injury prevention), broadcasters (auto-highlights, commentary, graphics), and officials (offside, goal-line, ball-in/out). The community is unusually open — StatsBomb, SoccerNet, and the NFL Big Data Bowl release production-grade data — making it a strong sandbox for computer vision, spatiotemporal modeling, and multi-agent RL.

## Taxonomy

| Sub-area | What it does | Representative work |
|---|---|---|
| Tracking & detection | Multi-object tracking of players/ball/referees, jersey-number ID | SoccerNet-Tracking, TrackNet |
| Pose & biomechanics | 2D/3D pose for technique, load, injury risk | SoccerNet pose, basketball/3x3 pose |
| Possession value | Value of passes/dribbles/shots (xG, xT, VAEP) | socceraction, xG models |
| Win probability / outcome | In-game win prob, score/shot prediction | NFL Big Data Bowl, NBA trajectory models |
| Video understanding | Action spotting, dense captioning, highlight clips | SoccerNet action spotting & caption |
| Officiating CV | Offside, goal-line, line-calling, ball tracking | Hawk-Eye-style ball tracking (TrackNet) |
| Tactical / RL simulators | Agent learning, decision states, what-if tactics | Google Research Football |

## ⚽ Deep-dive sections

| Section | What's inside |
|---|---|
| [Football Match & Betting Prediction](./Football_Match_and_Betting_Prediction/) | The most complete open index of **football (soccer) match & betting prediction** — models (Dixon-Coles/Elo/xG/ML/DL), features, statistics & probability, global datasets & APIs, odds & value theory, tools, papers, and research platforms. Fact-checked, worldwide coverage, responsible-gambling framed (research/education only). |

## Key datasets & tools

| Name | Sport | Type | Link |
|---|---|---|---|
| StatsBomb open data | Football | Event data | https://github.com/statsbomb/open-data |
| SoccerNet | Football | Video benchmarks | https://github.com/SoccerNet/SoccerNet |
| SoccerNet-Tracking | Football | MOT video + jersey IDs | https://arxiv.org/abs/2204.06918 |
| Google Research Football | Football | RL environment | https://github.com/google-research/football |
| socceraction (VAEP / xT) | Football | Possession-value lib | https://github.com/ML-KULeuven/socceraction |
| kloppy | Football | Tracking/event loader | https://github.com/PySport/kloppy |
| NFL Big Data Bowl 2024 | NFL | Player tracking (Kaggle) | https://www.kaggle.com/c/nfl-big-data-bowl-2024 |
| TrackNet (ball/shuttle tracking) | Tennis/Badminton | CV model + code | https://github.com/yastrebksv/TrackNet |
| SoccerTrack | Football | Multi-view tracking | https://atomscott.github.io/SoccerTrack-v2/ |
| SoccerNet-V3 (Voxel51) | Football | Spatial annotations (HF) | https://huggingface.co/datasets/Voxel51/SoccerNet-V3 |

## Benchmarks

| Benchmark | Task | Link |
|---|---|---|
| SoccerNet Action Spotting | Localize goals/cards/subs in untrimmed video | https://www.soccer-net.org/ |
| SoccerNet Ball Action Spotting (2024) | Fine-grained ball-touch spotting | https://arxiv.org/abs/2409.10587 |
| SoccerNet-Tracking | Multiple object tracking (MOT) | https://arxiv.org/abs/2204.06918 |
| SoccerNet-Caption | Dense video commentary generation | https://arxiv.org/abs/2304.04565 |
| Google Research Football benchmarks | RL full-game scenarios (PPO/IMPALA/Ape-X) | https://arxiv.org/abs/1907.11180 |

## Key papers

| Year | Title | Link |
|---|---|---|
| 2018 | SoccerNet: A Scalable Dataset for Action Spotting in Soccer Videos | https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w34/Giancola_SoccerNet_A_Scalable_CVPR_2018_paper.pdf |
| 2019 | Google Research Football: A Novel Reinforcement Learning Environment | https://arxiv.org/abs/1907.11180 |
| 2019 | TrackNet: Deep Learning for Tracking High-speed and Tiny Objects in Sports | https://arxiv.org/abs/1907.03698 |
| 2016 | Applying Deep Learning to Basketball Trajectories | https://arxiv.org/abs/1608.03793 |
| 2022 | SoccerNet-Tracking: MOT Dataset and Benchmark in Soccer Videos | https://arxiv.org/abs/2204.06918 |
| 2023 | SoccerNet-Caption: Dense Video Captioning for Soccer Broadcasts | https://arxiv.org/abs/2304.04565 |
| 2023 | A Machine Learning Approach for Player- and Position-Adjusted Expected Goals | https://arxiv.org/abs/2301.13052 |
| 2024 | Deep Learning for Action Spotting in Association Football Videos | https://arxiv.org/abs/2410.01304 |
| 2023 | Boosting Multi-Agent RL on Google Research Football: Past, Present, Future | https://arxiv.org/abs/2309.12951 |

## Cross-references in AIForge

- [19 Computer Vision Applications](../19_Computer_Vision_Applications/) — detection, tracking, and pose backbones used here.
- [22 Gaming AI](../22_Gaming_AI/) — shared RL/simulation methods (Google Research Football overlaps both).
- [18 Predictive AI](../18_Predictive_AI/) — win-probability and outcome forecasting techniques.
- [09 Entertainment and Creative AI](../09_Entertainment_and_Creative_AI/) — automated highlights, commentary, and broadcast generation.

## Sources

- StatsBomb open data — https://github.com/statsbomb/open-data
- SoccerNet — https://github.com/SoccerNet/SoccerNet and https://www.soccer-net.org/
- SoccerNet-Tracking — https://arxiv.org/abs/2204.06918
- SoccerNet-Caption — https://arxiv.org/abs/2304.04565
- SoccerNet 2024 Challenges Results — https://arxiv.org/abs/2409.10587
- Google Research Football — https://github.com/google-research/football and https://arxiv.org/abs/1907.11180
- socceraction (VAEP/xT) — https://github.com/ML-KULeuven/socceraction
- kloppy — https://github.com/PySport/kloppy
- NFL Big Data Bowl 2024 — https://www.kaggle.com/c/nfl-big-data-bowl-2024
- TrackNet — https://arxiv.org/abs/1907.03698 and https://github.com/yastrebksv/TrackNet
- Player/position-adjusted xG — https://arxiv.org/abs/2301.13052
- Basketball trajectories (deep learning) — https://arxiv.org/abs/1608.03793

_Expanded from a verified high-value gap seed. Contributions welcome (see CONTRIBUTING.md)._
