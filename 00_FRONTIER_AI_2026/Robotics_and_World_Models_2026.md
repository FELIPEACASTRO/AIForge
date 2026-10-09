# 🛰️ Robotics, VLAs & World Models — 2026 Radar

> "Physical AI" — humanoid robots, vision-language-action (VLA) models and world models — is one of 2026's fastest-moving fronts, with OEMs moving from demos into mass production.

**Refreshed:** 2026-10-09 · **Scope:** releases through October 2026.

> ⚠️ Fast-moving field: entries reflect public reporting at refresh time. Always confirm versions, licenses, and benchmarks at the source links before relying on them. Benchmark and capability figures below are attributed to their source, not asserted as independently verified fact.

---

## 🤖 Humanoid robots — hardware & production

Newest first.

| Item | Date | What's new |
|---|---|---|
| [XPeng IRON humanoid production line](https://electrek.co/2026/09/07/xpeng-iron-humanoid-robot-production-line/) | 2026-09-07 | XPeng commissions a highly automated ("robots building robots", reported 80%+ automated) IRON humanoid line in Guangzhou; per the report the first unit walked off under its own power, with mass production targeted by year-end 2026. |
| [Unitree STAR Market IPO + H2 humanoid](https://www.caixinglobal.com/2026-07-03/unitree-robotics-wins-approval-for-618-million-star-market-ipo-102460136.html) | 2026-08-19 | First major humanoid-robot IPO: Unitree listed on Shanghai's STAR Market (ticker 688836), reported to close up ~460–487% on day one; flagship H2 humanoid plus low-cost R1 expand the lineup. |
| [Apptronik Apollo 2 + Robot Park (with Google DeepMind)](https://roboticsandautomationnews.com/2026/07/06/apptronik-launches-robot-park-to-train-apollo-humanoid-robots-with-google-deepmind/103069/) | 2026-07-06 | Apptronik opens a ~90,000 sq-ft "Robot Park" in Austin where Apollo 2 humanoids collect real-world data to train Gemini Robotics; Apollo 2 shown doing whole-body autonomy with Gemini Robotics 2. |
| [Figure 03 arrives at BMW](https://www.figure.ai/news/f-03-at-bmw) | 2026-06-30 | Third-generation Figure 03 humanoid (mass-producible, with tactile palms/cameras, built for the Helix stack) begins real production-line work at BMW; production ramp announced Apr 29, 2026. |
| [1X NEO home humanoid (Redwood AI)](https://www.1x.tech/discover/neo-home-robot) | 2026-05-01 | 1X begins production (Hayward, CA) of NEO, a consumer humanoid (reported ~$20K) running the onboard Redwood AI model with a human "Expert Mode" teleop fallback; first US deliveries slated before end of 2026. |
| [Tesla Optimus V3 (status: not yet revealed)](https://electrek.co/2026/04/22/tesla-optimus-production-fremont-model-sx-line/) | 2026-04-22 | As of Oct 2026 Tesla has not publicly unveiled Gen-3 Optimus; first-gen production lines are being installed on the former Model S/X line at Fremont. Radar-watch only — V3 specs/date are unverified. |
| [Boston Dynamics Atlas + Toyota Research Large Behavior Model (LBM)](https://www.tri.global/news/ai-powered-robot-boston-dynamics-and-toyota-research-institute-takes-key-step-towards-general) | 2025-08-20 | A single end-to-end Large Behavior Model directly controls Atlas's whole body for long-horizon, language-conditioned manipulation with on-the-fly disturbance recovery; electric Atlas tied to a FIFA World Cup 2026 debut. (Core LBM demo from Aug 2025.) |

---

## 🧠 VLAs & embodied foundation models

Newest first.

| Item | Date | What's new |
|---|---|---|
| [Figure Helix 2.5 — zero-shot 30-home generalization](https://www.figure.ai/news/helix-2-5-zero-shot-30-home-generalization) | 2026-09-17 | Figure's humanoid VLA "brain"; as reported by Figure, Helix 2.5 demonstrates zero-shot generalization across 30 never-seen homes — a step toward general-purpose home autonomy. Latest in the Helix System 0/1/2 line (Helix 02 was Jan 27, 2026). |
| [Skild AI S1 (robot foundation model)](https://www.therobotreport.com/skild-ai-unveils-s1-flagship-robot-foundation-model/) | 2026-08-31 | Skild AI unveils S1, a general-purpose "robot brain" that, per the report, learns new complex tasks (up to ~10 min) from a single demonstration video via in-context learning, and runs on quadrupeds, humanoids and arms. |
| [Google DeepMind Gemini Robotics 2 (VLA + ER 2 + On-Device 2)](https://deepmind.google/models/model-cards/gemini-robotics-er-2/) | 2026-07-30 | New Gemini Robotics family per the model card: a VLA action model, an ER 2 embodied-reasoning model (built on Gemini 3.5 Flash) and On-Device 2; adds whole-body humanoid control and the ASIMOV-Agentic safety benchmark. Demoed on Apptronik Apollo 2. |
| [Mistral Robostral Navigate](https://mistral.ai/news/robostral-navigate) | 2026-07-08 | Mistral's first embodied-navigation VLA (8B): per Mistral, it moves a robot from a single RGB camera + plain-language instruction, with no depth/LiDAR — a new robotics direction for Mistral (tech report arXiv:2607.20785). |
| [NVIDIA Isaac GR00T N1.7 (GA + LeRobot)](https://github.com/NVIDIA/Isaac-GR00T) | 2026-07-01 | Open VLA humanoid foundation model moves from early access to GA; per the repo it swaps in a Cosmos-Reason2 (Qwen3-VL) backbone and 20K hrs of EgoScale human-video pretraining, and ships via Hugging Face LeRobot. |
| [NVIDIA Alpamayo 2 Super (robotaxi reasoning VLA)](https://www.globenewswire.com/news-release/2026/06/01/3303988/0/en/NVIDIA-Launches-Alpamayo-2-Super-Open-Reasoning-Model-for-Robotaxis.html) | 2026-06-01 | 32B chain-of-thought reasoning VLA for L4 robotaxis (successor to the 10B Alpamayo 1 from CES Jan 2026); reasons/plans/acts across the driving stack, shipped with the AlpaGym closed-loop RL framework. |
| [Physical Intelligence π0.7 (steerable VLA)](https://www.pi.website/blog) | 2026-04-16 | Steerable robot foundation model (Gemma-3 4B VLM + 860M action expert) reported to achieve a step-change in compositional generalization via language subgoals; paired with Multi-Scale Embodied Memory (10+ min tasks). Per the blog index entry "π0.7: a Steerable Model with Emergent Capabilities." |
| [AgiBot Genie Operator GO-1 / GO-2 (ViLLA) + AgiBot World](https://agibot-world.com/blog/go1) | 2026-04-09 | Leading Chinese embodied foundation model using a Vision-Language-Latent-Action (ViLLA) + MoE framework trained on the open AgiBot World dataset; successor GO-2 and an updated 2026 dataset followed during AgiBot's April 2026 "AI Release Week" (GO-2 date via secondary coverage). |
| [NVIDIA Isaac GR00T N2 / DreamZero (world-action model, preview)](https://nvidianews.nvidia.com/news/nvidia-and-global-robotics-leaders-take-physical-ai-to-the-real-world) | 2026-03-16 | Next-gen robot foundation model based on DreamZero "world-action model" research (14B, image-to-video diffusion backbone); NVIDIA claims >2x success on new tasks/environments vs leading VLAs. Previewed at GTC; targeted for end of 2026 — not yet released as of Oct 9. |

---

## 🌍 World models & embodied datasets

Newest first.

| Item | Date | What's new |
|---|---|---|
| [World Labs Atlas (omni world model)](https://www.worldlabs.ai/blog/atlas) | 2026-09-01 | Fei-Fei Li's World Labs unveils Atlas, an "omni" world model trained from scratch across text/image/video/3D that, per the blog, generates spatially-consistent new views and scenes — unifying what Marble and RTFM did separately. |
| [Figure "Index" — largest physical-interaction dataset](https://www.figure.ai/news/introducing-index) | 2026-08-25 | Figure launches Index, billed (by Figure) as the world's largest and most diverse physical-interaction dataset for training embodied foundation models at scale. |
| [NVIDIA Cosmos 3](https://blogs.nvidia.com/blog/cosmos-3-physical-ai-open-world-foundation-model/) | 2026-05-31 | New open world foundation model for physical AI that unifies vision reasoning and multimodal generation (text/video/image/sound/action) in one model, folding the old Predict + Reason lines together. Announced at GTC Taipei (COMPUTEX). |
| [Google DeepMind Project Genie / Genie 3 + SIMA 2 loop](https://9to5google.com/2026/01/29/google-project-genie/) | 2026-01-29 | Genie 3 real-time interactive world model (reported 720p/24fps, ~1-min memory) rolled out as "Project Genie" to US AI Ultra subscribers; DeepMind also runs SIMA 2 agents inside Genie 3 worlds as a closed-loop embodied-AI training environment. |
| [Meta V-JEPA 2 (video world model)](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/) | 2025-06-11 | Self-supervised JEPA world model (1.2B params, trained on 1M+ hrs of video + ~62 hrs robot data) enabling zero-shot robot planning via goal images. A 2025 release that remains the reference JEPA model — no V-JEPA 3 found as of Oct 2026. |

---

## Related

- Radar index: [Frontier AI 2026 — Innovation Radar README](./README.md) · sibling pages [Reasoning & Agents](./Reasoning_and_Agents_2026.md), [Video & Image Generation](./Video_and_Image_Generation_2026.md), [Open-Source LLMs](./Open_Source_LLMs_2026.md)
- Stable pillars: [`02_LLM_AND_AI_MODELS/World_Models`](../02_LLM_AND_AI_MODELS/World_Models/) · [`05_VERTICAL_APPLICATIONS/10_Robotics_and_Embodied_AI`](../05_VERTICAL_APPLICATIONS/10_Robotics_and_Embodied_AI/) · [`05_VERTICAL_APPLICATIONS/11_Autonomous_Vehicles_AI`](../05_VERTICAL_APPLICATIONS/11_Autonomous_Vehicles_AI/)

**Sources:** [Figure Helix 2.5](https://www.figure.ai/news/helix-2-5-zero-shot-30-home-generalization) · [Figure 03 at BMW](https://www.figure.ai/news/f-03-at-bmw) · [Figure Index](https://www.figure.ai/news/introducing-index) · [Gemini Robotics ER 2 model card](https://deepmind.google/models/model-cards/gemini-robotics-er-2/) · [NVIDIA Isaac GR00T (GitHub)](https://github.com/NVIDIA/Isaac-GR00T) · [NVIDIA GR00T N2 / DreamZero](https://nvidianews.nvidia.com/news/nvidia-and-global-robotics-leaders-take-physical-ai-to-the-real-world) · [NVIDIA Cosmos 3](https://blogs.nvidia.com/blog/cosmos-3-physical-ai-open-world-foundation-model/) · [NVIDIA Alpamayo 2 Super](https://www.globenewswire.com/news-release/2026/06/01/3303988/0/en/NVIDIA-Launches-Alpamayo-2-Super-Open-Reasoning-Model-for-Robotaxis.html) · [World Labs Atlas](https://www.worldlabs.ai/blog/atlas) · [Physical Intelligence blog](https://www.pi.website/blog) · [Unitree STAR Market IPO](https://www.caixinglobal.com/2026-07-03/unitree-robotics-wins-approval-for-618-million-star-market-ipo-102460136.html) · [Apptronik Robot Park](https://roboticsandautomationnews.com/2026/07/06/apptronik-launches-robot-park-to-train-apollo-humanoid-robots-with-google-deepmind/103069/) · [Skild AI S1](https://www.therobotreport.com/skild-ai-unveils-s1-flagship-robot-foundation-model/) · [XPeng IRON line](https://electrek.co/2026/09/07/xpeng-iron-humanoid-robot-production-line/) · [1X NEO](https://www.1x.tech/discover/neo-home-robot) · [AgiBot GO-1](https://agibot-world.com/blog/go1) · [Google Project Genie](https://9to5google.com/2026/01/29/google-project-genie/) · [Meta V-JEPA 2](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/) · [Boston Dynamics + TRI LBM](https://www.tri.global/news/ai-powered-robot-boston-dynamics-and-toyota-research-institute-takes-key-step-towards-general) · [Tesla Optimus (Fremont line)](https://electrek.co/2026/04/22/tesla-optimus-production-fremont-model-sx-line/) · [Mistral Robostral Navigate](https://mistral.ai/news/robostral-navigate)

**Keywords:** physical AI 2026, humanoid robots, vision-language-action models, VLA, GR00T, Gemini Robotics 2, Physical Intelligence pi0.7, Skild AI S1, Figure 03, Figure Helix, 1X NEO, Unitree H2, XPeng IRON, Apptronik Apollo 2, world models, World Labs Atlas, NVIDIA Cosmos 3, Genie 3, V-JEPA 2, embodied foundation models, robot foundation models, robotaxi Alpamayo, embodied AI, latest robotics AI releases, October 2026.
