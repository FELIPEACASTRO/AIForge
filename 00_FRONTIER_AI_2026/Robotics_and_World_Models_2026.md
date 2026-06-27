# Robotics, Physical AI & World Models — 2026 Radar

"Physical AI" — Vision-Language-Action (VLA) models + world models — is one of 2026's fastest-moving fronts, with OEMs entering mass production.

## Latest Releases

| Model | Org | Released | Notes |
|---|---|---|---|
| **GR00T N1.7** (Early Access) | NVIDIA | 2026-04-17 | 3B-param open, commercially-licensed VLA on a **Cosmos-Reason2-2B** backbone; 32-layer DiT for low-level motor control; **Action Cascade** dual-system architecture. |
| **WholebodyVLA** | OpenDriveLab | ICLR 2026 | Unified latent VLA for whole-body loco-manipulation control. |

## State of the Field (2026)

- **World models becoming standard** — per industry reporting, world models are about to become standard equipment for embodied systems; OEMs are entering and accelerating mass production.
- **Perception → decision → execution closed loop** — VLAs made significant progress, but still largely "imitate" training patterns and lack foresight of action consequences — the key open research bottleneck.
- **11 tech giants in play** — Alibaba, NVIDIA, Google DeepMind, OpenAI, Microsoft, Huawei, and others now ship embodied/robot large models.

## Key Building Blocks (current)

- **VLAs**: GR00T (NVIDIA), OpenVLA, π0 / π0.5 (Physical Intelligence), Octo, Helix (Figure), RT-2/RT-X
- **World models**: NVIDIA Cosmos, DeepMind Genie line, V-JEPA 2 (Meta), GAIA (Wayve)
- **Simulators**: Isaac Sim / Isaac Lab, MuJoCo Playground, Genesis, Habitat 3.0, ManiSkill 3
- **Stacks**: ROS 2, HuggingFace LeRobot, NVIDIA Isaac ROS

## Curated external list

- **awesome-physical-ai** — VLAs, world models, embodied AI, robotic foundation models — https://github.com/keon/awesome-physical-ai

## Where to go deeper in AIForge

- [`05_VERTICAL_APPLICATIONS/10_Robotics_and_Embodied_AI`](../05_VERTICAL_APPLICATIONS/10_Robotics_and_Embodied_AI/)
- [`02_LLM_AND_AI_MODELS/World_Models`](../02_LLM_AND_AI_MODELS/World_Models/)

## Sources
- [MarkTechPost — Top 10 Physical AI models 2026](https://www.marktechpost.com/2026/04/28/top-10-physical-ai-models-powering-real-world-robots-in-2026/)
- [GlobeNewswire — Embodied AI Robot Large Model Report 2026](https://www.globenewswire.com/news-release/2026/05/13/3293694/0/en/embodied-ai-robot-large-model-including-vla-research-report-2026-world-models-are-about-to-become-standard-and-oems-enter-and-accelerate-mass-production-and-application.html)
- [awesome-physical-ai (GitHub)](https://github.com/keon/awesome-physical-ai)
- [OpenDriveLab/WholebodyVLA (GitHub)](https://github.com/OpenDriveLab/WholebodyVLA)
