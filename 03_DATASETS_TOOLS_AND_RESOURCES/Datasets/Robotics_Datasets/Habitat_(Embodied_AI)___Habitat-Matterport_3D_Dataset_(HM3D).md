# Habitat (Embodied AI) / Habitat-Matterport 3D Dataset (HM3D)

## Description
Habitat is a high-performance simulation platform for Embodied AI research, developed by Meta AI. The main associated dataset is the **Habitat-Matterport 3D Dataset (HM3D)**, the largest collection of high-resolution 3D indoor spaces (digital twins) for training embodied agents (virtual robots and egocentric assistants) in photorealistic and efficient environments. The platform consists of: (i) **Habitat-Sim**, a high-performance 3D simulator with physics, and (ii) **Habitat-Lab**, a high-level library for agent development and training. The most recent version is **Habitat 3.0** (2023), which focuses on the cohabitation of humans, avatars, and robots.

## Statistics
**HM3D:** 1,000 building-scale 3D scenes. Total navigable space of **112,500 m²**. Total floor space of **365,420 m²**. Notable versions include **Habitat 3.0** (2023) and **HM3D-Sem** (2023), which adds semantic annotations. The original HM3D paper is from 2021.

## Features
**Habitat platform:** Photorealistic and efficient 3D simulation, with physics support (via Bullet), configurable sensors (RGB-D, egomotion), and robots described via URDF (Fetch, Franka, AlienGo). Habitat-Sim reaches thousands of frames per second (FPS). **HM3D dataset:** 1,000 building-scale 3D reconstructions of real environments (residential, commercial, civic). Each scene consists of textured 3D meshes and detailed metadata (reviewer rating, number of floors/rooms, navigable space, navigation complexity, and scene clutter). HM3D-Sem (2023) adds dense semantic annotations.

## Use Cases
Training and evaluation of Embodied AI agents, such as domestic robots and egocentric assistants. Autonomous navigation tasks (ObjectNav, ImageNav), object manipulation, environment rearrangement (Habitat 2.0), and human-agent interaction (Habitat 3.0). The HM3D dataset is a benchmark for research in active perception, long-horizon planning, and interactive learning in realistic 3D environments.

## Integration
Access to the HM3D dataset is free for academic and non-commercial research purposes. The download is done through the Matterport website, requiring the following steps: 1. Create a free Matterport account. 2. Access the **Developer Tools** in the account settings. 3. Request access to the **Habitat - Matterport 3D Research Dataset** and fill out the form. After approval, the dataset can be downloaded manually or programmatically using the `datasets_download` utility from Habitat-Sim. Habitat-Lab is installed via pip or conda.

## URL
[https://aihabitat.org/](https://aihabitat.org/)
