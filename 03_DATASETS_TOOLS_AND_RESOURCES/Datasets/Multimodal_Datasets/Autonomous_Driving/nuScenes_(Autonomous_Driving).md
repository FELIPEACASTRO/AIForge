# nuScenes (Autonomous Driving)

## Description
nuScenes is a **large-scale public dataset** for autonomous driving, developed by Motional (formerly nuTonomy). It was designed to allow researchers to study challenging urban driving situations using the full sensor suite of a real autonomous vehicle. The dataset is composed of 1000 scenes of 20 seconds each, collected in Boston and Singapore, cities known for their dense traffic and complex driving situations. nuScenes is the first large-scale dataset to provide data from a full sensor suite (6 cameras, 1 LIDAR, 5 RADAR, GPS, IMU) and includes precise 3D bounding box annotations for 23 object classes, along with attributes such as visibility, activity, and pose. Extended versions, such as **nuScenes-lidarseg** (semantic segmentation of LIDAR points) and **Panoptic nuScenes** (panoptic segmentation and tracking of point clouds), were released for more advanced tasks.

## Statistics
**Scenes:** 1000 scenes of 20 seconds each (approximately 5.5 hours of driving). **Images:** 1,400,000 camera images. **LIDAR Sweeps:** 390,000 LIDAR sweeps. **3D Bounding Boxes:** 1.4M 3D bounding boxes annotated across 40k *keyframes*. **Annotated LIDAR Points (nuScenes-lidarseg):** 1.4 billion points annotated across 40,000 point clouds. **Main Version:** nuScenes v1.0 (released in March 2019). **Devkit Versions:** The most recent devkit version is v1.2.0 (August 2025), compatible with Python 3.9 and 3.12.

## Features
**Full Sensor Suite:** 1x LIDAR (Velodyne HDL32E), 5x RADAR (Continental ARS 408-21), 6x Cameras (Basler acA1600-60gc), IMU, and GPS. **3D Annotations:** 1.4M manually annotated 3D bounding boxes for 23 object classes. **Geographic Diversity:** Data collected in two distinct cities (Boston and Singapore), enabling the study of algorithm generalization across different traffic conditions (left-hand vs. right-hand traffic), weather, and environments. **Expansions:** Includes expansions such as **nuScenes-lidarseg** (1.4 billion LIDAR points annotated with 32 semantic classes) and **Panoptic nuScenes** (panoptic segmentation of point clouds). **Low-Level Data:** CAN bus expansion with low-level vehicle data (wheel speed, steering angle, etc.).

## Use Cases
**3D Object Detection and Tracking:** Main application for the development of perception algorithms. **Point Cloud Semantic Segmentation:** Using the nuScenes-lidarseg expansion. **Panoptic Segmentation and Tracking:** With Panoptic nuScenes. **Behavior Prediction:** Agent trajectory prediction challenges. **Sensor Fusion:** Development of models that combine data from cameras, LIDAR, and RADAR. **Localization and Mapping:** Use of GPS/IMU data and HD maps.

## Integration
The nuScenes dataset is made available free of charge for strictly **non-commercial** use. To download it, you must **register and log in** on the official site (https://www.nuscenes.org/nuscenes#download) and agree to the Terms of Use. The download is done through multiple compressed files that must be unpacked into a specific folder structure (e.g., `/data/sets/nuscenes`). The development kit (**nuscenes-devkit**) is essential for working with the dataset and can be installed via pip: `pip install nuscenes-devkit`. The devkit provides tools for visualization, data manipulation, and model evaluation. The dataset is also available on the **Registry of Open Data on AWS**.

## URL
[https://www.nuscenes.org/](https://www.nuscenes.org/)
