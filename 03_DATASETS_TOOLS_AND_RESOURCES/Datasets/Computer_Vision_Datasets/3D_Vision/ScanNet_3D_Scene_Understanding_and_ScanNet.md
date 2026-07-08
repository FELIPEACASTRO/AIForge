# ScanNet (3D Scene Understanding) and ScanNet++

## Description
ScanNet is an RGB-D video dataset widely used for indoor 3D scene understanding. The original version (v2) contains 2.5 million views across more than 1500 scenes, annotated with 3D camera poses, surface reconstructions, and instance-level semantic segmentations. The most recent and high-fidelity version, **ScanNet++** (ICCV 2023 Oral), significantly expands the dataset, offering more than **1000 indoor scenes** with sub-millimeter resolution laser scans, registered 33-megapixel DSLR images, and RGB-D streams from commodity devices (iPhone). ScanNet++ focuses on long-tail and ambiguous semantic annotations to improve semantic understanding methods, and also supports Novel View Synthesis (NVS) benchmarks in high-quality and commodity configurations. The dataset is fundamental for advancing research in computer vision and robotics.

## Statistics
**ScanNet v2 (Original):**
*   **Scenes:** More than 1500.
*   **Views:** 2.5 million RGB-D frames.
*   **Version:** v2 (released in 2018).

**ScanNet++ (Most recent version):**
*   **Scenes:** More than 1000 (v2, December 2024).
*   **Resolution:** Sub-millimeter resolution laser scans.
*   **Images:** 33-megapixel DSLR images.
*   **Publication:** ICCV 2023 Oral.

## Features
**ScanNet v2:**
*   RGB-D videos.
*   2.5 million views.
*   More than 1500 indoor scenes.
*   3D camera pose annotations, surface reconstructions, and instance-level semantic segmentation.

**ScanNet++ (v2, December 2024):**
*   More than **1000 indoor scenes**.
*   **Sub-millimeter resolution** laser scans.
*   Registered **33-megapixel** DSLR images.
*   RGB-D streams from commodity devices (iPhone).
*   Long-tail and ambiguous semantic annotations.
*   Support for Novel View Synthesis (NVS) benchmarks.
*   **ScanNet200 Benchmark** for semantic segmentation with 200 class categories.

## Use Cases
*   **3D Scene Understanding:** 3D object classification, semantic voxel labeling, and 3D semantic and instance segmentation.
*   **Robotics:** Robot navigation and interaction in indoor environments.
*   **Augmented/Virtual Reality (AR/VR):** High-fidelity reconstruction of indoor environments.
*   **Novel View Synthesis (NVS):** Generation of new perspectives of indoor scenes.
*   **Contrastive Learning:** Data-efficient 3D scene understanding benchmarks (ScanNet-LA).

## Integration
Access to the original ScanNet dataset (v2) and to ScanNet++ requires acceptance of the Terms of Use and obtaining a non-commercial license.

1.  **ScanNet (v2):** The code and data are available in the official GitHub repository. You must agree to the terms of use and follow the instructions on the main site to obtain access to the data.
    *   **Code and Data URL:** `https://github.com/ScanNet/ScanNet`
2.  **ScanNet++:** You must create an account, log in, and submit an access request on the official site. After approval, a personalized token and a new download script are provided to access the data.
    *   **Download URL:** `https://scannetpp.mlsg.cit.tum.de/scannetpp/` (Requires registration and approval)

## URL
[http://www.scan-net.org/](http://www.scan-net.org/)
