# ShapeNet (3D Models)

## Description
ShapeNet is an ongoing effort to establish a large-scale dataset of richly annotated 3D models. It is a collaboration between researchers from Princeton, Stanford, and TTIC, organized according to the WordNet hierarchy. The dataset is fundamental for research in computer graphics, computer vision, and robotics. It has two main subsets: ShapeNetCore and ShapeNetSem. The original ShapeNet indexed more than 3,000,000 models.

## Statistics
*   **ShapeNet (Total):** More than 3,000,000 indexed models (in 2015).
*   **ShapeNetCore:** Approximately **51,300** unique 3D models, covering **55** object categories.
*   **ShapeNetSem:** Approximately **12,000** models, spanning **270** categories.
*   **Main Version:** The main technical report is from 2015, but the dataset continues to be the foundation for new research and derivations (such as OpenShape, 2023).

## Features
The dataset consists of 3D models in OBJ+MTL format. It is organized hierarchically using the WordNet taxonomy.
*   **ShapeNetCore:** A subset of clean, unique 3D models with manually verified category and alignment annotations. Covers 55 common object categories.
*   **ShapeNetSem:** A smaller, more densely annotated subset with 270 categories. Includes annotations of real-world dimensions, material composition, volume, and weight.
*   **ShapeNetPart:** A popular derivation of ShapeNetCore with part segmentation annotations.

## Use Cases
*   3D object recognition and classification.
*   3D object part segmentation (with ShapeNetPart).
*   3D shape synthesis and generation.
*   3D object pose estimation.
*   Robotics research for object perception and manipulation.
*   Training deep learning models for volumetric shape representation (3D ShapeNets).

## Integration
Access to the model data (OBJ+MTL) and to the processed annotation metadata is provided for non-commercial research and/or educational purposes. It is necessary to **register** an account on the official website. After verification and approval by a site administrator, the download is enabled. The ShapeNet Viewer is provided for visualizing and rendering the models. The main article to be cited is: "ShapeNet: An Information-Rich 3D Model Repository" (2015).

## URL
[https://shapenet.org/](https://shapenet.org/)
