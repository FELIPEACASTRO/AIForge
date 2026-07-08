# Geometric Deep Learning (GDL) and Equivariant Networks

## Description

Geometric Deep Learning (GDL) is a paradigm that aims to generalize Deep Learning models to data with underlying geometric structure, such as graphs, meshes, and manifolds, which are inherently non-Euclidean. The unique value proposition lies in incorporating **geometric inductive biases** (symmetries) directly into the network architecture, using **Group Theory** as the mathematical foundation. This results in **Equivariant Networks**, where the model's output transforms predictably (equivariance) or remains unchanged (invariance) under relevant transformations (e.g., rotations, translations). This approach drastically improves the **sample efficiency** and **generalization capability** of the models, being crucial for domains like chemistry and physics.

## Statistics

The **GeSS (Geometric Deep Learning under Scientific Applications with Distribution Shifts)** benchmark evaluates GDL models (such as EGNN and DGCNN) in scientific scenarios (Particle Physics, Materials Science, Biochemistry) under distribution shifts (Test-OOD). Results show that, in *Pileup Shift* ($\mathcal{Y}$-Conditional Shift) scenarios, the **EGNN (Equivariant Graph Neural Network)** achieved up to **89.41% Accuracy (ACC)** on *out-of-distribution* test data (Test-OOD), while *in-distribution* performance was above 95%. This gap highlights the challenge of OOD generalization and the need for robust architectures.

## Features

Key features of GDL and Equivariant Networks include: **Generalization to Non-Euclidean Data** (graphs, point clouds, meshes); **Equivariance and Invariance** (the model respects inherent data symmetries, such as the rotation of a molecule); **Parameter Efficiency** (imposing symmetries reduces the number of free parameters, preventing *overfitting* and requiring less data); and **Architecture Unification** (GDL provides a framework that encompasses CNNs, GNNs, and manifold networks).

## Use Cases

GDL is fundamental in areas where geometry and symmetry are crucial: **Drug Discovery**, where models with **$E(3)$ equivariance** (rotations, translations, and reflections) like **NequIP** are used to predict molecular and interatomic properties with high data efficiency; **Particle Physics (HEP)**, for particle track reconstruction in colliders like the LHC; **Materials Science**, for structure optimization and crystal property prediction; and **3D Computer Vision**, for point cloud processing and object recognition.

## Integration

Integration is done through libraries that extend existing Deep Learning frameworks. The main libraries are: **`e2cnn`** (for $E(2)$ in PyTorch, focused on 2D symmetries) and **`e3nn`** (for $E(3)$ in PyTorch, focused on 3D symmetries, essential for molecules). An integration example with `e2cnn` demonstrates defining the group space (`gspaces.Rot2dOnR2`), field types (`nn.FieldType`), and constructing equivariant convolutional (`nn.R2Conv`) and activation (`nn.ReLU`) layers, by wrapping the input tensor in `nn.GeometricTensor`.

## URL

https://geometricdeeplearning.com/