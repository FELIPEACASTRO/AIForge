# PPDock (Pocket Prediction-Based Protein–Ligand Blind Docking)

## Description

**PPDock** (Pocket Prediction-Based Protein–Ligand Blind Docking) is a protein–ligand blind *docking* model based on Graph Neural Networks (GNNs), proposed in 2025. It addresses the limitation of traditional *docking* methods that require the binding site (pocket) to be predefined. PPDock operates in two main stages: first, it predicts the binding site (pocket) on the protein, and then it performs the *docking* of the ligand within the predicted region. The architecture uses a combination of Euclidean Graph Neural Networks (EGNNs) to capture the complex spatial and structural relationships between the protein and the ligand, resulting in greater accuracy and efficiency, especially in blind *docking* scenarios.

## Statistics

- **Publication Year:** 2025
- **Key Metric:** Ligand RMSD (Root Mean Square Deviation) and Centroid Distance.
- **Performance:** Achieved a Ligand RMSD < 2 Å rate of **45.2%** on the original test set of 363 complexes, significantly outperforming state-of-the-art machine learning-based blind *docking* methods.
- **Architecture:** EGNN (Euclidean Graph Neural Network) and GNN.
- **Citations:** 5 (as of November 2025, indicating a very recent work).
- **Training/Test Dataset:** PDBbind.

## Features

- **Pocket Prediction-Based Blind Docking:** Requires no prior knowledge of the protein binding site.
- **EGNN Architecture:** Uses Euclidean Graph Neural Networks (EGNNs) to model the 3D interactions between protein and ligand, preserving rotation and translation invariance.
- **High Accuracy:** Outperforms state-of-the-art blind *docking* methods on benchmark datasets.
- **Computational Efficiency:** Offers a balance between accuracy and speed, crucial for large-scale virtual screening.

## Use Cases

- **Virtual Screening:** Fast and accurate identification of new drug candidates by predicting the binding pose of millions of molecules to a protein target.
- **Lead Optimization:** Refinement of promising compounds by predicting the most stable binding poses and affinity.
- **Blind Docking:** Application in cases where the protein binding site is unknown or difficult to determine experimentally.
- **Drug-Target Affinity (DTA) Prediction:** Although focused on *docking*, the predicted pose is crucial for accurately estimating affinity.

## Integration

The PPDock model is implemented in Python and requires GNN libraries such as PyTorch Geometric (PyG). The GitHub repository provides pre-trained model weights for both parts (Pocket Prediction and Docking).

**Usage Example (Conceptual):**
```python
# Installation (assuming a Python/PyTorch environment)
# pip install torch_geometric
# git clone https://github.com/JieDuTQS/PPDock
# cd PPDock

# Load model weights (available in the repository)
# model_pocket = load_model('pocket_prediction_weights.pth')
# model_dock = load_model('docking_weights.pth')

# 1. Data preparation (protein and ligand structure)
# protein_graph = preprocess_protein('protein.pdb')
# ligand_graph = preprocess_ligand('ligand.mol2')

# 2. Pocket Prediction
# pocket_coords = model_pocket.predict(protein_graph)

# 3. Blind Docking
# docked_pose = model_dock.dock(protein_graph, ligand_graph, pocket_coords)

# 4. Pose Evaluation
# rmsd = calculate_rmsd(docked_pose, true_pose)
```
The GitHub repository (`JieDuTQS/PPDock`) is the primary source for the code and integration instructions.

## URL

https://pubs.acs.org/doi/abs/10.1021/acs.jcim.4c01373