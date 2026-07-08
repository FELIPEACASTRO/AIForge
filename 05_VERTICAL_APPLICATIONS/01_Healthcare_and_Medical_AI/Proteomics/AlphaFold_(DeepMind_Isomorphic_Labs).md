# AlphaFold (DeepMind/Isomorphic Labs)

## Description

AlphaFold is an artificial intelligence system developed by Google DeepMind and Isomorphic Labs that solved the 'protein folding problem' by predicting the 3D structure of proteins from their amino acid sequence with remarkable accuracy. AlphaFold 3 expanded this capability to predict the joint structure and interactions of biomolecular complexes, including proteins, DNA, RNA, ligands, and antibodies, revolutionizing structural proteomics and drug discovery.

## Statistics

More than 200 million predicted protein structures (covering almost the entire catalogued proteome). More than 2 million researchers in over 190 countries use the AlphaFold Protein Structure Database (AlphaFold DB). It is estimated to have saved up to 1 billion years of research. AlphaFold 3 demonstrated 50% greater accuracy in predicting protein-ligand interactions compared to previous methods.

## Features

3D Protein Structure Prediction (AlphaFold 2). Joint Structure Prediction of Biomolecular Complexes (AlphaFold 3). Modeling of Protein-Protein Interactions (PPI), Protein-Nucleic Acid (DNA/RNA), and Protein-Ligand interactions. High accuracy in structure prediction, measured by the confidence score (pLDDT). AlphaFold 3 source code and model weights available for academic use.

## Use Cases

Acceleration of Drug Discovery and Therapeutic Targets. Enzyme Engineering for Biocatalysis and Plastic Recycling. Understanding Disease Mechanisms (e.g., Parkinson's, Cancer) through the structure of defective proteins. Vaccine and Antibody Design. Optimization of Agricultural Crops and Pathogen Control.

## Integration

AlphaFold DB offers a public API for programmatic access to the more than 200 million predicted structures. The source code for AlphaFold 2 and 3 (for academic use) is available on GitHub, allowing local execution via Docker or Python scripts. Tools such as AlphaPulldown facilitate the prediction of protein-protein interactions (PPI) using AlphaFold-Multimer. Integration with the AlphaFold Server (AlphaFold 3) is done via a web interface for commercial and academic use.

## URL

https://deepmind.google/science/alphafold/