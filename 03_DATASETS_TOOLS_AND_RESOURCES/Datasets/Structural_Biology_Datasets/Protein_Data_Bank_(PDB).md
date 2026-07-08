# Protein Data Bank (PDB)

## Description

The Protein Data Bank (PDB) is the primary global repository of experimentally determined three-dimensional (3D) structures of biological macromolecules, such as proteins and nucleic acids (determined by X-ray crystallography, NMR, cryo-EM). It is a fundamental resource for structural biology and bioinformatics. Recently (2023-2025), the focus has expanded to include Integrative Structures and Computed Structure Models (CSM) from sources such as AlphaFold DB and ModelArchive, consolidating the PDB as a hub for molecular structure data.

## Statistics

As of November 2025, the PDB archive contained 244,730 available structures. Annual growth in 2025 was 15,064 new structures, following the trend of steady growth. The archive also includes more than 1 million Computed Structure Models (CSM).

## Features

3D Structures of Proteins and Nucleic Acids; Experimental Data (X-ray, NMR, Cryo-EM); Biocuration Metadata; Integrative Structures; Computed Structure Models (CSM).

## Use Cases

Molecular modeling and simulation; Drug discovery and design; Analysis of protein-ligand interactions; Studies of protein folding and function; Training of Machine Learning models for structure and function prediction.

## Integration

Access can be done via web interface (rcsb.org) or programmatically. RCSB PDB offers the Python package `rcsb-api` (available on PyPI) for simplified access to search and data services, allowing complex queries and data retrieval via REST and GraphQL APIs. Installation example: `pip install rcsb-api`. Data API usage example: `https://data.rcsb.org/rest/v1/core/entry/1A2C`.

## URL

https://www.rcsb.org/