# DeepVariant and AlphaFold

## Description

DeepVariant is an analysis pipeline that uses a deep neural network to call genetic variants from next-generation DNA sequencing data. It is designed to achieve high accuracy in identifying single nucleotide polymorphisms (SNPs) and small insertions/deletions (indels). AlphaFold is an AI system developed by Google DeepMind that predicts a protein's 3D structure from its amino acid sequence, recognized for its high accuracy in the Critical Assessment of protein Structure Prediction (CASP).

## Statistics

DeepVariant: Achieves over 99.9% accuracy for SNP variant calling; Reduces error rates by up to 10-fold compared to traditional methods; High performance on various sequencing platforms, including Illumina and PacBio. AlphaFold: Achieved a median backbone accuracy of 0.96 Å in CASP14; Predicted the structure of over 200 million proteins; AlphaFold DB provides open access to these predicted structures.

## Features

DeepVariant: Utilizes a deep convolutional neural network to identify genetic variants; Supports different sequencing technologies and data types; Open-source and available on Google Cloud, with support for GPU acceleration. AlphaFold: Predicts protein structures with high accuracy, often comparable to experimental methods; Can predict structures of protein complexes (multimers); Provides confidence scores (pLDDT) for its predictions.

## Use Cases

DeepVariant: Germline variant calling for whole genome and exome sequencing; Population-scale genomics studies; Clinical genetics and rare disease research. AlphaFold: Drug discovery and design; Understanding disease mechanisms; Enzyme engineering and synthetic biology.

## Integration

DeepVariant: Can be integrated into bioinformatics pipelines using Workflow Description Language (WDL) or executed as a standalone tool. Example integration with WDL can be found in the vgteam/vg_wdl GitHub repository. It can also be used within the Galaxy platform and as part of NVIDIA's Parabricks suite. AlphaFold: Predictions can be accessed via the AlphaFold Protein Structure Database. The source code is available on GitHub, and it can be run on Google Colab notebooks. For example, the sokrypton/ColabFold repository provides an easy-to-use interface for running AlphaFold predictions.

## URL

https://github.com/google/deepvariant, https://alphafold.ebi.ac.uk/