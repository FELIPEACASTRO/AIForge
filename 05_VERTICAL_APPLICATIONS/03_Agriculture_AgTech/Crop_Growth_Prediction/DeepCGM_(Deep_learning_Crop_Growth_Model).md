# DeepCGM (Deep learning Crop Growth Model)

## Description

**DeepCGM (Deep learning Crop Growth Model)** is a deep-learning-based crop growth model that incorporates knowledge-guided constraints to ensure physically plausible simulations. It addresses the limitations of traditional process-based models (such as ORYZA2000), which suffer from oversimplification and difficulty in parameter estimation, and of classical machine learning models, which are criticized for being "black boxes" and requiring large volumes of data. DeepCGM uses a mass-conservation architecture and crop physiological constraints to operate with sparse time-series data.

## Statistics

**Accuracy Improvement:** Outperforms the traditional process-based ORYZA2000 model, with overall accuracy (weighted normalized root mean square error) across all variables improving by **8.3% (2019)** and **16.9% (2018)**. **Citations:** The associated paper, "Knowledge-guided machine learning with multivariate sparse data for crop growth modelling" (J. Han et al., 2025), already has **2 citations** (in 2025), indicating recent relevance in the scientific community. **Publication:** The work was published in the journal *Field Crops Research* in 2025.

## Features

**Mass-Conservation Architecture:** Adheres to crop growth principles, such as mass conservation, to ensure physically realistic predictions. **Knowledge-Guided Constraints:** Includes crop physiology and model convergence constraints, enabling accurate predictions even with sparse data. **Multivariable Prediction:** Simulates multiple crop growth variables (e.g., biomass, leaf area) within a single framework. **Open Source:** The code is available on GitHub, facilitating research and implementation.

## Use Cases

**Crop Growth Simulation:** Accurate and physically plausible simulation of growth variables such as Plant Area Index (PAI), individual organ biomass (leaf, stem, grain), and total aboveground biomass (WAGT). **Modeling with Sparse Data:** Ideal for agricultural scenarios where time-series data collection is sparse or incomplete. **Replacement of Process-Based Models:** Serves as a more accurate and robust alternative to traditional process-based crop growth models, such as ORYZA2000.

## Integration

The model is implemented in Python and can be trained using the `train.py` script in the GitHub repository. Installation is done via `conda` and `pip` from the `requirements.txt` file.

**Training Example:**
```shell
git clone https://github.com/WUR-AI/DeepCGM.git
cd DeepCGM
conda create -n DeepCGM python==3.10.16
conda activate DeepCGM
pip install -r ./requirements.txt
python train.py --model DeepCGM --target spa --input_mask 1 --convergence_loss 1 --tra_year 2018
```
The script allows specifying the model type (`NaiveLSTM`, `MCLSTM`, `DeepCGM`), the training label (`spa` for sparse data), and enabling the use of the input mask and convergence loss.

## URL

https://github.com/WUR-AI/DeepCGM
