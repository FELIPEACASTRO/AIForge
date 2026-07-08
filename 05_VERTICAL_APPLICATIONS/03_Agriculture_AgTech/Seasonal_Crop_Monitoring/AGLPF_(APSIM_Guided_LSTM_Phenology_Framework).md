# AGLPF (APSIM Guided LSTM Phenology Framework)

## Description

The AGLPF (APSIM Guided LSTM Phenology Framework) is a continual learning framework based on physics-guided deep learning, developed to dynamically simulate changes in maize phenology. It combines the interpretability of process-based models (PBMs), such as APSIM, with the pattern-extraction capability of artificial intelligence models (AIMs), specifically an LSTM (Long Short-Term Memory) network with an attention mechanism. The framework is designed to overcome the limitations of both types of models, enabling continuous simulation and progressive improvement with new data. Initially trained on PBM output data, AGLPF is able to self-tune with real incremental phenology data, continuously improving its accuracy over time.

## Statistics

- **Initial Accuracy (Training with APSIM):** Average Root Mean Square Error (RMSE) of 0.8 days for the vegetative and flowering phases, 1.4 days for the grain-filling phase, and 2.0 days for the complete growth cycle.
- **Improvement with Continual Learning:** The RMSE of the complete growth cycle decreased from 27.8 days to **5.5 days** after self-tuning with real incremental phenology data (0 to 12 years).
- **Self-Tuning Advantage:** The self-tuning method outperformed the training-from-scratch method across all phenological phases.
- **Citations:** Cited by 2 (in 2025, according to the initial snippet).
- **Publication:** Agricultural and Forest Meteorology, Volume 373, 1 June 2025, 110562.

## Features

- **Continual Learning:** Ability to self-tune and improve performance through the incremental incorporation of new real phenology data, without forgetting prior knowledge.
- **Physics-Guided Deep Learning:** Integration of knowledge from process-based models (APSIM) to ensure interpretable and temporally continuous simulations.
- **Hybrid Model:** Combination of PBM (APSIM) and AIM (LSTM with Attention) for synergy between interpretability and predictive power.
- **Dynamic Phenology Simulation:** Ability to dynamically simulate the crop's growth phases (vegetative, flowering, grain filling).
- **Easy Updating and Interpretability:** Framework designed to be easily updated with new insights and to provide interpretable outputs.

## Use Cases

- **Dynamic Simulation of Crop Phenology:** Primary application in simulating changes in the growth phases of maize across large regions, such as the Chinese Maize Belt.
- **Real-Time Agricultural Monitoring:** Potential for use in real-time monitoring systems, where model accuracy can be continuously improved as new field data arrives.
- **Harvest Forecasting and Risk Management:** Improved accuracy in forecasting harvest dates and assessing risks related to climate and management, thanks to the model's higher accuracy and interpretability.
- **Adaptation to Environmental Change:** The continual learning capability allows the model to adapt to seasonal variations and climate change over time.

## Integration

The paper describes AGLPF as a conceptual and methodological framework. Implementation involves:
1. **PBM (APSIM):** Generation of an initial phenology dataset.
2. **AIM (LSTM with Attention):** Initial training of the deep learning model with the APSIM dataset.
3. **Self-Tuning:** Use of real incremental phenology data to refine the model via continual learning.

Although the specific source code is not available in the abstract, integration requires the use of deep learning libraries (such as PyTorch or TensorFlow) to implement the LSTM network with attention, and integration with a crop simulation model (such as APSIM) for data generation and physical guidance. The concept of "Physics-Guided Deep Learning" suggests the inclusion of constraints or variables from the physical model in the loss function of the deep learning model.

## URL

https://www.sciencedirect.com/science/article/abs/pii/S0168192325001820
