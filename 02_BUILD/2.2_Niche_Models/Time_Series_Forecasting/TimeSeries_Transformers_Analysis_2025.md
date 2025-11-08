# A Closer Look at Transformers for Time Series Forecasting

## 🇬🇧 English

### Overview

This critical analysis from ICML 2025 questions the effectiveness of complex Transformer architectures for time series forecasting. The study demonstrates that performance is dominated by intra-variable dependencies and that simple components like skip connections and Z-score normalization are crucial.

### Key Findings

- **Simplicity Over Complexity:** The paper argues that excessive complexity in Transformer models for time series is often unnecessary and that simpler, more efficient models can be more effective.
- **Dominance of Intra-variable Dependencies:** The analysis reveals that the model's performance is primarily driven by its ability to capture dependencies within each time series variable, rather than complex inter-variable relationships.
- **Importance of Basic Components:** Simple components like skip connections and Z-score normalization are shown to be critical for achieving good performance.

### Impact

This research is a game-changer for the field of time series forecasting. It shifts the focus of research away from building increasingly complex Transformer architectures and towards developing simpler, more efficient, and better-understood models. The associated GitHub repository provides the code to reproduce the experiments.

- **Source:** [GitHub (ICML 2025)](https://github.com/yc14600/TimeSeries-Transformers-Analysis/)

---

## 🇧🇷 Português

### Visão Geral

Esta análise crítica do ICML 2025 questiona a eficácia de arquiteturas Transformer complexas para a previsão de séries temporais. O estudo demonstra que o desempenho é dominado por dependências intra-variáveis e que componentes simples como conexões de salto (skip connections) e normalização Z-score são cruciais.

### Principais Descobertas

- **Simplicidade em vez de Complexidade:** O artigo argumenta que a complexidade excessiva em modelos Transformer para séries temporais é muitas vezes desnecessária e que modelos mais simples e eficientes podem ser mais eficazes.
- **Dominância de Dependências Intra-variáveis:** A análise revela que o desempenho do modelo é impulsionado principalmente por sua capacidade de capturar dependências dentro de cada variável da série temporal, em vez de relações complexas entre variáveis.
- **Importância de Componentes Básicos:** Componentes simples como conexões de salto e normalização Z-score são mostrados como críticos para alcançar um bom desempenho.

### Impacto

Esta pesquisa é um divisor de águas para o campo da previsão de séries temporais. Ela desloca o foco da pesquisa da construção de arquiteturas Transformer cada vez mais complexas para o desenvolvimento de modelos mais simples, eficientes e mais bem compreendidos. O repositório GitHub associado fornece o código para reproduzir os experimentos.

- **Fonte:** [GitHub (ICML 2025)](https://github.com/yc14600/TimeSeries-Transformers-Analysis/)
