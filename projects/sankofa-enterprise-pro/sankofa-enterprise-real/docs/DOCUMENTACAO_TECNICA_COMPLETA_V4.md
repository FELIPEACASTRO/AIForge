




# Documentação Técnica Completa - Sankofa Enterprise Pro V4.0

**Versão:** 4.0

**Data:** 21 de Setembro de 2025

**Autor:** Manus AI

## 1. Introdução

Esta documentação descreve em detalhes a arquitetura, o funcionamento e o uso do motor de detecção de fraude **Sankofa Enterprise Pro V4.0 - Ultra-Precision**. Esta nova versão foi desenvolvida para superar as limitações da V3.0, com foco principal em aumentar a **precisão** e a **robustez** do sistema, mantendo um alto **recall** e performance.

O motor V4.0 introduz uma série de melhorias significativas, incluindo um ensemble de modelos mais sofisticado, calibração de probabilidades, feature engineering avançada e um sistema de múltiplos thresholds para diferentes apetites de risco.

## 2. Arquitetura do Motor V4.0

O motor V4.0 é baseado em uma arquitetura de **stacking ensemble**, que combina as predições de múltiplos modelos base para treinar um meta-modelo final. Essa abordagem permite capturar diferentes tipos de padrões nos dados e produzir uma predição mais acurada e robusta.

A arquitetura é composta pelas seguintes camadas:

1.  **Camada 0: Modelos Base:** Um conjunto de 5 modelos de machine learning heterogêneos que aprendem diretamente com os dados de transação.
2.  **Camada de Calibração:** Cada modelo base tem suas probabilidades calibradas para garantir que os scores de fraude reflitam com mais precisão a probabilidade real de uma transação ser fraudulenta.
3.  **Camada 1: Meta-Modelo:** Um modelo de regressão logística que aprende a combinar as predições calibradas dos modelos base de forma ponderada.
4.  **Sistema de Ponderação Adaptativa:** Os pesos de cada modelo base no meta-modelo são calculados dinamicamente com base em sua performance (precision e F1-score) em um conjunto de validação.

### 2.1. Ensemble de Modelos

O coração do motor V4.0 é seu ensemble de 5 modelos base, escolhidos por suas diferentes fortalezas na captura de padrões complexos:

| Modelo | Nome | Principais Características |
| :--- | :--- | :--- |
| `rf` | **Random Forest Classifier** | Robusto, bom para relações não-lineares e interações entre features. |
| `gb` | **Gradient Boosting Classifier** | Alta capacidade preditiva, constrói modelos de forma sequencial, corrigindo os erros dos anteriores. |
| `et` | **Extra Trees Classifier** | Similar ao Random Forest, mas com maior aleatoriedade na seleção de features e thresholds, o que pode reduzir o overfitting. |
| `lr` | **Logistic Regression** | Modelo linear simples e interpretável, bom como baseline e para capturar relações lineares. |
| `svm` | **Support Vector Classifier** | Eficaz em espaços de alta dimensão, encontra o hiperplano que melhor separa as classes. |

### 2.2. Meta-Modelo

O meta-modelo é uma **Regressão Logística** que recebe como entrada as probabilidades de fraude previstas por cada um dos 5 modelos base calibrados. Ele aprende a atribuir pesos a cada modelo, combinando suas predições para gerar o score de fraude final. Essa abordagem de stacking permite que o sistema aprenda com os erros e acertos de cada modelo individual, resultando em uma predição final mais precisa.

