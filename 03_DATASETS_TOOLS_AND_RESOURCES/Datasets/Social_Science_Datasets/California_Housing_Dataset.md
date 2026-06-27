# California Housing Dataset

## Description
O **California Housing Dataset** é um conjunto de dados clássico de regressão amplamente utilizado em aprendizado de máquina para prever o valor médio de casas em distritos da Califórnia. Os dados foram coletados durante o censo de 1990 nos EUA. O objetivo é prever o valor mediano das casas (em unidades de $100.000) com base em oito características geográficas e socioeconômicas. Embora os dados sejam de 1990, eles continuam sendo um recurso fundamental para o ensino e teste de algoritmos de regressão, sendo frequentemente referenciado em pesquisas e tutoriais recentes (2023-2025) como um benchmark.

## Statistics
- **Total de Amostras (n_samples):** 20.640
- **Dimensionalidade (n_features):** 8
- **Tamanho:** Aproximadamente 1.5 MB (em formato CSV)
- **Versões:** O dataset é estático (baseado no censo de 1990), mas é constantemente atualizado e reempacotado em novas versões de bibliotecas como scikit-learn (e.g., versão 1.7.2).
- **Variável Alvo (Target):** Valor Mediano da Casa (real 0.15 - 5.0, em unidades de $100.000).

## Features
O dataset contém 8 atributos preditores (features) para cada bloco censitário:
1. **MedInc**: Renda mediana no bloco (em dezenas de milhares de dólares).
2. **HouseAge**: Idade mediana da casa no bloco.
3. **AveRooms**: Número médio de cômodos por domicílio.
4. **AveBedrms**: Número médio de quartos por domicílio.
5. **Population**: População do bloco.
6. **AveOccup**: Ocupação média do domicílio.
7. **Latitude**: Posição de latitude do bloco.
8. **Longitude**: Posição de longitude do bloco.
A variável alvo é o **Valor Mediano da Casa** (Median House Value), expressa em unidades de $100.000.

## Use Cases
- **Regressão Preditiva:** É o caso de uso primário, focado na previsão do valor mediano da casa.
- **Análise Exploratória de Dados (EDA):** Utilizado para praticar técnicas de visualização e compreensão de dados.
- **Comparação de Modelos:** Serve como um benchmark padrão para comparar o desempenho de diferentes algoritmos de regressão (e.g., Regressão Linear, Random Forests, Gradient Boosting).
- **Engenharia de Features:** Usado para demonstrar e testar a criação de novas features a partir das existentes (e.g., taxa de quartos por população).
- **Tutoriais e Ensino:** Amplamente empregado em cursos e tutoriais de Machine Learning devido à sua facilidade de acesso e estrutura limpa.

## Integration
O dataset é mais comumente acessado através da biblioteca **scikit-learn** em Python, o que facilita sua integração em projetos de aprendizado de máquina.

**Instruções de Uso (Python/scikit-learn):**
1. **Instalação da Biblioteca:** Certifique-se de que o scikit-learn esteja instalado: `pip install scikit-learn`
2. **Carregamento do Dataset:** Use a função `fetch_california_housing` para baixar e carregar o dataset diretamente.

```python
from sklearn.datasets import fetch_california_housing
import pandas as pd

# Carrega o dataset
housing = fetch_california_housing(as_frame=True)

# Cria um DataFrame para visualização
df = housing.frame

# Exibe as primeiras linhas
print(df.head())

# Variáveis X (features) e y (target)
X = housing.data
y = housing.target
```

O dataset também está disponível em plataformas como Kaggle e Hugging Face, geralmente em formato CSV.

## URL
[https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
