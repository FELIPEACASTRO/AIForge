# Curriculum Learning - Progressive Training Strategies

## Description

Curriculum Learning (CL) é uma estratégia de treinamento progressivo em Machine Learning que imita o processo de aprendizado humano, começando com exemplos mais fáceis e gradualmente introduzindo dados mais difíceis. Seu principal objetivo é melhorar a velocidade de convergência e a precisão final do modelo, especialmente em tarefas complexas ou com dados ruidosos. A CL é amplamente aplicada em Visão Computacional, Processamento de Linguagem Natural (PLN) e Aprendizado por Reforço (RL), demonstrando consistentemente ganhos de desempenho e estabilidade de treinamento. A estratégia envolve duas funções principais: uma função de pontuação (scoring function) para medir a dificuldade da amostra e uma função de ritmo (pacing function) para determinar a ordem e a taxa de introdução de novos exemplos.

## Statistics

**Melhoria na Convergência:** Estudos demonstram que a CL pode levar a uma taxa de convergência mais rápida (aceleração do treinamento) em comparação com o treinamento direto em todo o conjunto de dados [1] [9]. **Ganho de Precisão:** A CL tem sido associada a um ganho na precisão final (accuracy) e melhor desempenho de generalização, pois ajuda o modelo a evitar mínimos locais ruins [1] [9]. **Estabilidade:** A aplicação de CL, especialmente em Aprendizado por Reforço, pode aumentar a estabilidade do treinamento e reduzir a variância [10]. **Impacto da Dificuldade:** O ganho de desempenho obtido com a CL é diretamente proporcional ao comprimento do currículo (o número de etapas de dificuldade) [9].

## Features

**Estratégia de Treinamento Progressivo:** Ordena os dados de treinamento de exemplos "fáceis" para "difíceis" [1] [2]. **Função de Pontuação (Scoring Function):** Define a dificuldade de cada amostra de dados. Pode ser baseada em heurísticas (por exemplo, tamanho da imagem, complexidade da frase) ou métodos automáticos (por exemplo, perda do modelo, incerteza) [2] [3]. **Função de Ritmo (Pacing Function):** Controla a taxa na qual a dificuldade do currículo aumenta. Pode ser fixa (por exemplo, linear, exponencial) ou adaptativa (por exemplo, baseada no desempenho do modelo) [2]. **Tipos de Currículo:** Inclui Currículo por Transferência (começa com uma tarefa mais fácil e transfere o conhecimento para uma mais difícil) e Currículo por Amostra (ordena as amostras de dados) [4]. **Melhora na Generalização:** Ajuda o modelo a evitar mínimos locais ruins e a alcançar um mínimo global melhor, resultando em melhor desempenho de generalização [1].

## Use Cases

**Processamento de Linguagem Natural (PLN):** Treinamento de modelos de linguagem em tarefas como tradução automática, começando com frases curtas e simples e progredindo para frases mais longas e complexas [1] [11]. **Visão Computacional:** Treinamento de Redes Neurais Convolucionais (CNNs) para classificação de imagens, começando com imagens de alta qualidade e fácil distinção e progredindo para imagens ruidosas ou com baixa resolução [1] [7]. **Aprendizado por Reforço (RL):** Treinamento de agentes em ambientes simulados, começando com tarefas mais fáceis (por exemplo, objetivos mais próximos, menos obstáculos) e aumentando gradualmente a complexidade do ambiente ou da tarefa [1] [6]. **Reconhecimento de Fala:** Melhoria da robustez de modelos de reconhecimento de fala, começando com amostras de áudio limpas e progredindo para amostras com ruído de fundo [1].

## Integration

A implementação do Curriculum Learning geralmente envolve a criação de um *data loader* personalizado que aplica a lógica de pontuação e ritmo. Em frameworks como PyTorch ou Keras, isso é feito ajustando o conjunto de dados ou o *sampler* em cada época ou passo de treinamento.

**Exemplo de Integração (Conceitual em Python/PyTorch):**

```python
import torch
from torch.utils.data import Dataset, DataLoader, Subset

class CurriculoDataset(Dataset):
    # ... (Implementação do Dataset) ...

    def calcular_dificuldade(self, indice):
        # Lógica para pontuar a dificuldade (por exemplo, baseada no tamanho ou ruído)
        return dificuldade

def obter_curriculo_sampler(dataset, epoca, total_epocas):
    # 1. Calcular as pontuações de dificuldade para todas as amostras
    pontuacoes = [dataset.calcular_dificuldade(i) for i in range(len(dataset))]
    
    # 2. Determinar o limite de dificuldade para a época atual (Função de Ritmo)
    # Exemplo: Aumentar linearmente o percentual de dados usados
    percentual_dados = epoca / total_epocas
    limite_dificuldade = sorted(pontuacoes)[int(len(pontuacoes) * percentual_dados)]
    
    # 3. Selecionar os índices que atendem ao currículo
    indices_selecionados = [i for i, p in enumerate(pontuacoes) if p <= limite_dificuldade]
    
    # 4. Criar um Subset ou Sampler com os dados selecionados
    return Subset(dataset, indices_selecionados)

# Uso no loop de treinamento
# for epoca in range(total_epocas):
#     subset = obter_curriculo_sampler(dataset_completo, epoca, total_epocas)
#     dataloader = DataLoader(subset, batch_size=...)
#     # ... (Treinar o modelo) ...
```

**Bibliotecas e Repositórios:**
*   **CurML:** Uma biblioteca e kit de ferramentas para Curriculum Learning [5].
*   **Syllabus:** Um framework modular para Aprendizado por Reforço que facilita a integração de CL em pipelines de RL existentes [6].
*   **Implementações em GitHub:** Vários repositórios demonstram implementações em Keras e PyTorch, muitas vezes replicando resultados de artigos de pesquisa [7] [8].

## URL

https://arxiv.org/pdf/2010.13166