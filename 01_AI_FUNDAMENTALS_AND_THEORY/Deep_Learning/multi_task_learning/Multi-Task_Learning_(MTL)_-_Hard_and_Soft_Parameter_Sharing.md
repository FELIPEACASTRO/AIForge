# Multi-Task Learning (MTL) - Hard and Soft Parameter Sharing

## Description

O Aprendizado Multi-Tarefa (MTL) é uma abordagem de aprendizado de máquina que treina um único modelo em múltiplas tarefas relacionadas simultaneamente. Ele atua como uma forma de transferência indutiva e regularização implícita, forçando o modelo a aprender uma representação compartilhada que generaliza melhor e reduz o risco de overfitting. Os dois métodos principais de compartilhamento de parâmetros são o **Compartilhamento de Parâmetros Rígido** (Hard Parameter Sharing), onde as camadas ocultas são totalmente compartilhadas, e o **Compartilhamento de Parâmetros Suave** (Soft Parameter Sharing), onde cada tarefa tem seu próprio modelo, mas a distância entre seus parâmetros é regularizada para incentivar a similaridade. O compartilhamento rígido é o mais comum e eficaz para tarefas intimamente relacionadas, enquanto o suave oferece maior flexibilidade para tarefas mais sutilmente relacionadas.

## Statistics

**Redução de Overfitting:** O risco de overfitting nos parâmetros compartilhados é uma ordem de magnitude $N$ (número de tarefas) menor do que nos parâmetros específicos da tarefa. **Ganhos de Desempenho:** Melhorias típicas de 1% a 5% ou mais em métricas de tarefas principais em comparação com modelos de tarefa única (STL). **Eficiência:** O compartilhamento rígido reduz drasticamente o número total de parâmetros do modelo.

## Features

**Compartilhamento de Parâmetros Rígido:** Camadas ocultas compartilhadas, camadas de saída específicas da tarefa. Redução drástica do risco de overfitting ($\mathcal{O}(1/N)$). **Compartilhamento de Parâmetros Suave:** Modelos separados por tarefa com regularização na distância dos parâmetros (e.g., norma $L_2$, norma de traço). Maior flexibilidade e robustez contra transferência negativa. **Benefícios Gerais do MTL:** Aumento implícito de dados, foco de atenção em recursos relevantes, viés indutivo para melhor generalização.

## Use Cases

**Processamento de Linguagem Natural (NLP):** Modelos que realizam simultaneamente marcação de parte da fala (POS tagging), reconhecimento de entidade nomeada (NER) e análise sintática. **Visão Computacional:** Previsão conjunta de segmentação semântica e estimativa de profundidade a partir de uma única imagem. **Sistemas de Recomendação:** Previsão simultânea da probabilidade de clique e do tempo de permanência do usuário em um item. **Carros Autônomos:** Previsão da direção do volante usando tarefas auxiliares como previsão de características da estrada (e.g., marcações de faixa). **Medicina e Bioinformática:** Previsão simultânea de múltiplos sintomas ou a atividade de múltiplos compostos em descoberta de medicamentos.

## Integration

A integração é tipicamente realizada em frameworks de Deep Learning como PyTorch ou TensorFlow/Keras. O **Compartilhamento Rígido** é implementado definindo camadas ocultas comuns seguidas por cabeças de saída (output heads) separadas para cada tarefa. A otimização é feita minimizando uma perda total combinada, geralmente uma soma ponderada das perdas individuais de cada tarefa. O **Compartilhamento Suave** é implementado adicionando um termo de regularização à função de perda total que penaliza a diferença entre os parâmetros dos modelos específicos da tarefa.

```python
import torch
import torch.nn as nn

# Exemplo de Compartilhamento Rígido (PyTorch)
class HardSharingMTLModel(nn.Module):
    def __init__(self, input_size, shared_hidden_size, task_a_output_size, task_b_output_size):
        super().__init__()
        # Camadas Compartilhadas
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, shared_hidden_size),
            nn.ReLU(),
            nn.Linear(shared_hidden_size, shared_hidden_size),
            nn.ReLU()
        )
        # Cabeças Específicas da Tarefa
        self.task_a_head = nn.Linear(shared_hidden_size, task_a_output_size)
        self.task_b_head = nn.Linear(shared_hidden_size, task_b_output_size)

    def forward(self, x):
        shared_representation = self.shared_layers(x)
        output_a = self.task_a_head(shared_representation)
        output_b = self.task_b_head(shared_representation)
        return output_a, output_b

# Otimização da Perda Combinada:
# total_loss = weight_a * loss_a + weight_b * loss_b
```

## URL

https://www.ruder.io/multi-task/