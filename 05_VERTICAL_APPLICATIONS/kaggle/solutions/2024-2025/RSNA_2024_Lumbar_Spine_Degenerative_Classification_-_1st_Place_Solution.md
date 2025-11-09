# RSNA 2024 Lumbar Spine Degenerative Classification - 1st Place Solution

## Description

Solução de 1º lugar para a competição RSNA 2024 Lumbar Spine Degenerative Classification, focada na classificação degenerativa da coluna lombar a partir de imagens médicas. A abordagem de **2 estágios** é a proposta de valor única: o primeiro estágio foca na localização precisa de coordenadas (`test_label_coordinates.csv`) usando modelos 3D e 2D ConvNeXt e Efficientnet-v2-l, e o segundo estágio realiza a previsão da gravidade usando **Multiple Instance Learning (MIL)** com bi-LSTM e atenção. A solução demonstrou que modelos menores (ConvNeXt-small) e arquiteturas convolucionais superaram modelos maiores e Vision Transformers.

## Statistics

**Abordagem:** 2 estágios (Localização de Coordenadas + Previsão de Gravidade). **Modelos:** 3 tipos de modelos (previsão de `instance_number`, previsão de coordenadas e previsão de gravidade). **Arquiteturas:** 3D ConvNeXt, 2D ConvNeXt-base, Efficientnet-v2-l, ConvNeXt-small e Efficientnet-v2-s com MIL. **Perdas:** L1 Loss e Cross Entropy Loss. **Aumento de Dados:** Deslocamento aleatório de coordenadas e `instance_number`, ShiftScaleRotate, RandomBrightnessContrast.

## Features

1. **Localização Precisa:** Uso de modelos 3D e 2D para prever o `instance_number` e as coordenadas (x, y, z) das vértebras, essencial para o corte e foco na área de interesse. 2. **Classificação de Gravidade com MIL:** O segundo estágio utiliza uma abordagem Multiple Instance Learning (MIL) com bi-LSTM e atenção para a classificação final da gravidade. 3. **Robustez:** Uso de ensemble de previsões de coordenadas e aumento de dados (como o deslocamento de `instance_number`) para aumentar a robustez do modelo. 4. **Otimização de Arquitetura:** Descoberta de que modelos menores (ConvNeXt-small) e arquiteturas convolucionais foram mais eficazes.

## Use Cases

1. **Diagnóstico Médico Assistido por IA:** Classificação automatizada da gravidade da doença degenerativa da coluna lombar a partir de exames de ressonância magnética. 2. **Análise de Imagens Médicas:** Aplicação de técnicas de localização e classificação em imagens 3D (DICOM) para identificar e avaliar patologias. 3. **Desenvolvimento de Modelos de Visão Computacional:** Demonstração de uma pipeline eficaz de 2 estágios para tarefas complexas de visão computacional em dados médicos.

## Integration

A solução utiliza PyTorch e módulos personalizados. O componente chave de MIL é implementado com a seguinte estrutura em Python/PyTorch:
```python
class LSTMMIL(nn.Module):
    def __init__(self, input_dim):
        super(LSTMMIL, self).__init__()
        self.lstm = nn.LSTM(input_dim, input_dim//2, num_layers=2, batch_first=True, dropout=0.1, bidirectional=True)
        self.aux_attention = nn.Sequential(
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )
        self.attention = nn.Sequential(
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )
    def forward(self, bags):
        batch_size, num_instances, input_dim = bags.size()
        bags_lstm, _ = self.lstm(bags)
        attn_scores = self.attention(bags_lstm).squeeze(-1)
        aux_attn_scores = self.aux_attention(bags_lstm).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        weighted_instances = torch.bmm(attn_weights.unsqueeze(1), bags_lstm).squeeze(1)
        return weighted_instances, aux_attn_scores
```
O código de treinamento está no Google Colab, e o código de inferência está em um notebook Kaggle separado.

## URL

https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/avengers-1st-place-solution