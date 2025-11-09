# Estratégias de Aumento de Dados (Data Augmentation)

## Description

O Aumento de Dados é um conjunto de técnicas utilizadas em aprendizado de máquina para expandir artificialmente o tamanho e a diversidade de um conjunto de dados de treinamento. Isso é alcançado através da criação de cópias modificadas de dados existentes, como imagens, texto ou áudio. A principal proposta de valor é combater o **overfitting** em modelos de aprendizado profundo, especialmente em cenários de dados limitados, e aumentar a **robustez** e a **capacidade de generalização** do modelo para dados não vistos [1] [2] [3].

## Statistics

Embora as estatísticas quantitativas variem amplamente por domínio e tarefa, o aumento de dados é consistentemente relatado como um fator chave para o sucesso de modelos de Deep Learning [4] [5]. Estudos empíricos demonstram que a aplicação de técnicas de aumento pode levar a **melhorias significativas na precisão** do modelo, especialmente quando o conjunto de dados de treinamento é pequeno. Por exemplo, em visão computacional, o aumento pode ser o diferencial para atingir a precisão de estado da arte (SOTA) em tarefas como classificação de imagens e detecção de objetos [6].

## Features

As estratégias de aumento são específicas para cada modalidade de dados:

**Imagem (Visão Computacional):**
*   **Transformações Geométricas:** Rotação, reflexão (flip), corte (crop), translação, cisalhamento (shear).
*   **Transformações de Cor/Pixel:** Ajuste de brilho, contraste, saturação, matiz (hue), adição de ruído Gaussiano.
*   **Técnicas Avançadas:** Cutout (mascaramento de regiões), MixUp (combinação linear de amostras), Mosaic (combinação de 4 imagens em uma) [6].

**Texto (Processamento de Linguagem Natural - PLN):**
*   **Substituição:** Substituição de sinônimos (WordNet), substituição de palavras por embeddings (Word2Vec).
*   **Manipulação de Sentenças:** Inserção, exclusão ou troca aleatória de palavras.
*   **Geração:** Back-translation (tradução reversa) e uso de Modelos de Linguagem de Máscara (MLMs) como BERT para gerar variações contextuais [7].

**Áudio (Processamento de Fala):**
*   **Transformações de Tempo:** Estiramento de tempo (time stretching), mudança de passo (pitch shifting).
*   **Transformações de Volume/Ruído:** Adição de ruído de fundo (background noise), mudança de volume.
*   **Transformações de Espectrograma:** Masking de frequência ou tempo no espectrograma (análogo ao Cutout para imagens) [8].

## Use Cases

O aumento de dados é crucial em diversas aplicações de IA:

*   **Diagnóstico Médico por Imagem:** Criação de variações de raios-X ou ressonâncias magnéticas para treinar modelos de detecção de doenças raras, onde os dados originais são escassos.
*   **Veículos Autônomos:** Geração de cenários de iluminação e clima variados (neblina, chuva, noite) para aumentar a robustez dos modelos de detecção de objetos e segmentação semântica [9].
*   **Análise de Sentimento:** Aumento de dados de texto para cobrir variações regionais, gírias e erros de digitação, melhorando a precisão da classificação de sentimentos.
*   **Reconhecimento de Comandos de Voz:** Adição de ruído de fundo (como em ambientes industriais ou de escritório) aos dados de áudio para tornar os modelos de reconhecimento de fala mais resistentes a interferências [10].

## Integration

A integração é tipicamente feita usando bibliotecas especializadas, aplicadas *on-the-fly* durante o treinamento do modelo.

**Exemplo de Aumento de Imagem (Python com Albumentations):**
```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define a pipeline de transformações
transform = A.Compose([
    A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.7),
    A.GaussNoise(var_limit=(20.0, 50.0), mean=0, p=0.3),
    ToTensorV2()
])

# Aplica a transformação a uma imagem (img_np é um array NumPy)
# augmented_image = transform(image=img_np)['image']
```

**Exemplo de Aumento de Texto (Python com NLPAug):**
```python
import nlpaug.augmenter.word as naw

# Aumentador de substituição de sinônimos
augmenter = naw.SynonymAug(aug_src='wordnet', stopwords=['I', 'the'])

text = "The phone case is great and durable. I absolutely love it."
aug_text = augmenter.augment(text)

# print("Original:", text)
# print("Augmented:", aug_text)
```

**Exemplo de Aumento de Áudio (Python com Audiomentations):**
```python
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift

# Define a pipeline de aumento de áudio
augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
])

# Aplica a transformação a amostras de áudio (samples é um array NumPy/Tensor)
# aug_samples = augment(samples=samples, sample_rate=sample_rate)
```

## URL

https://www.digitalocean.com/community/tutorials/data-augmentation-vision-language-audio-research