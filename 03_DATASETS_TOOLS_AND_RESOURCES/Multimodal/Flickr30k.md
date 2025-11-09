# Flickr30k

## Description
O Flickr30k é um dataset de referência (benchmark) amplamente utilizado para tarefas de descrição de imagens baseada em sentenças (Image Captioning) e recuperação de texto-imagem (Text-Image Retrieval). Ele consiste em 31.783 imagens coloridas obtidas do Flickr, cada uma pareada com cinco legendas (captions) distintas, escritas por humanos. Uma versão estendida, o **Flickr30k Entities**, aumenta a utilidade do dataset ao adicionar anotações de caixas delimitadoras (bounding boxes) e cadeias de correferência para as entidades mencionadas nas legendas, permitindo o alinhamento entre regiões da imagem e frases nas descrições. É um dos datasets mais importantes para o desenvolvimento de modelos multimodais.

## Statistics
*   **Imagens:** 31.783 imagens coloridas.
*   **Legendas (Captions):** 158.915 legendas em Inglês (5 por imagem).
*   **Versão Estendida (Flickr30k Entities):** Inclui 244.000 cadeias de correferência e 276.000 caixas delimitadoras.
*   **Versões Recentes (2024):** O dataset continua sendo usado como base para novas extensões, como o **Flickr30K-CFQ** (2024) e o **FM30K** (2024, com legendas em Português do Brasil), indicando sua relevância contínua.

## Features
*   **Multimodalidade:** Combina dados visuais (imagens) e textuais (legendas).
*   **Diversidade:** Imagens capturam pessoas engajadas em atividades cotidianas, garantindo uma ampla variedade de cenas e conceitos.
*   **Múltiplas Legendas:** Cada imagem possui 5 legendas independentes, o que permite uma avaliação mais robusta dos modelos de *image captioning*.
*   **Extensão de Entidades:** A versão Flickr30k Entities adiciona 244 mil cadeias de correferência e 276 mil caixas delimitadoras anotadas manualmente, ligando entidades textuais a regiões visuais.

## Use Cases
*   **Image Captioning:** Geração automática de descrições textuais para imagens.
*   **Text-Image Retrieval:** Busca de imagens a partir de uma descrição textual e vice-versa.
*   **Grounding de Entidades Visuais:** Alinhamento de frases nas legendas com regiões específicas da imagem (especialmente com a versão Flickr30k Entities).
*   **Modelos Multimodais:** Treinamento e avaliação de modelos que integram visão computacional e processamento de linguagem natural (CV+NLP).
*   **Tradução Multimodal:** Utilizado como base para datasets multilíngues, como o Multi30K e o FM30K (Português do Brasil).

## Integration
O dataset original pode ser obtido através da página oficial da Universidade de Illinois, que fornece links para as imagens e as legendas. Para a versão publicamente distribuível (links de imagem + legendas), o processo envolve o download de arquivos de anotação e a obtenção das imagens diretamente do Flickr (sujeito aos Termos de Uso do Flickr).

**Integração via Hugging Face (Recomendada):**
A maneira mais moderna e simples de integrar o dataset é através da biblioteca `datasets` do Hugging Face, que já gerencia as anotações e a estrutura de dados:
```python
from datasets import load_dataset

# Para a versão base
dataset = load_dataset("nlphuji/flickr30k")

# Para a versão com legendas em Português do Brasil (FM30K)
# dataset = load_dataset("FrameNetBrasil/FM30K")
```
**Observação:** O usuário deve garantir que possui as imagens do Flickr para uso não comercial, pois os arquivos de anotação geralmente contêm apenas os links ou IDs das imagens. Plataformas como Kaggle frequentemente fornecem versões pré-empacotadas do dataset, mas a fonte oficial deve ser consultada para os termos de uso.

## URL
[https://shannon.cs.illinois.edu/DenotationGraph/data/index.html](https://shannon.cs.illinois.edu/DenotationGraph/data/index.html)
