# LJSpeech Dataset

## Description
O **LJ Speech Dataset** é um corpus de domínio público amplamente utilizado para treinamento e avaliação de modelos de **Text-to-Speech (TTS)**. Consiste em 13.100 clipes de áudio curtos de uma única falante feminina lendo passagens de 7 livros de não-ficção em inglês. Cada clipe de áudio é acompanhado por uma transcrição correspondente, tanto na forma original quanto em uma versão normalizada onde números e abreviações são expandidos em palavras completas. Os textos foram publicados entre 1884 e 1964, e o áudio foi gravado em 2016-2017 como parte do projeto LibriVox. O dataset é conhecido por sua alta qualidade de gravação e consistência, sendo um padrão de referência na pesquisa de síntese de fala.

## Statistics
*   **Versão Mais Recente:** 1.1
*   **Tamanho do Download:** 2.6 GB
*   **Total de Clipes:** 13.100
*   **Duração Total:** 23 horas, 55 minutos e 17 segundos (23:55:17)
*   **Duração Média do Clipe:** 6.57 segundos
*   **Total de Palavras:** 225.715
*   **Taxa de Amostragem:** 22050 Hz
*   **Formato de Áudio:** WAV, mono, 16-bit PCM

## Features
*   **Domínio Público:** O dataset, incluindo textos, áudio e anotações, está em domínio público, sem restrições de uso.
*   **Falante Única:** Contém gravações de uma única falante feminina, o que o torna ideal para modelos de TTS de falante única.
*   **Transcrição Normalizada:** Inclui transcrições originais e normalizadas (com números e abreviações expandidos), facilitando o pré-processamento para modelos de TTS.
*   **Alta Qualidade:** Áudio em formato WAV, mono, 16-bit PCM, com taxa de amostragem de 22050 Hz.
*   **Segmentação:** Os clipes de áudio são curtos (1 a 10 segundos), segmentados automaticamente com alinhamento manual e garantia de qualidade (QA) para precisão.

## Use Cases
*   **Síntese de Fala (Text-to-Speech - TTS):** É o principal caso de uso, servindo como um dataset de referência para treinar e avaliar modelos de TTS, como Tacotron, FastSpeech, Glow-TTS e VITS.
*   **Clonagem de Voz (Voice Cloning):** Utilizado para criar modelos de clonagem de voz de falante única.
*   **Pesquisa em Processamento de Áudio:** Usado para experimentos em pré-processamento de áudio, análise de espectrogramas e técnicas de normalização de texto.
*   **Treinamento de Modelos de Linguagem de Fala (Speech Language Models):** Embora seja um dataset de TTS, é frequentemente usado em conjunto com outros datasets maiores (como LibriSpeech) para pré-treinamento de componentes de modelos de linguagem de fala.

## Integration
O dataset pode ser baixado diretamente do site do criador ou acessado através de plataformas como Hugging Face Datasets e Kaggle.

**Download Direto:**
1.  Acesse a URL principal: [https://keithito.com/LJ-Speech-Dataset/](https://keithito.com/LJ-Speech-Dataset/)
2.  Clique no link de download (2.6 GB).

**Uso com Python (Exemplo Hugging Face Datasets):**
Para uso em projetos de aprendizado de máquina, a integração mais comum é via biblioteca `datasets` do Hugging Face:

```python
from datasets import load_dataset

# Carrega a versão 1.1 do dataset
ljspeech_dataset = load_dataset("keithito/lj_speech")

# O dataset é carregado como um objeto DatasetDict com as divisões 'train', 'validation' e 'test' (se aplicável)
# A estrutura de dados inclui 'audio' (caminho do arquivo e array de áudio), 'text' (transcrição original) e 'normalized_text' (transcrição normalizada).
print(ljspeech_dataset)
```

**Estrutura de Arquivos:**
O dataset consiste em:
*   Um arquivo `metadata.csv` (ou `transcripts.csv`) com três campos por linha: `ID` (nome do arquivo WAV), `Transcription` (transcrição original) e `Normalized Transcription` (transcrição normalizada).
*   Uma pasta contendo 13.100 arquivos de áudio no formato `.wav`.

## URL
[https://keithito.com/LJ-Speech-Dataset/](https://keithito.com/LJ-Speech-Dataset/)
