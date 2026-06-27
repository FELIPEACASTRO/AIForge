# LibriSpeech ASR Corpus

## Description
O LibriSpeech é um corpus de grande escala (aproximadamente 1000 horas) de fala em inglês lida, projetado especificamente para o treinamento e avaliação de sistemas de Reconhecimento Automático de Fala (ASR). O dataset é derivado de audiolivros lidos do projeto LibriVox, que estão em domínio público. O corpus foi cuidadosamente segmentado e alinhado para fornecer transcrições precisas para cada segmento de áudio. É um dos benchmarks mais utilizados na pesquisa de ASR devido ao seu tamanho e à sua divisão em subconjuntos "limpos" (clean) e "outros" (other), que representam diferentes níveis de dificuldade de ruído e qualidade de gravação. Uma extensão notável é o Multilingual LibriSpeech (MLS), que expande o conceito para 8 idiomas, incluindo o português.

## Statistics
- **Tamanho Total da Fala:** Aproximadamente 1000 horas.
- **Tamanho Total do Arquivo:** Cerca de 60 GB (apenas áudio e transcrições).
- **Divisões Principais:**
    - `train-clean-100`: 100 horas (treinamento, limpo)
    - `train-clean-360`: 360 horas (treinamento, limpo)
    - `train-other-500`: 500 horas (treinamento, outros)
    - `dev-clean`: ~5 horas (desenvolvimento, limpo)
    - `dev-other`: ~5 horas (desenvolvimento, outros)
    - `test-clean`: ~5 horas (teste, limpo)
    - `test-other`: ~5 horas (teste, outros)
- **Versão Original:** Lançado em 2015. A versão mais recente é a original, mas o dataset continua sendo um benchmark padrão, com extensões como o Multilingual LibriSpeech (MLS) e o LibriSpeech-Long (Google DeepMind, 2024).

## Features
- **Grande Escala:** Aproximadamente 1000 horas de fala.
- **Domínio Público:** Derivado de audiolivros do LibriVox, sob licença CC BY 4.0.
- **Divisão de Dificuldade:** Subconjuntos de treinamento e teste divididos em "clean" (limpo, mais fácil) e "other" (outros, mais desafiador, com mais ruído ou variações de gravação).
- **Taxa de Amostragem:** Áudio em 16kHz.
- **Alinhamento Preciso:** Segmentos de fala cuidadosamente alinhados com suas transcrições.
- **Extensões:** Existem variações como o LibriSpeech-PC (com pontuação e capitalização restauradas) e o Multilingual LibriSpeech (MLS) com suporte a múltiplos idiomas.

## Use Cases
- **Reconhecimento Automático de Fala (ASR):** Treinamento e avaliação de modelos de ASR de última geração.
- **Processamento de Fala:** Pesquisa em segmentação, alinhamento e síntese de fala.
- **Transfer Learning:** Uso como dataset de pré-treinamento para modelos de ASR em outros idiomas ou domínios.
- **Avaliação de Modelos:** Serve como um benchmark padrão para comparar o desempenho de diferentes arquiteturas de modelos de fala (ex: RNNs, Transformers, Conformer).

## Integration
O dataset LibriSpeech pode ser baixado diretamente do site oficial do OpenSLR (SLR12) em vários arquivos `.tar.gz`, cada um correspondendo a uma divisão específica (treinamento, desenvolvimento, teste).

**Download Direto (OpenSLR):**
O usuário deve baixar os arquivos de interesse (ex: `train-clean-100.tar.gz`, `test-clean.tar.gz`) e descompactá-los.

**Integração com Bibliotecas:**
Muitas bibliotecas de aprendizado de máquina e processamento de áudio, como o **Torchaudio** (PyTorch), oferecem APIs para download e carregamento simplificado do LibriSpeech.

*Exemplo com Torchaudio (Python):*
```python
import torchaudio

# O Torchaudio gerencia o download e o carregamento
dataset = torchaudio.datasets.LIBRISPEECH(
    root="/caminho/para/dados",
    url="train-clean-100", # ou "test-clean", "dev-other", etc.
    download=True
)
# O dataset pode ser iterado para obter (waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id)
```
O dataset também está disponível no **Hugging Face Datasets** (`openslr/librispeech_asr`), facilitando o uso em pipelines de modelos de linguagem.

## URL
[https://www.openslr.org/12](https://www.openslr.org/12)
