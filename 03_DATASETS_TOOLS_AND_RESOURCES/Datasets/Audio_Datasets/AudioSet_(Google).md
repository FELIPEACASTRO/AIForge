# AudioSet (Google)

## Description
O AudioSet é um dataset de grande escala composto por uma coleção de **2.084.320** clipes de áudio de 10 segundos, extraídos de vídeos do YouTube e rotulados por humanos. O objetivo principal é fornecer um recurso abrangente para o treinamento de modelos de classificação de eventos de áudio. O dataset é baseado em uma **ontologia** hierárquica de **632 classes** de eventos sonoros, cobrindo uma ampla gama de sons do cotidiano, desde sons humanos e de animais até sons naturais, ambientais e musicais. O dataset não fornece os arquivos de áudio brutos diretamente, mas sim os metadados (IDs de vídeo do YouTube, tempos de início e fim) e *embeddings* de áudio de 128 dimensões extraídos a 1Hz usando o modelo VGGish.

## Statistics
- **Tamanho Total:** 2.084.320 segmentos de áudio.
- **Duração dos Segmentos:** 10 segundos.
- **Classes:** 632 classes de eventos sonoros.
- **Divisão:**
    - Avaliação: 20.383 segmentos.
    - Treinamento Balanceado: 22.176 segmentos.
    - Treinamento Não Balanceado: 2.042.985 segmentos.
- **Tamanho dos *Embeddings*:** 2.4 GB (em formato TensorFlow Record).
- **Versão:** A versão principal referenciada é a "v1" (lançamento inicial de 2017), com atualizações de qualidade de rótulos (rerating) incluídas.

## Features
- **Ontologia Extensa:** 632 classes de eventos sonoros organizadas hierarquicamente.
- **Grande Escala:** Mais de 2 milhões de segmentos de áudio rotulados.
- **Duração Fixa:** Todos os clipes de áudio têm 10 segundos de duração.
- **Metadados e *Embeddings*:** Fornece IDs de vídeo do YouTube, metadados de tempo e *embeddings* de áudio (features) em vez dos arquivos de áudio brutos.
- **Divisão em Subconjuntos:** Dividido em conjuntos de avaliação (20.383 segmentos), treinamento balanceado (22.176 segmentos) e treinamento não balanceado (2.042.985 segmentos).

## Use Cases
- **Classificação de Eventos de Áudio (AEC):** Treinamento de modelos para identificar e classificar sons em gravações.
- **Marcação Automática de Conteúdo:** Aplicação em plataformas de vídeo (como o YouTube) para categorizar e indexar conteúdo com base no áudio.
- **Sistemas de Vigilância e Monitoramento:** Detecção de eventos sonoros específicos (ex: alarmes, tiros, choro de bebê).
- **Pesquisa em Processamento de Áudio:** Desenvolvimento de novas arquiteturas de rede neural e técnicas de *embedding* de áudio (como o VGGish).
- **Transfer Learning:** Uso dos *embeddings* pré-treinados (VGGish) como *features* para tarefas de áudio relacionadas.

## Integration
O dataset é disponibilizado em dois formatos:
1.  **Arquivos CSV:** Contêm os metadados de cada segmento: ID do vídeo do YouTube, tempo de início, tempo de fim e rótulos (classes). Os arquivos principais são `eval_segments.csv`, `balanced_train_segments.csv` e `unbalanced_train_segments.csv`.
2.  ***Embeddings* de Áudio (Features):** *Features* de áudio de 128 dimensões extraídas a 1Hz, armazenadas como arquivos TensorFlow Record (2.4 GB no total).

**Acesso aos *Embeddings*:**
Os *embeddings* podem ser baixados manualmente via `tar.gz` de *buckets* do Google Cloud Storage (GCS) ou usando o utilitário `gsutil` para sincronização:
`gsutil rsync -d -r features gs://{região}_audioset/youtube_corpus/v1/features` (onde {região} é 'us', 'eu' ou 'asia').

**Acesso ao Áudio Bruto:**
O dataset não fornece os arquivos de áudio brutos. É necessário usar os IDs de vídeo do YouTube e os metadados de tempo para baixar e extrair os clipes de áudio dos vídeos originais do YouTube, o que requer ferramentas de terceiros e está sujeito à disponibilidade do vídeo. O modelo VGGish e o código de suporte estão disponíveis no repositório GitHub do TensorFlow models.

## URL
[https://research.google.com/audioset/](https://research.google.com/audioset/)
