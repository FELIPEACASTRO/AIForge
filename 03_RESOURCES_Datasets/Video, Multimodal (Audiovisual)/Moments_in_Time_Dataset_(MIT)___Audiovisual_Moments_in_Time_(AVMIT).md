# Moments in Time Dataset (MIT) / Audiovisual Moments in Time (AVMIT)

## Description
O **Moments in Time Dataset (MIT)** é uma coleção massiva de vídeos, originalmente composta por um milhão de clipes de 3 segundos, cada um rotulado com uma de 339 classes de ação, para o reconhecimento e compreensão de eventos dinâmicos. A versão mais recente e relevante é o **Audiovisual Moments in Time (AVMIT)**, publicado em 2023, que se concentra em ações audiovisuais. O AVMIT é um subconjunto anotado do MIT, com foco na correspondência entre os fluxos de áudio e visual, tornando-o um recurso valioso para o treinamento de modelos multimodais.

## Statistics
**Moments in Time (MIT) Original:**
*   **Amostras:** 1 milhão de vídeos de 3 segundos.
*   **Classes:** 339 classes de ação.

**Audiovisual Moments in Time (AVMIT - 2023):**
*   **Anotações:** 171.630 anotações em 57.177 vídeos audiovisuais.
*   **Duração:** 23.160 vídeos (19.3 horas) de ações audiovisuais rotuladas.
*   **Conjunto de Teste Curado:** 960 vídeos (16 classes, 60 vídeos cada).
*   **Tamanho dos Arquivos (Embeddings):** 2.6 GB (total dos arquivos `.tar` no Zenodo).
*   **Versão:** 1 (Publicada em 16 de agosto de 2023).

## Features
**MIT Original:**
*   **Escala:** Um milhão de vídeos curtos (3 segundos).
*   **Classes:** 339 classes de ação rotuladas por humanos.
*   **Foco:** Capturar a essência de um evento dinâmico (ação).
*   **Diversidade:** Grande variação inter-classe e intra-classe (ex: "abrir" portas, presentes, olhos).

**AVMIT (Audiovisual Moments in Time - 2023):**
*   **Natureza:** Extensão audiovisual anotada do MIT.
*   **Análise Multimodal:** Focado na correspondência audiovisual (ação visual e som).
*   **Recursos Adicionais:** Oferece *embeddings* de características pré-computadas (VGGish/YamNet para áudio e VGG16/EfficientNetB0 para visual), facilitando a entrada para pesquisa em Redes Neurais Profundas (DNNs) audiovisuais.
*   **Conjunto de Teste Curado:** Inclui um conjunto de teste curado de 16 classes de ação distintas, com 60 vídeos cada, apropriado para experimentos controlados.

## Use Cases
*   **Reconhecimento de Ação em Vídeos:** Treinamento de modelos para identificar eventos dinâmicos em clipes curtos.
*   **Aprendizado Multimodal:** Pesquisa em modelos que integram informações visuais e auditivas para uma compreensão mais rica de eventos (especialmente com AVMIT).
*   **Transfer Learning:** Uso de modelos pré-treinados no MIT/AVMIT como *backbones* para tarefas de visão computacional e áudio em novos domínios.
*   **Análise de Correspondência Audiovisual:** Estudo da relação entre o que é visto e o que é ouvido em eventos do mundo real.

## Integration
O dataset original **Moments in Time** requer o preenchimento de um formulário de solicitação no site oficial do MIT-IBM Watson AI Lab para obter os links de download.

A versão mais recente, **Audiovisual Moments in Time (AVMIT)**, está disponível no Zenodo e inclui os *embeddings* de características pré-computadas, além de um arquivo CSV com os metadados do conjunto de teste.

**Passos para Integração (AVMIT):**
1.  **Download dos Arquivos:** Baixar os arquivos `.tar` e o `.csv` da página do Zenodo.
    *   `AVMIT_VGGish_VGG16.tar` (498.7 MB)
    *   `AVMIT_YamNet_EffNetB0.tar` (2.1 GB)
    *   `test_set.csv` (metadados do conjunto de teste)
2.  **Acesso aos Vídeos:** O AVMIT utiliza vídeos do MIT original. O arquivo `test_set.csv` contém a coluna `video_location` que indica a localização dos vídeos originais do MIT que devem ser usados.
3.  **Uso dos Embeddings:** Os arquivos `.tar` contêm os *embeddings* de características, que podem ser carregados diretamente em modelos de aprendizado de máquina para tarefas de reconhecimento de ação audiovisual.

**Observação:** Para obter os vídeos brutos do MIT, é necessário seguir o processo de solicitação no site oficial.

## URL
[http://moments.csail.mit.edu/](http://moments.csail.mit.edu/)
