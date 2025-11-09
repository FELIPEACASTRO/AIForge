# YouTube-8M

## Description
O **YouTube-8M** é um *dataset* de vídeo em larga escala, rotulado e diversificado, desenvolvido pelo Google Research para acelerar a pesquisa em compreensão de vídeo em escala, aprendizado de representação, modelagem de dados ruidosos, aprendizado por transferência e adaptação de domínio. O *dataset* consiste em milhões de IDs de vídeos do YouTube, com anotações de alta qualidade geradas por máquina a partir de um vocabulário de mais de 3.800 entidades visuais. Ele é notável por fornecer *features* audiovisuais pré-computadas, permitindo que pesquisadores treinem modelos de linha de base robustos em menos de um dia usando uma única GPU. A versão mais recente (Junho de 2019) inclui o **YouTube-8M Segments**, uma extensão com cerca de 237 mil segmentos rotulados e verificados por humanos em 1.000 classes, focando na localização temporal de entidades em vídeos de 5 segundos. O *dataset* é disponibilizado sob a licença Creative Commons Attribution 4.0 International (CC BY 4.0).

## Statistics
**Versão de Junho de 2019 (Segments - Atual):**
- **Amostras:** 230K segmentos rotulados e verificados por humanos.
- **Classes:** 1.000 classes.
- **Formato:** *Features* de nível de *frame* (Frame-level features).

**Versão de Maio de 2018 (Vídeo - Atual):**
- **Vídeos:** 6.1 Milhões de IDs de vídeos.
- **Duração:** 350.000 horas de vídeo.
- **Features:** 2.6 Bilhões de *features* audiovisuais.
- **Classes:** 3.862 classes.
- **Tamanho Total (Frame-level features):** 1.53 Terabytes.
- **Tamanho Total (Video-level features):** 31 Gigabytes.
- **Rótulos:** Média de 3.0 rótulos por vídeo.

## Features
- **Escala Massiva:** Milhões de vídeos e bilhões de *features* audiovisuais pré-computadas.
- **Diversidade:** Vídeos amostrados uniformemente para preservar a distribuição de conteúdo popular do YouTube.
- **Features Pré-computadas:** Fornece *features* de nível de vídeo (média de *features* RGB e áudio) e *features* de nível de *frame* (RGB e áudio a cada segundo).
- **Rótulos Multi-rótulo:** Cada vídeo possui múltiplos rótulos associados a entidades do Knowledge Graph.
- **Segmentos Verificados por Humanos:** A versão Segments adiciona anotações verificadas por humanos para localização temporal de entidades.
- **Fácil Acesso:** Os dados são fornecidos como arquivos TensorFlow Record, otimizados para treinamento em larga escala.

## Use Cases
- **Classificação de Vídeo em Larga Escala:** Treinamento e avaliação de modelos para categorizar vídeos com múltiplos rótulos.
- **Localização Temporal de Eventos:** Uso da versão Segments para identificar o momento exato em que uma entidade ou evento ocorre no vídeo.
- **Aprendizado de Representação:** Desenvolvimento de novas arquiteturas de rede neural para extrair *features* de vídeo e áudio.
- **Modelagem de Dados Ruidosos:** Pesquisa sobre como lidar com anotações geradas por máquina que podem conter ruído.
- **Aprendizado por Transferência e Adaptação de Domínio:** Utilização do *dataset* como base para transferir conhecimento para tarefas de vídeo mais específicas.

## Integration
O *dataset* é distribuído como arquivos **TensorFlow Record** e pode ser baixado usando um *script* Python fornecido pelo Google Research.
1.  **Instalação:** Certifique-se de ter Python e `curl` instalados.
2.  **Download do Script:** O *script* de download (`download.py`) é acessado via `curl` e executado com Python.
3.  **Estrutura de Diretórios:** Crie um diretório para os dados, por exemplo: `mkdir -p ~/data/yt8m/video; cd ~/data/yt8m/video`.
4.  **Download (Exemplo para *Video-level features*):**
    ```bash
    curl data.yt8m.org/download.py | partition=2/video/train mirror=us python
    curl data.yt8m.org/download.py | partition=2/video/validate mirror=us python
    curl data.yt8m.org/download.py | partition=2/video/test mirror=us python
    ```
    - **Espelhamento (*Mirror*):** Substitua `mirror=us` por `mirror=eu` ou `mirror=asia` para acelerar a transferência dependendo da sua localização.
    - **Subamostragem:** É possível baixar uma fração do *dataset* usando o parâmetro `shard=1,100` para 1/100 dos dados.
5.  **Código Inicial (*Starter Code*):** O Google Research fornece um repositório GitHub com código inicial para treinamento e avaliação de modelos.
    - **Frame-level features:** Requerem cerca de 1.53 TB de espaço em disco.
    - **Video-level features:** Requerem cerca de 31 GB de espaço em disco.

## URL
[https://research.google.com/youtube8m/](https://research.google.com/youtube8m/)
