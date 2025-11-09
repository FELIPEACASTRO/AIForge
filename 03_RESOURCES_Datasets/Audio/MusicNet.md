# MusicNet

## Description
MusicNet é uma coleção de 330 gravações de música clássica com licença livre, totalizando mais de 1 milhão de rótulos anotados que indicam o tempo preciso de cada nota, o instrumento que a toca e sua posição na estrutura métrica da composição. Os rótulos são adquiridos a partir de partituras musicais alinhadas às gravações por meio de "dynamic time warping". O dataset é um recurso fundamental para o treinamento de modelos de aprendizado de máquina e um benchmark comum para tarefas de transcrição musical automática. Uma versão mais recente, MusicNet-16k + EM for YourMT3 (Abril de 2023), oferece o áudio reamostrado para 16 kHz e rótulos refinados (MusicNet EM) para tarefas específicas.

## Statistics
**Gravações:** 330 gravações de música clássica. **Rótulos:** Mais de 1 milhão de rótulos anotados. **Tamanho:** O arquivo principal `musicnet.tar.gz` tem 11.1 GB. **Versões:** Versão 1.0 (Novembro de 2016). Versão MusicNet-16k + EM for YourMT3 (v6, Abril de 2023), com 6.7 GB. **Duração:** O dataset original contém mais de 330 horas de áudio.

## Features
Contém áudio PCM-encoded (.wav) e rótulos de notas correspondentes em formato CSV. Inclui metadados em nível de faixa e arquivos MIDI de referência. Os rótulos são verificados por músicos treinados, com uma taxa de erro estimada em 4%. A versão MusicNet-16k oferece áudio reamostrado para 16 kHz (mono, 16-bit) e rótulos refinados (MusicNet EM) para melhor desempenho em tarefas de transcrição.

## Use Cases
**Transcrição Automática de Música (AMT):** Principal aplicação para treinar modelos que convertem áudio em notação musical. **Reconhecimento de Instrumentos:** Identificação do instrumento que toca cada nota. **Análise de Estrutura Métrica:** Estudo da posição da nota na estrutura rítmica e métrica da composição. **Pesquisa em Aprendizado de Máquina para Música:** Serve como benchmark para comparar o desempenho de diferentes métodos.

## Integration
O dataset original pode ser baixado em três arquivos principais: `musicnet.tar.gz` (áudio e rótulos CSV), `musicnet_metadata.csv` (metadados) e `musicnet_midis.tar.gz` (arquivos MIDI de referência). O acesso e uso são facilitados por uma interface PyTorch disponível no GitHub, que permite carregar e processar os dados de forma eficiente. Para a versão MusicNet-16k, o download é feito diretamente pelo Zenodo, e as instruções de uso e divisões de dados (splits) estão no repositório do projeto YourMT3.

## URL
[https://zenodo.org/records/5120004](https://zenodo.org/records/5120004)
