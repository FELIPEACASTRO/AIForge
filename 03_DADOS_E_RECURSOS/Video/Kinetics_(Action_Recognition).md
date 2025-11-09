# Kinetics (Action Recognition)

## Description
O Kinetics é uma coleção de datasets de larga escala e alta qualidade, projetados para o reconhecimento de ações humanas em vídeos. O dataset consiste em links de URLs para clipes de vídeo do YouTube, abrangendo interações humano-objeto (como tocar instrumentos) e interações humano-humano (como apertar as mãos). Cada clipe tem aproximadamente 10 segundos de duração e é anotado com uma única classe de ação. A versão mais recente e abrangente é a Kinetics-700-2020.

## Statistics
O dataset Kinetics possui múltiplas versões: Kinetics-400 (400 classes), Kinetics-600 (600 classes) e Kinetics-700-2020 (700 classes). A versão Kinetics-700-2020 é a mais atualizada e contém aproximadamente 635.000 clipes de vídeo no total. O tamanho total do dataset (vídeos) é de cerca de 710 GB. As divisões de dados (CVDF split) são: Treino (534.073 vídeos), Teste (64.260 vídeos) e Validação (33.914 vídeos).

## Features
Larga escala com 700 classes de ações humanas. Alta qualidade de anotação, com cada clipe anotado por humanos. Foco em ações humanas dinâmicas e interações. Os vídeos são curtos (cerca de 10s) e extraídos do YouTube. O dataset é frequentemente usado como benchmark para modelos de reconhecimento de ação.

## Use Cases
Treinamento e avaliação de modelos de reconhecimento de ação em vídeo (Action Recognition). Pesquisa em visão computacional e aprendizado profundo para análise de vídeo. Transferência de aprendizado (pre-training) para tarefas de vídeo mais específicas. Desenvolvimento de sistemas de vigilância e análise de comportamento.

## Integration
O dataset consiste em URLs do YouTube. Devido à volatilidade dos links, a CVDF (Computer Vision Foundation) hospeda os vídeos no AWS S3. A integração é tipicamente feita através de scripts de download (disponíveis no GitHub oficial) que baixam os vídeos a partir das URLs ou dos arquivos tar.gz hospedados. Ferramentas como o FiftyOne também oferecem métodos simplificados para carregar e gerenciar o dataset (ex: `foz.load_zoo_dataset(\"kinetics-700-2020\")`). É necessário ter o `ffmpeg` instalado para trabalhar com os arquivos de vídeo.

## URL
[https://deepmind.google/research/open-source/kinetics](https://deepmind.google/research/open-source/kinetics)
