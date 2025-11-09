# Visual Genome

## Description
O Visual Genome é um dataset e uma base de conhecimento que visa conectar conceitos visuais estruturados à linguagem. Ele fornece anotações densas para 108.077 imagens, incluindo objetos, atributos, relacionamentos e descrições de regiões, permitindo um entendimento mais profundo da cena do que apenas a classificação ou detecção de objetos. O dataset é fundamental para pesquisas em Visão Computacional e Processamento de Linguagem Natural, especialmente em tarefas que exigem raciocínio visual e compreensão contextual. A versão mais recente é a 1.4, lançada em julho de 2017, mas o dataset continua sendo uma base importante para trabalhos recentes, muitas vezes sendo modificado ou estendido (como no caso do VGARank e Synthetic Visual Genome).

## Statistics
Imagens: 108.077. Descrições de Regiões: 5.4 Milhões. Respostas a Perguntas Visuais (VQA): 1.7 Milhões. Instâncias de Objetos: 3.8 Milhões. Atributos: 2.8 Milhões. Relacionamentos: 2.3 Milhões. Versão mais recente: 1.4 (Julho de 2017). Tamanho total das imagens (v1.2): ~14.67 GB.

## Features
Anotações densas e detalhadas para cada imagem. Inclui objetos, atributos, e os relacionamentos entre eles (grafos de cena). Mapeamento de todos os conceitos para Wordnet Synsets. Contém descrições de regiões e pares de Perguntas e Respostas Visuais (VQA). É um dataset multimodal que conecta visão e linguagem.

## Use Cases
Geração de Legendas de Imagens (Image Captioning). Respostas a Perguntas Visuais (Visual Question Answering - VQA). Geração de Grafos de Cena (Scene Graph Generation). Raciocínio Visual e Compreensão Contextual. Detecção de Objetos e Atribuição de Atributos. Treinamento de modelos multimodais que exigem um entendimento profundo da relação entre objetos em uma cena.

## Integration
O dataset pode ser baixado em partes diretamente do site oficial (versões 1.0, 1.2 e 1.4). Os dados estão divididos em arquivos JSON e imagens. A versão 1.4, a mais recente, inclui arquivos separados para objetos, relacionamentos, aliases e synsets. O uso geralmente requer o download dos arquivos de anotação e das imagens (cerca de 14.67 GB no total para as imagens da v1.2). Também está disponível em plataformas como Hugging Face e Kaggle. É recomendado o uso da API e do tutorial fornecidos pelos criadores para facilitar o processamento e a integração.

## URL
[https://homes.cs.washington.edu/~ranjay/visualgenome/index.html](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html)
