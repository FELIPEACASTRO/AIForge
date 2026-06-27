# GQA (Scene Graph Question Answering)

## Description
O GQA (Scene Graph Question Answering) é um dataset inovador para **Raciocínio Visual no Mundo Real e Resposta a Perguntas Composicionais (VQA)**. Ele foi criado para superar as limitações de datasets VQA anteriores, que eram suscetíveis a vieses de linguagem e falta de composicionalidade semântica. O GQA utiliza **Grafos de Cena** detalhados para representar objetos, atributos e relações nas imagens, e **Programas Funcionais** para estruturar a lógica de raciocínio das perguntas. Isso permite um diagnóstico mais preciso do desempenho dos modelos e incentiva o desenvolvimento de sistemas de VQA mais robustos e interpretáveis.

## Statistics
- **Imagens:** 113K imagens.
- **Perguntas:** Mais de 22 milhões de perguntas diversas de raciocínio.
- **Versões e Tamanhos de Download (Dados Principais):**
    - Scene Graphs (ver 1.1): 42.7MB
    - Questions (ver 1.2): 1.4GB
    - Images (ver 1.1): 73.9GB (Total) ou 20.3GB (Arquivos de Imagem)
- **Versão Balanceada:** 1.7M perguntas.

## Features
- **Raciocínio Composicional:** As perguntas exigem múltiplas habilidades de raciocínio, compreensão espacial e inferência em várias etapas.
- **Grafos de Cena:** Cada imagem é associada a um grafo de cena detalhado (objetos, atributos e relações), baseado no Visual Genome, mas refinado.
- **Programas Funcionais:** Cada pergunta é associada a uma representação estruturada de sua semântica, um programa funcional que especifica as etapas de raciocínio necessárias para respondê-la.
- **Métricas Aprimoradas:** Inclui novas métricas para testar a consistência, validade e plausibilidade das respostas dos modelos, além da precisão.
- **Dataset Balanceado:** Uma versão balanceada de 1.7M perguntas foi criada para mitigar vieses de linguagem.

## Use Cases
- Desenvolvimento e avaliação de modelos de **Raciocínio Visual** e **VQA Composicional**.
- Pesquisa em **Compreensão de Cena** e **Interpretabilidade de Modelos** (devido aos programas funcionais e grafos de cena).
- Treinamento de modelos para serem mais robustos a viesos de linguagem e condicionais.

## Integration
O dataset pode ser baixado diretamente da página oficial da Stanford (consulte a URL principal). Os componentes principais são:
1. **Scene Graphs:** Arquivo `scene_graphs.json` (ver 1.1 / 42.7MB).
2. **Questions:** Arquivo `questions.json` (ver 1.2 / 1.4GB).
3. **Images:** Arquivo `images.zip` (ver 1.1 / 20.3GB para arquivos de imagem).
A página de download também oferece Spatial Features (32.1GB) e Object Features (21.4GB) separadamente. É necessário concordar com os termos de uso para realizar o download.

## URL
[https://cs.stanford.edu/people/dorarad/gqa/](https://cs.stanford.edu/people/dorarad/gqa/)
