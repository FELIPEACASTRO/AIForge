# VizWiz-VQA (Visual Question Answering)

## Description
O VizWiz-VQA é um dataset de Questionamento Visual de Perguntas (VQA) único, pois é o primeiro a ser construído a partir de perguntas visuais reais feitas por pessoas cegas e com baixa visão. As imagens e perguntas foram coletadas através de um aplicativo móvel, onde os usuários tiravam uma foto e gravavam uma pergunta falada sobre ela. O dataset é crucial para o desenvolvimento de tecnologias assistivas, pois reflete os desafios visuais cotidianos enfrentados por essa população. A versão mais recente (a partir de janeiro de 2023) substituiu respostas como "inadequado" por "não respondível" para maior clareza e possui um conjunto de dados maior e mais limpo. O dataset suporta tarefas de VQA, incluindo a previsão da resposta correta e a previsão da capacidade de resposta da pergunta.

## Statistics
- **Versão Atualizada (Jan/2023):**
  - **Treinamento:** 20.523 pares imagem/pergunta e 205.230 pares resposta/confiança.
  - **Validação:** 4.319 pares imagem/pergunta e 43.190 pares resposta/confiança.
  - **Teste:** 8.000 pares imagem/pergunta.
- **Total de Amostras:** 32.842 pares imagem/pergunta.
- **Versões:** A versão de Jan/2023 é a mais recente, substituindo a versão anterior de Dez/2019. A principal mudança foi a substituição de "unsuitable" por "unanswerable" e a expansão do conjunto de treinamento e validação.

## Features
- **Origem Real:** Imagens e perguntas coletadas de pessoas cegas e com baixa visão em situações cotidianas.
- **Formato Multimodal:** Combina imagens e perguntas em linguagem natural (originalmente faladas).
- **Anotação Detalhada:** Cada pergunta visual possui 10 respostas crowdsourced, permitindo uma avaliação robusta.
- **Desafio de Resposta:** Inclui a tarefa de prever se uma pergunta visual pode ser respondida, abordando a qualidade da imagem e a clareza da pergunta.
- **Versão Atualizada:** A versão de 2023 possui um esquema de anotação aprimorado e um conjunto de dados maior e mais limpo.

## Use Cases
- **Tecnologias Assistivas:** Desenvolvimento de sistemas de VQA que podem ajudar pessoas cegas e com baixa visão a obter informações sobre o mundo ao seu redor.
- **Pesquisa em VQA:** Treinamento e avaliação de modelos de VQA em um cenário de dados do mundo real, com imagens de baixa qualidade e perguntas complexas.
- **Análise de Qualidade de Imagem:** Estudo da relação entre a qualidade da imagem (muitas vezes ruim devido à deficiência visual do usuário) e a capacidade de resposta das perguntas.
- **Privacidade Visual:** O dataset VizWiz-Priv, um subconjunto relacionado, é usado para reconhecer a presença e o propósito de informações visuais privadas.

## Integration
O dataset VizWiz-VQA pode ser baixado diretamente do site oficial (vizwiz.org).
1. **Download dos Arquivos:** Baixe os conjuntos de imagens (treinamento, validação e teste) e os arquivos de anotações JSON (treinamento, validação e teste) através dos links fornecidos na seção "Dataset" da página VQA.
2. **Estrutura dos Arquivos:** Os arquivos JSON contêm os detalhes de cada pergunta visual, incluindo a imagem, a pergunta, o tipo de resposta e as 10 respostas crowdsourced com seus níveis de confiança.
3. **Código de Exemplo:** O site fornece código de exemplo e APIs para demonstrar como analisar os arquivos JSON e avaliar os métodos em relação ao gabarito.
4. **Submissão para Desafios:** Para participar dos desafios, os resultados devem ser submetidos ao servidor de avaliação EvalAI, seguindo as instruções específicas para as partições `test-dev`, `test-challenge` e `test-standard`.

## URL
[https://vizwiz.org/tasks-and-datasets/vqa/](https://vizwiz.org/tasks-and-datasets/vqa/)
