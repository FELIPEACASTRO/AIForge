# ActivityNet

## Description
O ActivityNet é um benchmark de vídeo em larga escala projetado para o entendimento de atividades humanas. Seu objetivo é cobrir uma ampla gama de atividades complexas de interesse na vida diária. O dataset é usado para comparar algoritmos em tarefas como classificação global de vídeo, classificação de atividades segmentadas (trimmed) e detecção de atividades. A versão principal é a ActivityNet 1.3, que inclui 200 classes de atividades. O dataset é notável por sua estrutura de ontologia semântica, que organiza as atividades de acordo com relações sociais e locais de ocorrência, fornecendo uma hierarquia rica com pelo menos quatro níveis de profundidade. O dataset original foi apresentado em 2015, mas continua sendo uma referência fundamental e é frequentemente usado em desafios e pesquisas recentes (2023-2025) através de suas extensões como ActivityNet-QA e ActivityNet-Entities.

## Statistics
- **Versão Principal:** ActivityNet 1.3 (lançada em 2016, mas a base para pesquisa recente).
- **Classes de Atividades:** 200 classes.
- **Vídeos Totais:** Aproximadamente 20.000 vídeos (10.024 treino, 4.926 validação, 5.044 teste).
- **Instâncias de Atividades:** 15.410 instâncias de atividades no conjunto de treino.
- **Duração Total:** 849 horas de vídeo.
- **Média:** 1.54 instâncias de atividade por vídeo.

## Features
- **Escala e Diversidade:** 200 classes de atividades, 20.000 vídeos no total (treino, validação e teste) e 849 horas de vídeo.
- **Ontologia Semântica:** Estrutura hierárquica de atividades com quatro níveis de profundidade, permitindo o estudo de relações entre atividades.
- **Anotações Detalhadas:** Fornece anotações de instâncias de atividades com segmentos de tempo (início e fim) em vídeos não segmentados (untrimmed).
- **Extensões:** Serviu de base para datasets mais recentes como ActivityNet-QA (Perguntas e Respostas sobre Vídeos) e ActivityNet-Entities (Anotações de caixas delimitadoras de entidades).
- **Integração com FiftyOne:** Suporte nativo para carregamento, visualização e avaliação através da ferramenta de código aberto FiftyOne.

## Use Cases
- **Detecção e Reconhecimento de Atividades Humanas:** Principalmente em vídeos não segmentados (untrimmed).
- **Classificação de Vídeo:** Classificação global de vídeos.
- **Localização Temporal de Ações:** Identificação precisa dos segmentos de tempo onde as atividades ocorrem.
- **Pesquisa em Visão Computacional:** Benchmark para o desenvolvimento e comparação de novos algoritmos de entendimento de vídeo.
- **Perguntas e Respostas Visuais (VQA):** Usado como base para o dataset ActivityNet-QA.
- **Grounding de Entidades:** Usado como base para o dataset ActivityNet-Entities, que adiciona anotações de caixas delimitadoras (bounding boxes) para entidades mencionadas nas legendas.

## Integration
O dataset ActivityNet (versões 100 e 200) pode ser facilmente carregado, visualizado e avaliado usando a ferramenta de código aberto **FiftyOne** da Voxel51.

**Instruções de Uso com FiftyOne:**
1.  **Instalação:** Instale o FiftyOne via pip: `pip install fiftyone`
2.  **Carregamento:** Use o `fiftyone.zoo` para carregar o dataset, especificando a versão e o split desejado. Por exemplo, para a versão 200 (ActivityNet 1.3):
    ```python
    import fiftyone.zoo as foz
    dataset = foz.load_zoo_dataset("activitynet-200", split="validation")
    ```
3.  **Download dos Vídeos:** Para obter os vídeos completos, é necessário preencher um formulário de solicitação no site oficial para ter acesso temporário aos arquivos hospedados no Google Drive ou Baidu Drive.

**Estrutura de Anotação:**
As anotações são fornecidas em arquivos JSON contendo a base de dados, a taxonomia hierárquica e a versão. A chave "database" contém informações do vídeo (duração, URL, subset) e a chave "annotations" lista as instâncias de atividades com `label` e `segment` (tempo de início e fim em segundos).

## URL
[http://activity-net.org/](http://activity-net.org/)
