# Deep Learning Architecture Prompts

## Description
"Deep Learning Architecture Prompts" (DLAP) é uma técnica avançada de Engenharia de Prompt que utiliza Large Language Models (LLMs) para auxiliar, automatizar ou guiar o processo de **Busca de Arquitetura Neural (NAS)**. Em vez de projetar manualmente a estrutura de uma rede neural (como o número de camadas, tipo de convolução, funções de ativação), o prompt é usado para instruir um LLM a gerar, modificar ou sugerir o código ou a estrutura da arquitetura.

Essa abordagem se baseia na capacidade dos LLMs de raciocinar sobre código e estruturas complexas, transformando a especificação de alto nível do problema (definida no prompt) em uma arquitetura de Deep Learning funcional. O conceito está intimamente ligado a metodologias como o **EvoPrompting**, onde o LLM atua como um operador adaptativo de mutação e cruzamento em um algoritmo evolutivo de NAS, otimizando a arquitetura em nível de código. O DLAP representa uma mudança de foco da engenharia manual de arquitetura para a engenharia de prompts que geram a arquitetura.

## Examples
```
1. **Geração de CNN para Classificação:** "Gere o código Python para uma arquitetura de Rede Neural Convolucional (CNN) usando PyTorch. A rede deve ser otimizada para classificação de imagens CIFAR-10 (10 classes, imagens 32x32x3). Inclua 3 blocos convolucionais, cada um seguido por ReLU e Max Pooling. O modelo final deve ter menos de 500.000 parâmetros."

2. **Design de GNN para Dados de Grafos:** "Projete uma arquitetura de Rede Neural Gráfica (GNN) usando o framework PyG (PyTorch Geometric) para uma tarefa de classificação de nós em um grafo de citação (dataset Cora). A arquitetura deve utilizar duas camadas Graph Convolutional Network (GCN) e uma camada de *dropout* de 0.5. Gere o código completo da classe do modelo."

3. **Otimização de Arquitetura Existente (EvoPrompting):** "Analise a seguinte arquitetura de rede neural (código fornecido). O objetivo é reduzir o número de parâmetros em 20% sem perder mais de 1% de acurácia no dataset MNIST. Sugira uma modificação na profundidade ou largura das camadas convolucionais e gere o código modificado."

4. **Especificação de Arquitetura Transformer:** "Crie um prompt detalhado para um LLM que gere uma arquitetura de Transformer para tradução de inglês para português. A arquitetura deve ter 6 camadas de encoder e 6 de decoder, com 8 cabeças de atenção (multi-head attention) e um tamanho de embedding de 512. O prompt deve solicitar o código em TensorFlow/Keras."

5. **Design de Rede Recorrente (RNN) para Séries Temporais:** "Gere uma arquitetura de rede neural para previsão de séries temporais (preço de ações). Use uma camada LSTM bidirecional com 128 unidades, seguida por uma camada densa de saída. O modelo deve ser implementado em PyTorch e aceitar sequências de entrada de 60 timesteps."

6. **Prompt de Restrição de Hardware:** "Desenvolva uma arquitetura de rede neural para detecção de objetos (bounding boxes) em tempo real, adequada para implantação em um dispositivo Edge com 4GB de RAM e latência máxima de 50ms por inferência. Sugira uma variante leve do YOLO ou MobileNet e forneça a especificação da arquitetura em formato YAML."
```

## Best Practices
**Especificidade da Tarefa e Dados:** O prompt deve detalhar o tipo de tarefa (ex: classificação de imagens, regressão de séries temporais) e as características do conjunto de dados (tamanho, dimensionalidade, tipo de dados). **Definição de Restrições:** Inclua restrições claras de hardware e desempenho, como latência máxima, memória disponível e número de parâmetros desejado. **Uso de Metodologias Estruturadas:** Empregue técnicas como o EvoPrompting (para otimização evolutiva) ou o Cognitive Prompt Architecture (para estruturação cognitiva do prompt) para guiar o LLM de forma mais eficaz. **Validação e Refinamento Iterativo:** Use o código ou a arquitetura gerada como ponto de partida. Valide o desempenho e use os resultados (ex: acurácia, perda) para refinar o prompt em iterações subsequentes. **Linguagem de Programação e Framework:** Especifique claramente a linguagem (Python) e o framework (TensorFlow, PyTorch) para garantir a geração de código funcional.

## Use Cases
**Busca de Arquitetura Neural (NAS) Acelerada:** Usar LLMs para explorar o espaço de busca de arquiteturas de forma mais eficiente do que métodos tradicionais de NAS baseados em algoritmos genéticos ou reforço. **Otimização de Arquiteturas para Edge Computing:** Gerar arquiteturas leves e eficientes que atendam a restrições rigorosas de latência e consumo de energia para dispositivos IoT e Edge. **Geração de Código de Prototipagem Rápida:** Criar rapidamente o código base de uma arquitetura complexa (ex: Transformer, GNN) para prototipagem e testes iniciais. **Transferência de Conhecimento de Domínio:** Incorporar *insights* de artigos de pesquisa recentes (via RAG) no prompt para guiar o LLM a projetar arquiteturas que utilizam as últimas inovações do campo. **Educação e Exploração:** Permitir que estudantes e pesquisadores explorem rapidamente as variações de arquiteturas para diferentes tarefas, entendendo o impacto das mudanças de hiperparâmetros.

## Pitfalls
**Vagueza na Especificação:** Prompts que não definem claramente a tarefa, o dataset ou as restrições de desempenho resultarão em arquiteturas genéricas e ineficazes. **Ignorar Restrições de Hardware:** Gerar uma arquitetura complexa sem considerar as limitações de memória ou poder de processamento do ambiente de implantação. **Falta de Validação:** Confiar cegamente no código gerado pelo LLM sem realizar testes de desempenho e *benchmarking* no dataset real. **Dependência Excessiva:** Usar o prompt para gerar a arquitetura inteira sem aplicar o conhecimento humano para refinar ou ajustar a estrutura, perdendo a oportunidade de incorporar *insights* específicos do domínio. **Foco Apenas no Código:** Concentrar-se apenas na geração do código da arquitetura e negligenciar o pipeline de dados, a função de perda e o esquema de otimização, que são cruciais para o treinamento.

## URL
[https://proceedings.neurips.cc/paper_files/paper/2023/hash/184c1e18d00d7752805324da48ad25be-Abstract-Conference.html](https://proceedings.neurips.cc/paper_files/paper/2023/hash/184c1e18d00d7752805324da48ad25be-Abstract-Conference.html)
