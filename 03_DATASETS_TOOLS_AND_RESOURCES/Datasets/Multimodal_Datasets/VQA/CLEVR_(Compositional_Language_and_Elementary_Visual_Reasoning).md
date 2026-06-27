# CLEVR (Compositional Language and Elementary Visual Reasoning)

## Description
O CLEVR (Compositional Language and Elementary Visual Reasoning) é um dataset de diagnóstico sintético projetado para testar uma ampla gama de habilidades de raciocínio visual e linguístico. Foi criado para abordar os vieses presentes em benchmarks anteriores de Resposta a Perguntas Visuais (VQA), garantindo que os modelos não possam explorar atalhos estatísticos para responder corretamente sem realmente raciocinar. O dataset consiste em imagens renderizadas de cenas 3D simples contendo objetos (cubos, cilindros, esferas) com atributos variados (cor, forma, tamanho, material) e perguntas complexas geradas programaticamente sobre essas cenas. Cada pergunta é acompanhada por uma representação de "programa funcional" que define a sequência exata de operações lógicas e visuais necessárias para chegar à resposta, permitindo uma avaliação precisa das capacidades de raciocínio dos modelos.

## Statistics
**Versão Principal:** CLEVR v1.0. **Tamanho do Download (v1.0):** Aproximadamente 18 GB (com imagens) ou 86 MB (somente anotações). **Tamanho do Dataset (TFDS):** 17.75 GiB. **Divisões (Main Dataset):** Treinamento (70.000 imagens, 699.989 perguntas), Validação (15.000 imagens, 149.991 perguntas), Teste (15.000 imagens, 14.988 perguntas). **Versão CoGenT:** 24 GB (com imagens) ou 106 MB (somente anotações), com divisões específicas para teste de generalização composicional. **Versão TFDS:** 3.1.0 (Adiciona texto de pergunta/resposta).

## Features
**Composição e Atributos:** Cenas 3D sintéticas com objetos simples (cubos, cilindros, esferas) variando em cor (8), forma (3), tamanho (2) e material (2). **Perguntas Complexas:** Perguntas geradas programaticamente que exigem raciocínio composicional, incluindo identificação de atributos, contagem, comparação, relações espaciais e operações lógicas. **Anotações Detalhadas:** Inclui grafos de cena (scene graphs) com localização, atributos e relações de objetos, e programas funcionais para cada pergunta, que servem como verdade fundamental para o raciocínio. **Mínimo Viés:** Projetado para ter um viés mínimo, forçando os modelos a realizar raciocínio visual e linguístico genuíno. **Extensões:** Possui extensões como CLEVR-CoGenT (para generalização composicional) e CLEVR-X (para explicações em linguagem natural).

## Use Cases
**Avaliação de Modelos VQA:** Serve como um benchmark diagnóstico rigoroso para modelos de Resposta a Perguntas Visuais (VQA), focando na capacidade de raciocínio em vez de atalhos estatísticos. **Raciocínio Composicional:** Usado para testar a capacidade dos modelos de combinar conceitos visuais e linguísticos de forma sistemática. **Generalização:** A versão CoGenT é usada para avaliar a generalização de modelos para novas combinações de atributos não vistas no treinamento. **Interpretabilidade:** O uso de programas funcionais auxilia no desenvolvimento de modelos VQA mais interpretáveis, onde o processo de raciocínio pode ser rastreado. **Desenvolvimento de Modelos:** Utilizado para treinar e validar arquiteturas de redes neurais focadas em atenção, memória e módulos de raciocínio explícito.

## Integration
O dataset CLEVR pode ser baixado diretamente da página oficial do projeto (links para as versões v1.0 e CoGenT, com e sem imagens). Para integração em projetos de aprendizado de máquina, é recomendado o uso de bibliotecas de datasets como o TensorFlow Datasets (TFDS) ou o Hugging Face Datasets, que oferecem a versão `clevr` para carregamento e pré-processamento simplificados.

**Exemplo de uso com TensorFlow Datasets (Python):**
```python
import tensorflow_datasets as tfds

# Carrega o dataset
ds = tfds.load('clevr', split='train', shuffle_files=True)

# Itera sobre os exemplos
for example in ds.take(1):
    print(example)
```
Alternativamente, os arquivos brutos (imagens e JSONs de perguntas/anotações) podem ser baixados e processados manualmente. O código de geração do dataset também está disponível no GitHub para renderizar novas imagens ou gerar novas perguntas.

## URL
[https://cs.stanford.edu/people/jcjohns/clevr/](https://cs.stanford.edu/people/jcjohns/clevr/)
