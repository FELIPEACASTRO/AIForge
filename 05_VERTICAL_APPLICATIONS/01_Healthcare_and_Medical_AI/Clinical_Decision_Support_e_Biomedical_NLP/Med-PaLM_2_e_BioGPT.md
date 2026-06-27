# Med-PaLM 2 e BioGPT

## Description

Med-PaLM 2 é um modelo de linguagem grande (LLM) desenvolvido pelo Google Research, especificamente ajustado para o domínio médico. Ele é uma versão aprimorada do Med-PaLM, projetado para fornecer respostas de alta qualidade a perguntas médicas, auxiliar no raciocínio clínico e superar o desempenho de modelos generalistas em tarefas médicas. BioGPT é um modelo de linguagem grande (LLM) baseado na arquitetura Generative Pre-trained Transformer (GPT), desenvolvido pela Microsoft. Ele é especificamente pré-treinado em 15 milhões de resumos de literatura biomédica (PubMed), tornando-o altamente eficaz para tarefas de Processamento de Linguagem Natural (PLN) no domínio biomédico.

## Statistics

**Med-PaLM 2:** Atingiu até 86,5% de precisão no conjunto de dados MedQA, uma melhoria de mais de 19% em relação ao Med-PaLM. Demonstrou aumentos dramáticos de desempenho em conjuntos de dados como MedMCQA, PubMedQA e MMLU (tópicos clínicos). Em um estudo piloto, especialistas preferiram as respostas do Med-PaLM 2 às de médicos generalistas em 65% das vezes. **BioGPT:** Atingiu 78,2% de precisão em uma tarefa de resposta a perguntas biomédicas, superando o desempenho anterior em 6,0%. Métricas específicas de tarefas incluem: 44,98% de pontuação F1 no BC5CDR, 38,42% no KD-DTI e 40,76% no DDI (extração de relação de ponta a ponta).

## Features

**Med-PaLM 2:** Raciocínio clínico avançado, resposta a perguntas médicas, capacidade de exceder a pontuação de 'aprovação' em exames de licenciamento médico dos EUA, e melhorias em estratégias de 'grounding' e raciocínio através de refinamento de conjunto e cadeia de recuperação. **BioGPT:** Geração de texto biomédico, resposta a perguntas biomédicas, extração de relações (como interações droga-droga e droga-alvo), e reconhecimento de entidades nomeadas biomédicas.

## Use Cases

**Med-PaLM 2:** Resposta a perguntas médicas, suporte à decisão clínica, educação médica, e potencial para aplicações médicas no mundo real. **BioGPT:** Medicina genômica personalizada, diagnóstico de doenças raras, detecção automatizada de doenças, rastreamento de medicamentos, e suporte à pesquisa em bioinformática.

## Integration

**Med-PaLM 2:** Não há código de integração público disponível, pois é um modelo proprietário do Google. A integração é tipicamente via API ou serviços de nuvem do Google Cloud. **BioGPT:** O modelo está disponível no Hugging Face e a implementação de código aberto pode ser encontrada no GitHub. Exemplo de implementação via Hugging Face Transformers: `from transformers import BioGptTokenizer, BioGptForCausalLM`.

## URL

https://www.nature.com/articles/s41591-024-03423-7 e https://github.com/microsoft/BioGPT