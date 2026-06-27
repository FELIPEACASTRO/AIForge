# ROT Prompting, Chain-of-Thought (CoT) e HealthBench

## Description

Pesquisa abrangente sobre técnicas de engenharia de prompt e avaliação de Large Language Models (LLMs) no contexto de **Recomendação de Tratamento** em saúde, com foco em desenvolvimentos de 2023 a 2025. Foram identificadas e detalhadas duas abordagens principais de prompt: o **ROT Prompting** (Role-Playing, Output-Constrained, and Thought-Process) e o **Chain-of-Thought (CoT)**, além do benchmark **HealthBench** para avaliação de LLMs médicos.

**ROT Prompting** demonstrou ser eficaz para aumentar a adesão de LLMs (como o GPT-4) a diretrizes clínicas baseadas em evidências, alcançando até 77.5% de consistência para recomendações fortes em um estudo sobre osteoartrite.

O **Chain-of-Thought (CoT)** é crucial para a transparência e rastreabilidade do raciocínio clínico, com modelos como o o1-mini atingindo alta precisão (88.4%) em tarefas de resumo de alta médica.

O **HealthBench** serve como um benchmark robusto, com 5.000 conversas multi-turn avaliadas por médicos, medindo o desempenho e a segurança dos LLMs em sete temas de saúde, incluindo precisão e qualidade da comunicação. Modelos de fronteira recentes (como o3 e GPT-4.1) demonstraram melhorias significativas.

## Statistics

**ROT Prompting:** No estudo de Wang et al. (2024) sobre as diretrizes de osteoartrite da AAOS, a combinação gpt-4-Web com ROT prompting alcançou a maior consistência geral (62.9%) e uma consistência de 77.5% para recomendações fortes.

**CoT Prompting:** O estudo de Jeon et al. (2025) mostrou que o modelo o1-mini alcançou 88.4% de precisão em tarefas de resumo de alta médica (EHRNoteQA) e 83.5% em notas clínicas usando CoT.

**HealthBench:** O desempenho varia de 0.16 (GPT-3.5 Turbo) a 0.60 (o3). O desempenho melhorou 28% nos modelos de fronteira da OpenAI. O eixo de 'Accuracy' (Precisão) representa 33% de todos os critérios de rubrica.

## Features

**ROT Prompting:** Melhora a adesão a diretrizes clínicas baseadas em evidências; Aumenta a consistência e confiabilidade das respostas em tarefas médicas complexas; Combina elementos de Role-Playing e Chain-of-Thought (CoT); Mais eficaz para recomendações de tratamento de alta evidência (nível forte).

**Chain-of-Thought (CoT):** Aumenta a transparência e a interpretabilidade do raciocínio do LLM; Melhora a precisão em tarefas de Question Answering (QA) e resumo de notas clínicas; Permite a intervenção de profissionais de saúde em qualquer ponto da cadeia de raciocínio (CoT Interativo).

**HealthBench:** Avaliação baseada em rubrica e consenso médico; 5.000 conversas multi-turn realistas; Sete temas (incluindo tarefas de dados de saúde e referências de emergência) e cinco eixos (incluindo precisão e completude); Duas variações: HealthBench Consensus (maior precisão) e HealthBench Hard (mais difícil).

## Use Cases

**Gerais:** Sistemas de Suporte à Decisão Clínica (CDSS); Avaliação da conformidade de LLMs com protocolos médicos estabelecidos; Resumo e análise de registros eletrônicos de saúde (EHR); Treinamento e avaliação de estudantes de medicina.

**Específicos:** Geração de recomendações de tratamento para condições médicas (ex: osteoartrite); Medição de segurança e confiabilidade de LLMs em cenários de saúde; Identificação de pontos fracos em modelos para melhoria (ex: precisão em tarefas de dados de saúde).

## Integration

**ROT Prompting (Exemplo de Estrutura):** 'Você é um médico especialista em [Especialidade]. Analise o seguinte caso clínico: [Caso Clínico]. Pense passo a passo sobre a evidência clínica e as diretrizes estabelecidas. Qual é a recomendação de tratamento mais consistente com as diretrizes baseadas em evidências? Apresente a recomendação e o nível de evidência.'

**CoT Prompting (Exemplo de Template):** 'Você é um [Especialidade Médica]. Caso Clínico: [Detalhes do Paciente, Sintomas, Resultados de Exames]. Pense passo a passo (Chain-of-Thought): 1. Qual é o diagnóstico mais provável? 2. Quais são as opções de tratamento baseadas em evidências para este diagnóstico? 3. Qual tratamento você recomenda e por quê? Resposta Final: [Recomendação de Tratamento].'

**HealthBench (Uso):** O uso prático envolve a submissão de respostas de um LLM a conversas do HealthBench para pontuação, comparando o desempenho em temas como referências de emergência e comunicação adaptada à experiência. Modelos como o3 e GPT-4.1 demonstraram superar outros modelos.

## URL

https://www.nature.com/articles/s41746-024-01029-4