# Estrutura de Prompt SMART (Seeker, Mission, AI Role, Register, Targeted Question) e Prompt Baseado em Métrica para Readability

## Description

Uma técnica de engenharia de prompt estruturada desenvolvida para melhorar a qualidade, precisão e relevância das respostas de LLMs em ambientes de saúde, particularmente para perguntas de pacientes e geração de materiais educativos. O acrônimo SMART representa os cinco componentes essenciais que guiam o LLM para uma resposta otimizada. Estudo de 2025 (Vaira et al.) demonstrou que o formato SMART melhorou significativamente a precisão, clareza, relevância, completude e utilidade geral das respostas do GPT-4o em cirurgia de cabeça e pescoço, em comparação com prompts não estruturados. Uma técnica de engenharia de prompt que incorpora restrições explícitas de legibilidade (readability) e métricas linguísticas (como nível de leitura de 6ª série, frases curtas e palavras simples) para forçar o LLM a gerar materiais de educação do paciente que atendam aos padrões de alfabetização em saúde. Esta abordagem é mais eficaz do que prompts que apenas solicitam 'linguagem simples'. Estudo de 2025 (Ellison et al.) em cirurgia colorretal demonstrou que o Prompt Baseado em Métrica produziu consistentemente o conteúdo mais legível. O ChatGPT, usando este prompt, gerou materiais com um Nível de Leitura de 5.2, significativamente melhor do que o nível médio de 8.1 dos materiais educacionais existentes.

## Statistics

Estudo de 2025 (Vaira et al.) demonstrou que o formato SMART melhorou significativamente a precisão, clareza, relevância, completude e utilidade geral das respostas do GPT-4o em cirurgia de cabeça e pescoço, em comparação com prompts não estruturados. Estudo de 2025 (Ellison et al.) em cirurgia colorretal demonstrou que o Prompt Baseado em Métrica produziu consistentemente o conteúdo mais legível. O ChatGPT, usando este prompt, gerou materiais com um Nível de Leitura de 5.2, significativamente melhor do que o nível médio de 8.1 dos materiais educacionais existentes.

## Features

Define o papel do usuário (Seeker), o objetivo da consulta (Mission), o papel que a IA deve assumir (AI Role), o tom e o estilo de linguagem (Register) e a pergunta específica (Targeted Question). Garante que a saída seja adaptada ao nível de conhecimento do paciente. Usa métricas quantificáveis (por exemplo, Flesch-Kincaid Grade Level, SMOG) diretamente no prompt. Garante a acessibilidade do conteúdo para pacientes com baixa alfabetização em saúde, um requisito chave para a adesão ao tratamento.

## Use Cases

Geração de respostas precisas e compreensíveis para perguntas de pacientes, criação de materiais de educação do paciente adaptados ao nível de alfabetização em saúde, otimização de chatbots de saúde para interações com o usuário final. Geração de folhetos informativos, instruções de alta e materiais de consentimento que atendam às diretrizes de saúde pública para legibilidade (geralmente nível de 6ª a 8ª série).

## Integration

Exemplo de Prompt para Educação do Paciente:\n\n**Seeker:** Eu sou um paciente buscando informações sobre meu diagnóstico recente de nódulo tireoidiano.\n**Mission:** Quero entender o que é um nódulo tireoidiano, as possíveis causas e as opções de tratamento.\n**AI Role:** Você é um especialista médico em endocrinologia, fornecendo informações de saúde claras e compreensíveis para pacientes.\n**Register:** Use linguagem simples e acessível, adequada para um paciente sem formação médica, e inclua referências a fontes de saúde confiáveis.\n**Targeted Question:** O que devo saber sobre nódulos tireoidianos, suas causas e opções de tratamento? Exemplo de Prompt Baseado em Métrica (adaptado):\n\n**Instrução:** Gere um material educativo para o paciente sobre [Condição Médica].\n**Restrições:** O texto DEVE ter um Nível de Leitura Flesch-Kincaid de 6ª série ou inferior. Use frases curtas (máximo de 15 palavras) e evite palavras polissilábicas. O tom deve ser encorajador e informativo.\n**Pergunta:** Explique [Condição Médica] e o que o paciente pode esperar durante o tratamento.

## URL

https://aao-hnsfjournals.onlinelibrary.wiley.com/doi/full/10.1002/oto2.70075; https://www.sciencedirect.com/science/article/abs/pii/S0039606024010110