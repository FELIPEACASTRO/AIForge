# GPT-4

## Description

O GPT-4 é um modelo multimodal grande (aceitando entradas de imagem e texto, emitindo saídas de texto) que representa o mais recente marco no esforço da OpenAI para escalar o aprendizado profundo. Sua proposta de valor única reside em sua confiabilidade, criatividade e capacidade de lidar com instruções muito mais matizadas do que o GPT-3.5.

## Statistics

Desempenho em Exames: Passou no exame simulado da ordem dos advogados com uma pontuação em torno dos 10% melhores (GPT-3.5 estava nos 10% inferiores). MMLU (Multilingual Massive Multitask Language Understanding): Supera o desempenho em inglês do GPT-3.5 e de outros LLMs em 24 das 26 línguas testadas. Tamanho do Modelo (Estimativa não oficial): Estima-se que tenha 1.5 trilhão de parâmetros, em comparação com 175 bilhões do GPT-3.5.

## Features

Multimodalidade: Aceita entradas de texto e imagem (Visão). Confiabilidade e Steerability (Direcionabilidade): Melhorias significativas em factualidade e capacidade de direcionar o comportamento da IA através de mensagens de \"sistema\". Criatividade: Mais criativo e capaz de lidar com instruções complexas e nuanceadas. Contexto: Capacidade de processar um contexto maior (até 32k tokens em algumas versões).

## Use Cases

Resolução de problemas complexos em domínios profissionais (ex: direito, medicina). Criação de conteúdo multimodal (ex: descrição de imagens). Tutoria Socrática e agentes de IA personalizados (via Steerability). Programação e otimização de código.

## Integration

Disponível via API e ChatGPT Plus. Suporta mensagens de sistema para personalizar o estilo e a tarefa da IA. Exemplo de Integração (Python - via `openai` SDK): ```python\nfrom openai import OpenAI\nclient = OpenAI()\nresponse = client.chat.completions.create(\n    model=\"gpt-4\",\n    messages=[\n        {\"role\": \"system\", \"content\": \"You are a helpful assistant.\"},\n        {\"role\": \"user\", \"content\": \"Explain the concept of instruction tuning in one sentence.\"}\n    ]\n)\nprint(response.choices[0].message.content)\n```

## URL

https://openai.com/index/gpt-4-research/