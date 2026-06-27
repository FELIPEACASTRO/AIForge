# Computer Vision Prompts (Prompts de Visão Computacional)

## Description
A Engenharia de Prompts para Visão Computacional (especialmente Modelos de Linguagem de Visão - VLMs) é a prática de ajustar a entrada de texto (o "prompt") fornecida a um modelo multimodal (que aceita imagens, vídeos e texto) para guiar sua saída e melhorar a qualidade e precisão da resposta. Essa técnica é uma alternativa leve e eficiente ao ajuste fino (fine-tuning) completo do modelo, permitindo que o usuário direcione o VLM para tarefas específicas como classificação, resposta a perguntas visuais (VQA), detecção de objetos e análise temporal de vídeos. O prompt atua como um contexto que "desbloqueia" as capacidades do modelo para um determinado domínio ou tarefa.

## Examples
```
1. **Localização Temporal (Vídeo):** `Prompt: Quando o trabalhador deixou cair a caixa?` (Espera-se um intervalo de tempo e uma descrição).
2. **Saída Estruturada (VQA):** `Prompt: Há um caminhão de bombeiros? Há um incêndio? Há bombeiros? Apresente a resposta para cada pergunta no formato JSON.`
3. **Estimação/VQA (Imagem Única):** `Prompt: Estime o nível de estoque da mesa de lanches em uma escala de 0 a 100.`
4. **Detecção de Objetos com Coordenadas (Imagem Única):** `Prompt: Identifique todos os objetos do tipo 'caminhão' na imagem e forneça suas coordenadas de caixa delimitadora (bounding box) no formato [x_min, y_min, x_max, y_max].`
5. **Extração de OCR e Estruturação (Análise de Documentos):** `Prompt: Você é um processador de faturas. Extraia o 'Número da Fatura', 'Data de Vencimento' e 'Valor Total' da imagem do documento. Apresente a saída em um objeto JSON com as chaves 'invoice_number', 'due_date' e 'total_amount'.`
6. **Comparação e Controle de Qualidade (Múltiplas Imagens):** `Prompt: Compare a 'Imagem A' (Produto de Referência) com a 'Imagem B' (Produto Inspecionado). Descreva as diferenças e classifique o Produto B como 'Aprovado' ou 'Rejeitado' com base na presença de defeitos visíveis.`
7. **Raciocínio Visual e Resolução de Problemas (Imagem Única):** `Prompt: Descreva o que está acontecendo na imagem e, em seguida, sugira a próxima ação lógica para resolver o problema apresentado. Você é um agente de manutenção.`
```

## Best Practices
**Clareza e Especificidade:** Defina claramente o objetivo, o formato de saída e o foco visual. **Função (Role):** Atribua uma função ao modelo (ex: "Você é um inspetor de segurança..."). **Saída Estruturada:** Solicite a saída em formatos estruturados (como JSON) para facilitar o processamento por tarefas a jusante (downstream tasks). **Aprendizagem em Contexto (In-Context Learning):** Forneça exemplos de pares (imagem/vídeo + prompt + resposta) para guiar o modelo. **Prompt Tuning (VP/VPT):** Para modelos que suportam, usar o Prompt Tuning (ajuste de "soft prompts" com pesos congelados) é mais eficiente e leve que o Fine-Tuning completo. **Contexto Temporal:** Para vídeos, use modelos que suportam compreensão sequencial para capturar a progressão de ações.

## Use Cases
**Compreensão de Imagem Única:** Classificação, legendagem, resposta a perguntas visuais (VQA), detecção de eventos básicos. **Compreensão de Múltiplas Imagens:** Comparar, contrastar e aprender a partir de múltiplas entradas visuais (ex: detecção de nível de estoque ao longo do tempo). **Compreensão de Vídeo (Localização Temporal):** Entender ações e tendências ao longo do tempo, identificando quando e onde eventos críticos ocorrem. **Detecção de OCR:** Extrair texto de imagens e documentos. **Análise de Documentos:** Processar formulários ou documentos digitalizados. **Monitoramento de Segurança:** Detecção de anomalias ou violações de regras em vídeos de vigilância. **Controle de Qualidade:** Comparar imagens de produtos para identificar defeitos.

## Pitfalls
**Prompts Vagos:** Instruções pouco específicas ou que não fornecem foco visual suficiente. **Contexto Sobrecarrregado:** Incluir muita informação de fundo ou múltiplas tarefas em um único prompt, diluindo o foco (token dilution). **Contradição:** Elementos do prompt que se anulam ou confundem o modelo. **Ignorar Saída Estruturada:** Não solicitar formatos estruturados (JSON, XML) quando a saída será usada por outros sistemas. **Falta de Contexto Específico do Domínio:** Em casos de uso específicos (ex: varejo, medicina), a falta de contexto relevante pode levar a respostas imprecisas.

## URL
[https://developer.nvidia.com/blog/vision-language-model-prompt-engineering-guide-for-image-and-video-understanding/](https://developer.nvidia.com/blog/vision-language-model-prompt-engineering-guide-for-image-and-video-understanding/)
