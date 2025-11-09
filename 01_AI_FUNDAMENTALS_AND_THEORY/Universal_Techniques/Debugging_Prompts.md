# Debugging Prompts

## Description
**Prompt Debugging** é o processo de análise e refinamento iterativo de prompts para melhorar a qualidade, a confiabilidade e a previsibilidade das saídas geradas por Modelos de Linguagem Grande (LLMs). Ao contrário da depuração de código tradicional, que se concentra em erros de sintaxe e lógica de programação, o Prompt Debugging lida com a **ambiguidade semântica** e a **inconsistência comportamental** do modelo. Envolve diagnosticar por que um prompt está produzindo resultados falhos, enganosos ou subótimos e aplicar técnicas de engenharia de prompt para corrigir a entrada. O objetivo é transformar a natureza não determinística dos LLMs em um comportamento mais controlado e modular, pensando como um linguista (para clareza) e um engenheiro (para estrutura).

## Examples
```
**Exemplo 1: Modularidade e Enquadramento de Papel (Debugging de Código)**
```
[ROLE]
Simule um Engenheiro de Software Sênior especializado em Python e otimização de performance. Sua tarefa é atuar como um depurador (debugger) de prompts.

[INSTRUÇÃO PRINCIPAL]
Analise o seguinte trecho de código Python e a descrição do erro. Sua análise deve ser modular:
1. Identifique a causa raiz do erro.
2. Proponha uma correção concisa.
3. Explique por que o erro ocorreu, focando em performance e segurança.

[CÓDIGO E ERRO]
Código:
def process_data(data_list):
    result = ""
    for item in data_list:
        result += str(item) + ", "
    return result

Erro: A função está lenta para listas grandes e o resultado final tem uma vírgula extra no final.

[RESTRIÇÃO DE SAÍDA]
A saída deve ser um objeto JSON com as chaves: "causa_raiz", "correcao_proposta", "explicacao_tecnica".
```

**Exemplo 2: Cláusula de Segurança (Evitando Alucinações)**
```
[ROLE]
Você é um Historiador Forense, limitado estritamente a fontes revisadas por pares e documentação primária.

[INSTRUÇÃO PRINCIPAL]
Pesquise e descreva a influência exata da Batalha de Alesia (52 a.C.) na política monetária da República Romana tardia.

[CLÁUSULA DE SEGURANÇA]
Se a informação não puder ser verificada com pelo menos duas fontes primárias, você DEVE retornar o seguinte objeto JSON e NUNCA tentar adivinhar ou inferir:
{"status": "inconclusivo", "motivo": "dados insuficientes ou fontes não primárias", "confianca": "0%"}

[RESTRIÇÃO DE SAÍDA]
Retorne a resposta como um parágrafo conciso, ou o objeto JSON de segurança.
```

**Exemplo 3: Correção de Contexto Sobrecarrregado**
```
[PROMPT RUIM]
"Eu sou um gerente de marketing que trabalha para uma startup de SaaS B2B em Nova York. Nossa plataforma usa IA para otimizar campanhas de e-mail. Escreva um tweet de 280 caracteres sobre o novo recurso de segmentação preditiva, mas inclua uma citação de Sun Tzu e mencione a importância do café para o marketing. O tom deve ser profissional, mas divertido. Use hashtags."

[INSTRUÇÃO DE DEBUG]
Reescreva o [PROMPT RUIM] para eliminar o "Contexto Sobrecarrregado" e a "Mistura de Instruções". Mantenha apenas o essencial para o tweet.

[RESTRIÇÃO DE SAÍDA]
Retorne APENAS o prompt reescrito.
```

**Exemplo 4: Simulação vs. Roleplay (Modelagem de Comportamento)**
```
[INSTRUÇÃO PRINCIPAL]
Crie duas respostas para a reclamação de um cliente sobre um produto defeituoso.

[CENÁRIO]
Cliente: "Meu novo fone de ouvido parou de funcionar após 3 dias. Exijo um reembolso imediato!"

[RESPOSTA 1: SIMULATE]
Simule um Agente de Suporte ao Cliente (Nível 3) que segue estritamente o protocolo de reembolso da empresa. A resposta deve ser lógica, neutra e focada no procedimento.

[RESPOSTA 2: ROLEPLAY]
Roleplay um Agente de Suporte ao Cliente (Nível 1) que é empático, mas ainda precisa seguir o procedimento. A resposta deve ser emocionalmente engajadora e usar linguagem de primeira pessoa.

[RESTRIÇÃO DE SAÍDA]
Separe as duas respostas claramente com os títulos "SIMULATE RESPONSE" e "ROLEPLAY RESPONSE".
```

**Exemplo 5: Camadas de Instrução (3-Layer Structure)**
```
[SPINE - Coluna Vertebral]
Simule um Analista de Dados Sênior especializado em tendências de mercado de tecnologia. Sua função principal é filtrar ruído e identificar anomalias.

[PROMPT COMPONENTS - Componentes]
1. Contexto: Analise dados de vendas de smartphones no último trimestre (Q3 2025).
2. Estilo: Use linguagem técnica e estatística.
3. Incerteza: Se a anomalia não tiver significância estatística (p-valor > 0.05), marque-a como "Ruído".

[PROMPT FUNCTIONS - Funções]
1. Compare o crescimento de vendas do 'Modelo X' com a média do mercado.
2. Gere um relatório conciso de 3 pontos sobre as anomalias encontradas.

[RESTRIÇÃO DE SAÍDA]
A saída deve ser um relatório em Markdown.
```

**Exemplo 6: Correção de Sinais Conflitantes**
```
[PROMPT RUIM]
"Escreva um poema épico e detalhado sobre a história da IA, mas ele deve ter exatamente 4 linhas e ser fácil de entender para uma criança de 5 anos."

[INSTRUÇÃO DE DEBUG]
Divida o [PROMPT RUIM] em dois prompts separados para eliminar o conflito.
1. Prompt A: Focado em "Épico e Detalhado".
2. Prompt B: Focado em "4 Linhas e Criança de 5 Anos".

[RESTRIÇÃO DE SAÍDA]
Retorne os dois prompts separados, rotulados como "Prompt A" e "Prompt B".
```

**Exemplo 7: Reforço de Restrições de Saída (Formato de Tabela)**
```
[INSTRUÇÃO PRINCIPAL]
Liste os 5 principais modelos de LLM por número de parâmetros (estimado).

[RESTRIÇÃO DE SAÍDA]
A saída DEVE ser uma tabela Markdown com as colunas: "Modelo", "Desenvolvedor", "Parâmetros (Estimado)". NENHUM texto introdutório ou conclusivo é permitido.
```
```

## Best Practices
**Modularidade e Estrutura em Camadas:** Sempre divida seu prompt em seções claras (e.g., [ROLE], [INSTRUÇÃO PRINCIPAL], [RESTRIÇÃO DE SAÍDA]). Para prompts complexos, use a estrutura de 3 camadas: **Spine** (Regras e Papel), **Components** (Contexto e Estilo) e **Functions** (Ações). **Enquadramento de Papel Preciso:** Use "Simulate" para tarefas lógicas e neutras (pensamento de sistema) e "Roleplay" para tarefas que exigem personalidade e emoção (comportamento humano). **Cláusula de Segurança (Fail-Safe):** Inclua uma instrução explícita para o modelo admitir incerteza (e.g., retornar JSON `{"status": "inconclusivo"}`) em vez de alucinar quando os dados forem insuficientes. **Restrições de Saída Explícitas:** Sempre especifique o formato de saída desejado (JSON, Markdown, lista numerada) para garantir consistência e facilitar o processamento posterior.

## Use Cases
**Depuração de Código:** Analisar e corrigir código gerado por IA que é logicamente incorreto, ineficiente ou inseguro. **Otimização de Fluxos de Trabalho:** Enviar dados estruturados (como o estado DOM de uma UI em JSON) para o LLM para depurar formulários e fluxos de usuário. **Auditoria e Logs de IA:** Usar protocolos de compressão simbólica (como glifos) para codificar saídas de IA, pontuações de confiança ou resultados de diagnóstico para logs e auditorias. **Geração de Conteúdo Consistente:** Garantir que o conteúdo gerado siga um tom, estilo e estrutura predefinidos de forma consistente. **Modelagem de Comportamento:** Usar "Simulate" para modelar o comportamento de sistemas e "Roleplay" para modelar o comportamento humano em cenários de diálogo.

## Pitfalls
**Contexto Sobrecarrregado (Overloaded Context):** Incluir muita informação de fundo ou tarefa de uma só vez, diluindo o foco do modelo. **Falta de Enquadramento de Papel:** Não atribuir um papel ou persona específico, resultando em respostas genéricas e brandas. **Camadas de Instrução Mistas:** Empilhar múltiplas instruções (tom, formato, conteúdo) na mesma frase, fazendo com que o modelo priorize a instrução errada. **Objetivos Ambíguos:** Não declarar claramente o que constitui uma resposta de sucesso, levando a saídas errantes. **Sinais Conflitantes:** Pedir criatividade e estrutura estrita simultaneamente sem priorização clara. **Ausência de Cláusula de Segurança:** Não dar permissão ao modelo para dizer "não sei" ou "dados insuficientes", o que leva a alucinações. **Falta de Pensamento Modular:** Escrever prompts como "paredes de texto" difíceis de manter, depurar e reutilizar.

## URL
[https://www.codestringers.com/insights/prompt-debugging/](https://www.codestringers.com/insights/prompt-debugging/)
