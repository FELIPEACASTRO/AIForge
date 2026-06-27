# Anomaly Detection Prompts

## Description
Prompts de Detecção de Anomalias utilizam Grandes Modelos de Linguagem (LLMs) para identificar padrões, eventos ou pontos de dados que se desviam significativamente do comportamento normal em um conjunto de dados. A técnica se baseia na capacidade do LLM de raciocínio contextual e conhecimento pré-treinado para distinguir o que é "normal" do que é "anormal". O prompt é crucial, pois deve converter dados não textuais (como logs, séries temporais ou dados tabulares) em um formato que o LLM possa processar, além de fornecer contexto, exemplos (few-shot) e restrições de raciocínio (Chain-of-Thought) para otimizar a precisão da detecção. As abordagens incluem a detecção direta, a geração de dados anômalos sintéticos para aumento de dados e a geração de explicações em linguagem natural para as anomalias detectadas.

## Examples
```
**1. Detecção de Anomalia em Logs de Servidor (CoT):**
```
**Instrução:** Você é um analista de segurança. Analise a sequência de logs abaixo e determine se há uma anomalia.
**Logs:**
[2025-11-08 10:00:01] INFO User 'admin' logged in from IP 192.168.1.10
[2025-11-08 10:00:05] DEBUG Database query successful
[2025-11-08 10:00:08] INFO User 'admin' accessed /api/v1/data
[2025-11-08 10:00:10] ERROR Failed to connect to external service: Timeout
[2025-11-08 10:00:11] INFO User 'admin' logged in from IP 203.0.113.55 (Novo login em 10 segundos)
**Passos de Raciocínio (CoT):**
1. O primeiro login do 'admin' ocorreu às 10:00:01.
2. O segundo login do 'admin' ocorreu às 10:00:11, apenas 10 segundos depois, de um IP completamente diferente (203.0.113.55).
3. Logins rápidos de IPs distintos para o mesmo usuário são um padrão incomum e sugerem sequestro de sessão ou acesso simultâneo não autorizado.
**Anomalia Detectada:** Sim/Não
**Explicação:** [Gere a explicação baseada no raciocínio]
```

**2. Detecção de Anomalia em Série Temporal (Dados Tabulares Convertidos):**
```
**Instrução:** Analise os dados de uso de CPU (em %) nas últimas 10 horas. O comportamento normal é entre 20% e 70%. Identifique o ponto de dados mais anômalo.
**Dados:**
Hora: 1, CPU: 35%
Hora: 2, CPU: 42%
...
Hora: 8, CPU: 68%
Hora: 9, CPU: 95%
Hora: 10, CPU: 55%
**Anomalia:** [Ponto de dados anômalo]
**Justificativa:** [Explique por que o ponto é anômalo em relação ao limite normal e ao contexto]
```

**3. Geração de Dados Anômalos Sintéticos (Data Augmentation):**
```
**Instrução:** Você é um gerador de logs de segurança. Crie 3 exemplos de logs de sistema que representem uma "Tentativa de Força Bruta" em um servidor web, mantendo o formato de log padrão.
**Formato Padrão:** [TIMESTAMP] [LEVEL] [SOURCE] [MESSAGE]
**Exemplos Sintéticos:** [Gere 3 logs que simulem a anomalia]
```

**4. Detecção de Anomalia em Texto (Revisão de Documentos):**
```
**Instrução:** Analise o parágrafo a seguir de um relatório financeiro. O tom esperado é formal e otimista. Identifique e justifique qualquer frase que apresente uma anomalia de tom ou conteúdo.
**Parágrafo:** "O crescimento do trimestre superou as expectativas, impulsionado por inovações estratégicas. No entanto, a equipe de liderança está secretamente planejando vender a empresa a um concorrente a um preço abaixo do mercado, o que é um desastre iminente."
**Anomalia:** [Frase anômala]
**Tipo de Anomalia:** [Ex: Factual, Tonal]
**Justificativa:** [Explique a anomalia]
```

**5. Detecção de Anomalia em Dados Tabulares (Zero-Shot):**
```
**Instrução:** Dada a tabela de transações de clientes, identifique o 'ID da Transação' que representa uma anomalia de valor.
**Tabela (CSV):**
ID da Transação, Cliente, Valor, Local
T001, João, 50.00, SP
T002, Maria, 120.50, RJ
T003, Pedro, 987500.00, MG
T004, Ana, 75.20, SP
**Anomalia:** [ID da Transação]
**Motivo:** [Explique o desvio do padrão]
```
```

## Best Practices
**1. Estruturação Detalhada do Prompt:** Inclua a descrição da tarefa, a definição de "normal" e "anomalia", o formato de saída desejado (ex: JSON), e o contexto dos dados (ex: tipo de log, frequência da série temporal).
**2. Few-Shot Learning:** Forneça exemplos rotulados de dados normais e anômalos (few-shot examples) para guiar o LLM, especialmente em domínios de nicho onde o conhecimento pré-treinado pode ser insuficiente.
**3. Chain-of-Thought (CoT):** Solicite ao LLM que justifique seu raciocínio antes de fornecer o veredito final. Isso aumenta a interpretabilidade e a precisão, forçando o modelo a seguir um processo lógico.
**4. Conversão de Dados:** Para dados não textuais (séries temporais, tabulares), desenvolva um pipeline robusto para converter os dados em um formato textual compreensível e sem perda de informação (ex: tokenização, descrição estatística).
**5. Validação Humana:** Use o LLM para gerar explicações em linguagem natural para as anomalias, facilitando a validação e a ação por analistas humanos.

## Use Cases
**1. Monitoramento de Sistemas e Segurança (Logs):** Análise de logs de servidor, rede ou aplicativos para identificar eventos incomuns, como tentativas de invasão, falhas de sistema ou picos de erro.
**2. Detecção de Fraudes Financeiras:** Análise de transações bancárias ou de cartão de crédito para identificar padrões de gastos que se desviem do perfil normal do usuário.
**3. Manutenção Preditiva (Séries Temporais):** Monitoramento de dados de sensores de máquinas (temperatura, vibração, pressão) para detectar desvios que sinalizem falha iminente de equipamento.
**4. Controle de Qualidade de Dados:** Identificação de valores discrepantes, erros de entrada ou registros inconsistentes em grandes bases de dados tabulares.
**5. Análise de Saúde (Registros Médicos):** Identificação de padrões incomuns em registros eletrônicos de saúde que possam indicar uma condição rara ou um erro de diagnóstico.

## Pitfalls
**1. Dependência Excessiva do Conhecimento Pré-treinado:** LLMs podem falhar em detectar anomalias em domínios de nicho ou em dados com padrões muito específicos que não estavam presentes no seu treinamento.
**2. Perda de Informação na Conversão:** A conversão de dados não textuais (séries temporais, imagens) para texto pode levar à perda de detalhes cruciais, resultando em falsos negativos ou positivos.
**3. Custo e Latência:** A detecção de anomalias em tempo real pode ser inviável devido ao alto custo computacional e à latência da inferência do LLM, especialmente para grandes volumes de dados.
**4. Alucinações:** O LLM pode "alucinar" anomalias ou explicações, gerando resultados que parecem plausíveis, mas são factualmente incorretos.
**5. Dificuldade em Definir "Normal":** A definição vaga ou ambígua de "comportamento normal" no prompt pode levar a resultados inconsistentes. O prompt deve ser o mais específico possível.

## URL
[https://towardsdatascience.com/boosting-your-anomaly-detection-with-llms/](https://towardsdatascience.com/boosting-your-anomaly-detection-with-llms/)
