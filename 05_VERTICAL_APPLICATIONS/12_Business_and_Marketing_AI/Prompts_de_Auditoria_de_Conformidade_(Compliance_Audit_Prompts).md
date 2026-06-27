# Prompts de Auditoria de Conformidade (Compliance Audit Prompts)

## Description
A prática de engenharia de prompt focada na criação de instruções detalhadas e contextuais para modelos de linguagem (LLMs) com o objetivo de auxiliar em tarefas de Auditoria Interna, Risco e Conformidade (GRC). Isso inclui a análise de documentos regulatórios, a identificação de lacunas de controle, a sumarização de relatórios de risco e a geração de trilhas de auditoria. O objetivo é aumentar a eficiência, a precisão e a abrangência dos processos de auditoria, transformando dados não estruturados em *insights* acionáveis e alinhados a *frameworks* como COSO, NIST e ISO 27001. Esses prompts atuam como uma ponte entre a abundância de dados e a necessidade de *insights* estratégicos e documentáveis para comitês de auditoria e reguladores.

## Examples
```
**1. Análise de Lacunas de Controle (Gap Analysis)**
```
Aja como um auditor de segurança da informação. Analise o "Relatório de Teste de Penetração de 2024" e o "Padrão de Criptografia ISO 27001". Identifique todas as lacunas de controle relacionadas à criptografia de dados em trânsito e em repouso. Liste as lacunas em uma tabela com três colunas: "Lacuna de Controle", "Risco Associado" e "Referência ISO 27001".
```

**2. Sumarização de Risco para Executivos**
```
Com base na "Matriz de Risco Empresarial 2025" e nos "Relatórios de Incidente do Último Trimestre", gere um memorando de resumo de risco de uma página para a liderança executiva. O memorando deve destacar os 3 principais riscos operacionais por probabilidade e impacto, quantificar o impacto potencial em termos financeiros (se possível) e sugerir duas estratégias de mitigação de curto prazo para cada risco.
```

**3. Criação de Trilha de Auditoria (Audit Trail)**
```
Você é um especialista em documentação de GRC. Liste os passos tomados nesta revisão de conformidade (incluindo documentos de origem referenciados, filtros aplicados e datas de interação) em um formato adequado para documentação de evidência de auditoria. O resultado deve ser um arquivo de log estruturado com carimbos de data/hora e links para os documentos de origem.
```

**4. Avaliação de Conformidade Regulatória (GDPR/LGPD)**
```
Analise a "Política de Privacidade do Cliente" e o "Fluxo de Processamento de Dados de Usuários". Crie uma lista de verificação (checklist) para avaliar a conformidade com os requisitos de "Direitos do Titular dos Dados" (e.g., direito ao esquecimento, portabilidade) do GDPR e da LGPD. Para cada item, forneça uma pontuação de 1 a 5 (1=Não Conforme, 5=Totalmente Conforme) e uma breve justificativa.
```

**5. Identificação de Anomalias em Transações**
```
Aja como um analista de fraude. Examine o conjunto de dados de "Transações Financeiras do Mês Passado" (anexado). Identifique e liste todas as transações que excedam 3 desvios-padrão da média de valor ou que envolvam um fornecedor não aprovado. Formate a saída como um relatório de exceções, incluindo ID da Transação, Valor, Fornecedor e Justificativa da Sinalização.
```

**6. Revisão de Contrato de Terceiros**
```
Analise o "Contrato de Serviço com o Fornecedor X". Extraia e liste todas as cláusulas relacionadas a requisitos de segurança de dados, direito de auditoria e responsabilidade por violação de dados. Compare essas cláusulas com o nosso "Padrão Mínimo de Segurança para Terceiros" e aponte quaisquer discrepâncias ou áreas de alto risco.
```

## Best Practices
**Estrutura I-I-O (Instrução, Informação, Objetivo):** Use uma estrutura clara: 1. **Instrução** (O que fazer - e.g., "Analise"), 2. **Informação** (Onde aplicar - e.g., "O documento X"), 3. **Objetivo** (O resultado esperado - e.g., "Em formato de tabela, destacando riscos altos"). **Definição de Papel (Role-Playing):** Comece o prompt com "Aja como um auditor interno sênior especializado em [área de conformidade]". Isso melhora a qualidade e o tom da resposta. **Contextualização e Restrições:** Forneça o máximo de contexto possível (e.g., "Considerando a LGPD e o GDPR") e defina restrições de formato, extensão e *frameworks* (e.g., "Use o framework COSO para a avaliação"). **Validação Humana (Human-in-the-Loop):** Nunca confie cegamente na saída do LLM. A saída deve ser um rascunho ou uma análise preliminar que requer revisão e validação final por um auditor humano. **Clareza sobre Dados de Entrada:** Especifique quais documentos ou dados o LLM deve analisar (e.g., "Anexe o relatório SOC 2 e a matriz de risco interna").

## Use Cases
**Auditoria Interna:** Acelerar testes de controle, identificar exceções em grandes volumes de dados e automatizar a criação de trilhas de auditoria. **Gestão de Risco (Risk Management):** Rascunhar resumos de risco alinhados a *frameworks* (COSO, NIST), analisar registros de risco e quantificar o impacto potencial de vulnerabilidades. **Conformidade Regulatória (Compliance):** Realizar análises de lacunas (*gap analysis*) entre políticas internas e regulamentações externas (e.g., GDPR, LGPD, SOX). **Revisão de Documentos:** Sumarizar relatórios longos (e.g., SOC 2, relatórios de terceiros), extrair cláusulas contratuais específicas e comparar documentos regulatórios. **Governança de IA:** Auditar o uso de outras ferramentas de IA dentro da organização para garantir que estejam em conformidade com princípios éticos e legais (e.g., transparência, justiça, explicabilidade).

## Pitfalls
**Confiança Excessiva (Over-Reliance):** Tratar a saída do LLM como verdade absoluta, ignorando a necessidade de julgamento e validação humana. **Alucinações e Imprecisão:** O LLM pode "alucinar" referências regulatórias ou criar lacunas de controle inexistentes. A verificação cruzada é essencial. **Vazamento de Dados Sensíveis:** Inserir dados confidenciais ou PII (Informações de Identificação Pessoal) no prompt. É crucial anonimizar ou usar dados simulados/mascarados. **Prompts Vagos:** Usar prompts como "Me ajude com a auditoria". A falta de contexto, função e formato de saída leva a resultados inúteis. **Viés Algorítmico:** Se o LLM for usado para avaliar viés (e.g., em recrutamento), ele pode perpetuar ou amplificar o viés existente se não for instruído a usar métricas de justiça específicas.

## URL
[https://empoweredsystems.com/blog/prompt-engineering-for-internal-auditors-a-new-skill-for-a-new-era/](https://empoweredsystems.com/blog/prompt-engineering-for-internal-auditors-a-new-skill-for-a-new-era/)
