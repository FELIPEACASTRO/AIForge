# A/B Test Analysis Prompts

## Description
Prompts de Análise de Teste A/B são instruções estruturadas fornecidas a Modelos de Linguagem Grande (LLMs) para interpretar, narrar e gerar *insights* acionáveis a partir de dados brutos de experimentos A/B. Em vez de substituir a análise estatística tradicional, o LLM atua como uma camada narrativa e de governança, traduzindo métricas complexas (como *lift*, intervalos de confiança e *p*-valores) em orientações claras para a tomada de decisão executiva, de produto e de crescimento. O objetivo principal é acelerar o ciclo de *insights*, melhorar o alinhamento interfuncional e garantir que as conclusões narrativas reflitam a rigor estatístico subjacente. É crucial que o prompt exija que o LLM se atenha aos resultados estatísticos fornecidos, evitando alucinações ou superinterpretação. Essa técnica é fundamental para equipes de Experimentação e Otimização de Taxa de Conversão (CRO).

## Examples
```
**1. Análise Estatística e Narrativa (Completo)**
"Atue como um Cientista de Dados Sênior. Analise os seguintes resultados de Teste A/B. Hipótese: A Versão B aumentará a Taxa de Conversão (CVR) em 5%. Dados: Versão A (Controle): 50.000 usuários, 1.500 conversões (CVR 3.0%). Versão B (Variante): 50.000 usuários, 1.650 conversões (CVR 3.3%). Significância Estatística: 95% de confiança, *p*-valor de 0.02.
1. Determine se a Versão B é a vencedora com base na significância.
2. Calcule o *lift* percentual exato.
3. Forneça uma narrativa de 3 parágrafos para o C-Level, explicando o resultado, o impacto no negócio e as 3 próximas ações recomendadas."

**2. Investigação de Segmento (Drill-Down)**
"O Teste A/B geral falhou em atingir a significância (p=0.15). No entanto, suspeitamos de um efeito no segmento de 'Novos Usuários'. Forneça os dados de ambos os segmentos (Novos Usuários: A=100 conversões/5k visitas, B=150 conversões/5k visitas; Usuários Recorrentes: A=1.400 conversões/45k visitas, B=1.500 conversões/45k visitas).
1. Calcule o *lift* e a significância (assuma um teste Z) para o segmento 'Novos Usuários'.
2. Explique a discrepância entre o resultado geral e o resultado do segmento.
3. Sugira uma estratégia de segmentação para o lançamento da Versão B."

**3. Relatório de Risco e Conflito de Métricas**
"O Teste A/B da nova página de *checkout* (Versão B) mostrou um aumento de 10% na métrica primária (Taxa de Conclusão de Compra) com 99% de confiança. No entanto, a métrica secundária (Taxa de Cancelamento de Assinatura) também aumentou em 2%.
1. Redija um alerta de risco para a equipe de Produto.
2. Sugira 3 hipóteses para o aumento da Taxa de Cancelamento.
3. Proponha um teste de acompanhamento para mitigar o risco secundário."

**4. Otimização de Prompt (Meta-Análise)**
"Analise as 5 versões de prompts que usamos para gerar descrições de produtos. A métrica de sucesso foi a Taxa de Cliques (CTR) no link 'Comprar Agora'.
- Prompt 1 (Foco em Benefício): CTR 4.5%
- Prompt 2 (Foco em Urgência): CTR 5.1%
- Prompt 3 (Foco em Recurso): CTR 3.9%
Com base nesses resultados (assuma significância), qual é o princípio de *copywriting* mais eficaz? Crie um novo 'Prompt Mestre' que combine os melhores elementos do Prompt 2 e do Prompt 1."

**5. Interpretação de Resultados Bayesianos**
"Atue como um estatístico. Recebi os resultados de um teste A/B Bayesiano. A 'Probabilidade de B ser Melhor' é de 98.5%. O 'Uplift Esperado' é de 6.2%.
1. Explique o que esses dois números significam para um gerente de marketing não-técnico.
2. Qual é o risco de implementar a Versão B?
3. Qual seria o tamanho de amostra necessário para atingir 99.9% de Probabilidade de B ser Melhor, mantendo o *uplift* atual?"
```

## Best Practices
**1. Forneça o Contexto Completo:** Inclua a hipótese inicial, a unidade experimental, as métricas primárias e secundárias, o método estatístico utilizado (e.g., frequentista ou Bayesiano) e o limiar de significância pré-especificado. **2. Estrutura de Dados Clara:** Apresente os dados do teste (impressões, cliques, conversões, receita, etc.) em um formato estruturado (tabela Markdown, CSV ou JSON) para evitar erros de interpretação. **3. Separe o Fato da Inferência:** Peça ao LLM para distinguir claramente entre as conclusões baseadas em significância estatística e as inferências qualitativas (o "porquê" do resultado). **4. Peça por Próximos Passos:** Não se contente apenas com a conclusão. Solicite recomendações acionáveis para a próxima rodada de testes ou para a implementação da variante vencedora. **5. Use a Persona de Especialista:** Comece o prompt instruindo o LLM a agir como um "Cientista de Dados Sênior" ou "Especialista em Otimização de Taxa de Conversão (CRO)".

## Use Cases
nan

## Pitfalls
**1. Confundir Narrativa com Rigor Estatístico:** Confiar cegamente na interpretação do LLM sem verificar os dados estatísticos subjacentes. O LLM é um narrador, não um motor estatístico. **2. Falta de Contexto:** Não fornecer a hipótese, as métricas e o método estatístico. Isso leva a análises genéricas e potencialmente incorretas. **3. Superinterpretação de Resultados Não-Significativos:** Pedir ao LLM para encontrar *insights* profundos em um teste que falhou em atingir a significância, o que pode levar a conclusões falsas (*false positives*). **4. Ignorar Métricas Secundárias:** Focar apenas na métrica primária e não pedir ao LLM para analisar o impacto em métricas secundárias ou de proteção (e.g., receita por usuário, taxa de rejeição). **5. Alucinação de Dados:** O LLM pode "inventar" dados ou estatísticas se o prompt for muito vago ou se o contexto for insuficiente. Sempre forneça os dados brutos ou sumarizados de forma explícita.

## URL
[https://www.gurustartups.com/reports/using-chatgpt-for-a-b-test-result-analysis](https://www.gurustartups.com/reports/using-chatgpt-for-a-b-test-result-analysis)
