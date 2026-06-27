# M&A Due Diligence Prompts

## Description
A técnica de **Prompts de Due Diligence de Fusões e Aquisições (M&A)** refere-se ao uso de modelos de linguagem avançados (LLMs) e Inteligência Artificial (IA) para acelerar, aprimorar e aprofundar o processo de investigação de uma empresa-alvo. Em um mercado competitivo, onde a velocidade e a precisão são cruciais, a IA atua como um "detector de metais" em um "palheiro" de dados, transformando a DD de um processo manual e demorado para um processo ágil e orientado por dados.

O principal benefício é a **aceleração da velocidade do negócio** e o **aumento da precisão**. A IA pode realizar a revisão de milhões de documentos, contratos e registros financeiros em horas, não semanas, identificando padrões, anomalias e cláusulas de risco (como "mudança de controle" ou "limites de responsabilidade") que poderiam ser negligenciados em uma revisão manual. Os prompts são projetados para extrair insights acionáveis, testar a tese de investimento e prever desafios de integração, permitindo que as equipes de M&A se concentrem na interpretação estratégica dos dados em vez da mera coleta.

## Examples
```
**1. Análise de Risco de Contrato (Legal):**
```
Aja como um advogado de M&A. Analise o seguinte conjunto de contratos de clientes [Anexar Contratos]. Extraia e liste todas as cláusulas de "Mudança de Controle" (Change of Control) e "Rescisão por Conveniência" (Termination for Convenience). Para cada uma, avalie o risco de perda de receita pós-aquisição em uma escala de 1 a 5 e forneça uma recomendação de mitigação.
```

**2. Mapeamento de Risco de Mercado (Estratégico):**
```
Aja como um estrategista de compra. Construa um mapa de calor dos alvos de aquisição no setor de {Indústria}, classificados pela probabilidade de sucesso da integração. Use sinais de risco públicos (picos de dívida, rotatividade de executivos C-suite, manchetes de litígios). Entregue os 10 principais em uma tabela com justificativas curtas.
```

**3. Teste de Pressão de Valuation (Financeiro):**
```
Interrogue a narrativa de valuation da {Empresa Alvo}. Construa uma matriz de cenários mostrando como choques macroeconômicos (ex: aumento da taxa de juros em 200 bps) e riscos da cadeia de suprimentos poderiam reduzir o valuation em 30%. Apresente o resultado em uma tabela de 3x3.
```

**4. Identificação de Poder de Precificação Oculto (Comercial):**
```
Analise as últimas 12 transcrições de chamadas de lucros da {Empresa Alvo}. Extraia pistas negligenciadas sobre a segmentação de clientes e o poder de precificação. Recomende duas experiências rápidas de precificação que valham mais de 8% de aumento de margem após o fechamento do negócio.
```

**5. Lista de Verificação de Bandeiras Vermelhas (Operacional/RH):**
```
Liste as 12 bandeiras vermelhas que prejudicam o moral pós-fusão em serviços habilitados por tecnologia. Classifique cada uma como Ignorar / Mitigar / Fator de Ruptura (Deal-Breaker).
```

**6. Blueprint de Gamificação de Earn-Out (Financeiro/RH):**
```
Projete um earn-out que alinhe a psicologia do vendedor com uma meta de receita de dois anos. Descreva a cesta de métricas, as salvaguardas e um "kicker" de "dragão duplo" se eles atingirem {X} em {Y} meses.
```

**7. Análise de Choque Cultural (RH/Integração):**
```
Mescle a {Adquirente} (orientada a dados, remota) com a {Alvo} (liderada pelo fundador, presencial). Identifique os três principais choques culturais e crie um plano de 30 dias para neutralizá-los.
```

**8. Back-Casting de Cronograma de Integração (Tecnologia/Operacional):**
```
Precisamos integrar o {Sistema} da {Empresa Alvo} ao nosso em 90 dias. Faça o back-casting do cronograma de integração, identificando as 5 dependências mais críticas e o ponto único de falha.
```
```

## Best Practices
**1. Abordagem Híbrida (Humano + IA):** A experiência humana é **essencial** para interpretar descobertas complexas, fornecer conhecimento local e contextual, e exercer o julgamento crítico final. A IA deve ser uma ferramenta de triagem e aceleração, não um substituto para a decisão final.
**2. Foco na Interpretação:** Use a IA para agilizar a triagem de documentos e a identificação de anomalias. O tempo economizado deve ser reinvestido na **interpretação** aprofundada dos "sinais" encontrados, garantindo que sejam relevantes e não falsos positivos.
**3. Criação de Hipóteses Robustas:** Utilize Large Language Models (LLMs) no início do processo para criar **hipóteses robustas** sobre o alvo (Target) e o mercado. Isso direciona a Due Diligence (DD) para as áreas de maior risco ou potencial de valor.
**4. Especificidade do Prompt:** Seja o mais específico possível. Defina o **papel** da IA (ex: "Estrategista de compra", "Advogado de IP"), o **formato de saída** (ex: "Tabela", "Resumo de 5 pontos") e o **contexto** (ex: "Setor de SaaS", "Transcrições de chamadas de lucros").
**5. Teste de Pressão Contínuo:** Use prompts para testar a narrativa de valor do alvo (Target) e os planos de integração. Crie cenários adversos (ex: "Choques macroeconômicos", "Falha na cadeia de suprimentos") para quantificar a resiliência do negócio.

## Use Cases
**1. Revisão Acelerada de Contratos (Contract Diligence):** Análise em massa de contratos de clientes, fornecedores e funcionários para identificar cláusulas de risco (ex: Change of Control, exclusividade, litígios pendentes).
**2. Análise Financeira e Contábil:** Triagem rápida de demonstrações financeiras, notas de rodapé e relatórios de auditoria para identificar anomalias, inconsistências ou áreas de agressividade contábil.
**3. Preparação de Listas de Solicitação (Request Lists):** Geração de listas de documentos de Due Diligence (DD) altamente personalizadas e abrangentes, adaptadas ao setor e ao tipo de transação.
**4. Avaliação de Risco de Conformidade Legal e Regulatória:** Escaneamento de documentos para garantir a conformidade com regulamentações específicas do setor (ex: GDPR, HIPAA, regulamentações ambientais) e sinalização de lacunas.
**5. Avaliação de Sinergias e Choque Cultural:** Uso de prompts para analisar documentos internos (ex: manuais de funcionários, comunicações internas) para prever desafios de integração cultural e quantificar o potencial de sinergia de custos e receitas.
**6. Detecção de Fraude e Vulnerabilidades:** Identificação de padrões incomuns em transações ou comunicações que possam indicar fraude ou vulnerabilidades de segurança cibernética.

## Pitfalls
**1. Dependência Excessiva e Falta de Nuance:** Confiar cegamente nas conclusões da IA sem supervisão humana. A IA pode ter dificuldade em interpretar relacionamentos complexos, acessar dados locais críticos ou capturar sutilezas em condições de mercado e cultura corporativa.
**2. Viés e Qualidade dos Dados:** A saída da IA é tão boa quanto a entrada. Se os documentos de Due Diligence (DD) estiverem incompletos, desorganizados ou contiverem vieses, a IA amplificará esses problemas. A limpeza e a curadoria dos dados de entrada são cruciais.
**3. Alucinações e Falsos Positivos:** LLMs podem "alucinar" fatos ou criar conexões lógicas que não existem nos documentos de origem. É obrigatório que os profissionais humanos **verifiquem a fonte** de cada risco ou anomalia sinalizada pela IA.
**4. Risco de Confidencialidade:** Usar modelos de IA públicos ou não seguros para analisar dados confidenciais de M&A. É essencial usar soluções de IA empresariais e privadas que garantam a segurança e a não retenção dos dados.
**5. Prompts Vagos:** Usar prompts genéricos como "Resuma os riscos". Isso resulta em saídas superficiais. O prompt deve ser altamente específico, definindo o papel, o objetivo, o formato e as restrições de saída.

## URL
[https://www.v7labs.com/blog/ai-due-diligence](https://www.v7labs.com/blog/ai-due-diligence)
