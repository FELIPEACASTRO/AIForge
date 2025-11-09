# Prompts de Diagnóstico Médico

## Description
A Engenharia de Prompts para Diagnóstico Médico é a disciplina de criar instruções estruturadas e calibradas para Modelos de Linguagem Grande (LLMs) com o objetivo de auxiliar no raciocínio clínico, na geração de diagnósticos diferenciais e na análise de dados de pacientes. Estudos recentes (2024-2025) demonstram que a precisão diagnóstica dos LLMs, como o Claude 3.5 Sonnet e GPT-4, pode ser significativamente melhorada através de técnicas de prompt que imitam o processo de raciocínio clínico humano, como a abordagem de duas etapas (Two-Step Clinical Reasoning) [1]. Essa abordagem envolve primeiro a sumarização e organização estruturada dos dados do paciente pelo LLM, seguida pela solicitação de diagnóstico baseada nesse resumo. A clareza, a precisão e a inclusão de contexto detalhado são cruciais para mitigar vieses e a ocorrência de "alucinações" (informações fabricadas) [3]. O uso de prompts de diagnóstico é uma ferramenta promissora para otimizar a rotina clínica, mas exige cautela e validação humana rigorosa.

## Examples
```
1. **Prompt de Raciocínio Clínico Estruturado (Two-Step):**
   *   **Passo 1 (Sumarização):** "Você é um radiologista experiente. Sua tarefa é resumir o caso clínico a seguir, organizando as informações nas seguintes categorias com marcadores concisos: Informações do Paciente (idade, sexo), Histórico da Doença Atual, Histórico Médico Pregresso, Sintomas Chave, Achados de Imagem. Caso uma categoria não tenha informação, escreva 'Sem informação'. [INSERIR DADOS BRUTOS DO PACIENTE]"
   *   **Passo 2 (Diagnóstico):** "Com base no resumo estruturado que você gerou, aja como um médico. Apresente o raciocínio passo a passo, o diagnóstico mais provável e os dois diagnósticos diferenciais mais prováveis. Inclua a justificativa para cada um."

2. **Prompt de Diagnóstico Diferencial com Restrição de Formato:**
   "Aja como um clínico geral. Um paciente de 45 anos, sexo masculino, apresenta febre persistente (38.5°C), tosse seca e dispneia progressiva há 10 dias. Histórico de viagem recente para o Sudeste Asiático. Raio-X de tórax mostra infiltrado intersticial bilateral.
   Liste os 5 diagnósticos diferenciais mais prováveis em uma tabela. Para cada diagnóstico, inclua: 1) Probabilidade (Alta, Média, Baixa), 2) Justificativa Clínica, e 3) Exame Complementar de Próximo Passo."

3. **Prompt para Análise de Resultados Laboratoriais:**
   "Você é um hematologista. Analise os seguintes resultados de hemograma e sugira a condição mais provável e a conduta inicial.
   *   Hemoglobina: 9.5 g/dL (Baixa)
   *   VCM: 75 fL (Baixo)
   *   Leucócitos: 12.000/mm³ (Alto)
   *   Plaquetas: 450.000/mm³ (Alto)
   *   Ferritina: 10 ng/mL (Muito Baixa)
   Condição mais provável: [Resposta]
   Raciocínio: [Resposta]
   Conduta inicial: [Resposta]"

4. **Prompt para Simulação de Paciente Virtual (Role-Play):**
   "Aja como um paciente de 68 anos, chamado João, com histórico de tabagismo. Você está na consulta com o médico. Seus sintomas são dor no peito ao tossir e perda de peso não intencional nos últimos 3 meses. Responda às perguntas do médico de forma concisa e com um tom de voz preocupado. Não revele o diagnóstico de câncer de pulmão até que o médico solicite um exame de imagem."

5. **Prompt para Revisão de Diretrizes Clínicas:**
   "Com base nas diretrizes mais recentes da Sociedade Brasileira de Cardiologia (SBC) (2023-2025), resuma o protocolo de tratamento de primeira linha para Insuficiência Cardíaca com Fração de Ejeção Reduzida (ICFEr). Inclua as classes de medicamentos recomendadas e a meta de dose ideal para cada uma."
```

## Best Practices
**Estrutura de Raciocínio Clínico em Duas Etapas (Two-Step Clinical Reasoning):** Divida o processo em duas etapas: 1) Solicite ao LLM que **resuma e organize** os dados clínicos brutos (histórico, sintomas, exames) em categorias estruturadas. 2) Use o **resumo estruturado** como entrada para a etapa de diagnóstico, solicitando o diagnóstico diferencial e o raciocínio. Isso comprovadamente aumenta a precisão [1]. **Definição de Papel (Role-Play Prompting):** Comece o prompt definindo o papel do LLM (ex: "Você é um radiologista experiente", "Aja como um clínico geral"). Isso alinha a resposta do modelo com o contexto clínico necessário [1]. **Inclusão de Contexto e Restrições:** Forneça o máximo de contexto possível (idade, sexo, histórico, resultados de exames). Restrinja o formato de saída (ex: "Liste os 3 diagnósticos mais prováveis em formato de tabela, incluindo a probabilidade e o próximo passo de investigação"). **Transparência e Raciocínio (Chain-of-Thought - CoT):** Peça ao LLM para detalhar seu processo de raciocínio (CoT) antes de apresentar o diagnóstico final. Isso aumenta a interpretabilidade e permite a validação humana do processo [2]. **Validação Humana Obrigatória:** O output do LLM deve ser sempre tratado como um **suporte à decisão** e nunca como um diagnóstico final. A revisão e validação por um profissional de saúde qualificado é essencial [3].

## Use Cases
**Suporte à Decisão Clínica:** Geração de listas de diagnósticos diferenciais para casos complexos ou atípicos, garantindo que nenhuma condição rara seja negligenciada. **Educação Médica e Simulação:** Criação de cenários clínicos detalhados, scripts para pacientes simulados (SPs) e estações de Exame Clínico Objetivo Estruturado (OSCE), otimizando o treinamento de estudantes e residentes [3]. **Análise de Dados Não Estruturados:** Extração e sumarização de informações clínicas relevantes de prontuários eletrônicos em formato de texto livre, relatórios de radiologia ou notas de progresso [1]. **Revisão de Literatura e Diretrizes:** Geração de resumos rápidos e comparativos de protocolos de tratamento e diretrizes clínicas atualizadas, auxiliando na tomada de decisão baseada em evidências. **Interpretação de Exames:** Auxílio na interpretação de resultados laboratoriais ou achados de imagem, sugerindo correlações clínicas e próximos passos de investigação.

## Pitfalls
**Alucinações (Fabricação de Fatos):** O LLM pode gerar informações médicas falsas, mas que parecem plausíveis. **Mitigação:** Sempre verifique o output com fontes médicas confiáveis e diretrizes clínicas. **Vieses de Dados:** O modelo pode refletir vieses presentes nos dados de treinamento (ex: sub-representação de certas etnias ou condições raras), levando a diagnósticos imprecisos ou incompletos para esses grupos. **Mitigação:** Inclua explicitamente informações demográficas e solicite que o modelo considere o diagnóstico diferencial em populações diversas. **Falta de Transparência (Black Box):** Sem a solicitação de um raciocínio passo a passo (CoT), o diagnóstico é um "caixa preta". **Mitigação:** Sempre use a técnica CoT para entender a lógica do modelo. **Uso de Dados Sensíveis:** Nunca insira informações de identificação pessoal (PII) ou dados de saúde protegidos (PHI) em LLMs de uso geral, devido a preocupações com a privacidade e conformidade (ex: LGPD, HIPAA) [3]. **Mitigação:** Use apenas dados anonimizados ou modelos LLM especializados e certificados para saúde (ex: MedPaLM, Azure Health).

## URL
[https://www.medrxiv.org/content/10.1101/2024.09.01.24312894v1.full-text](https://www.medrxiv.org/content/10.1101/2024.09.01.24312894v1.full-text)
