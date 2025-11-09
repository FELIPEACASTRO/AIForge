# Treatment Planning Prompts

## Description
**Prompts de Planejamento de Tratamento** são instruções de engenharia de *prompt* projetadas para auxiliar profissionais de saúde (médicos, psicólogos, terapeutas, dentistas) na criação, refinamento e documentação de planos de tratamento abrangentes e personalizados. Esta técnica aproveita a capacidade dos Grandes Modelos de Linguagem (LLMs) de sintetizar informações clínicas complexas, diretrizes baseadas em evidências e dados específicos do paciente para gerar objetivos, intervenções e resultados esperados de forma estruturada. O foco principal é aprimorar a eficiência, a conformidade regulatória (como HIPAA/GDPR) e a qualidade do cuidado, transformando a documentação clínica de uma tarefa administrativa demorada em um processo assistido por IA. A eficácia desses *prompts* depende da inclusão de detalhes clínicos precisos, da especificação do formato de saída desejado (ex: SOAP, DAP, SMART goals) e da adesão a princípios éticos, como a anonimização de dados. Esta técnica é fundamental para a adoção da IA na prática clínica moderna, especialmente em áreas como saúde mental e medicina especializada.

## Examples
```
1. **Geração de Plano de Tratamento Estruturado (Saúde Mental):**
```
Aja como um psicólogo clínico especializado em Terapia Cognitivo-Comportamental (TCC).
**Paciente:** [Nome Fictício], 32 anos, sexo feminino.
**Diagnóstico:** Transtorno de Ansiedade Generalizada (TAG).
**Sintomas-Alvo:** Preocupação excessiva diária, insônia, tensão muscular.
**Histórico:** Tentativa de tratamento com medicação (SSRI) com efeitos colaterais.
**Tarefa:** Crie um plano de tratamento de 12 sessões. O plano deve incluir:
1. Um **Objetivo de Longo Prazo** (SMART).
2. Três **Objetivos de Curto Prazo** (SMART) focados em psicoeducação, reestruturação cognitiva e relaxamento.
3. **Intervenções** específicas de TCC para cada objetivo.
4. **Critérios de Alta** (mensuráveis).
```

2. **Refinamento de Intervenções Médicas (Cardiologia):**
```
Aja como um cardiologista consultor.
**Cenário:** Paciente de 68 anos com Insuficiência Cardíaca com Fração de Ejeção Reduzida (ICFER, FE 35%), Hipertensão e Diabetes Tipo 2.
**Medicação Atual:** Enalapril e Metoprolol.
**Tarefa:** Recomende a próxima etapa na otimização do tratamento farmacológico, conforme as diretrizes da ACC/AHA de 2022.
**Saída:** Liste as classes de medicamentos (ex: SGLT2i, MRA) que devem ser adicionadas, justificando a escolha com base nos benefícios de mortalidade e hospitalização.
```

3. **Criação de Objetivos SMART (Terapia Ocupacional):**
```
Aja como um terapeuta ocupacional.
**Paciente:** Criança de 7 anos com Transtorno do Espectro Autista (TEA).
**Problema:** Dificuldade em vestir-se de forma independente (abotoar camisas).
**Tarefa:** Gere 3 Objetivos SMART para as próximas 8 semanas de terapia.
**Formato:** Objetivo (Específico, Mensurável, Alcançável, Relevante, Temporal) e Intervenção Associada.
```

4. **Elaboração de Documentação para Auditoria (Odontologia):**
```
Aja como um especialista em codificação e faturamento odontológico.
**Procedimento:** Colocação de coroa em molar (código D2740).
**Justificativa Clínica:** Fratura cuspídea extensa, restauração prévia falhada.
**Tarefa:** Redija uma "Declaração de Necessidade Médica" concisa e profissional para a seguradora, explicando por que a coroa é o tratamento necessário e não uma restauração simples.
```

5. **Geração de Material Educacional para o Paciente:**
```
Aja como um educador de saúde.
**Tópico:** Novo regime de insulina (uso de caneta e monitoramento de glicemia).
**Público:** Paciente diabético recém-diagnosticado, 55 anos, com baixa alfabetização em saúde.
**Tarefa:** Crie um guia passo a passo, usando linguagem simples (nível de leitura da 5ª série), para o uso correto da caneta de insulina. Inclua 3 pontos de alerta de segurança.
```

6. **Análise de Risco e Planejamento de Crise (Psiquiatria):**
```
Aja como um psiquiatra.
**Cenário:** Paciente com Transtorno Bipolar, histórico de não adesão à medicação e aumento recente de ideação suicida passiva.
**Tarefa:** Desenvolva um "Plano de Segurança" de crise. O plano deve incluir:
1. Três sinais de alerta de crise.
2. Três estratégias de enfrentamento imediato.
3. Dois contatos de emergência (fictícios).
4. O próximo passo de tratamento se as estratégias falharem.
```
```

## Best Practices
**Explicitação e Especificidade Clínica:** Sempre inclua detalhes clínicos relevantes, como idade, comorbidades, estágio da doença e diretrizes específicas (ex: "per 2023 ADA guidelines"). Isso reduz a ambiguidade e aumenta a validade clínica da resposta.
**Relevância Contextual:** Forneça o máximo de contexto possível. Em saúde mental, isso inclui o diagnóstico, sintomas-alvo, histórico de tratamento e o ambiente do paciente.
**Refinamento Iterativo:** Não aceite a primeira resposta. Use o *feedback* clínico para refinar o *prompt* (ex: "Refine o objetivo de tratamento para ser mais mensurável, focando na redução da frequência de ataques de pânico de 5 para 2 por semana").
**Considerações Éticas e de Privacidade:** Nunca insira dados de identificação pessoal (DIP). Use dados anonimizados ou cenários hipotéticos.
**Práticas Baseadas em Evidências:** Peça ao modelo para citar fontes ou alinhar as recomendações com diretrizes clínicas atualizadas (ex: "Quais são as intervenções baseadas em evidências para TCC no tratamento de TAG, citando estudos recentes?").
**Definição de Papel (Role-Playing):** Comece o *prompt* definindo o papel da IA (ex: "Aja como um psicólogo clínico especializado em Terapia Cognitivo-Comportamental (TCC)").

## Use Cases
nan

## Pitfalls
**Violação de Privacidade (HIPAA/GDPR):** Inserir dados de identificação pessoal (DIP) ou informações de saúde protegidas (PHI) em LLMs de consumo.
**Alucinações Clínicas:** A IA pode gerar recomendações de tratamento factualmente incorretas, desatualizadas ou perigosas. A verificação humana é obrigatória.
**Generalização Excessiva:** Usar *prompts* vagos (ex: "Plano de tratamento para depressão") que resultam em planos genéricos e não adaptados à complexidade individual do paciente.
**Viés e Iniquidade:** O modelo pode perpetuar vieses de dados de treinamento, levando a planos de tratamento subótimos ou injustos para certos grupos demográficos.
**Falta de Contexto:** Não fornecer informações cruciais (comorbidades, alergias, histórico de falha de tratamento) que são essenciais para um plano seguro e eficaz.
**Dependência Excessiva:** Confiar cegamente na saída da IA sem aplicar o julgamento clínico e a experiência profissional.

## URL
[https://www.jmir.org/2025/1/e72644](https://www.jmir.org/2025/1/e72644)
