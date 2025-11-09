# Geriatrics Prompts

## Description
**Geriatrics Prompts** refere-se à aplicação especializada da Engenharia de Prompts (Prompt Engineering) no domínio da Geriatria e cuidados com a saúde do idoso. Não é uma técnica de prompt em si, mas uma **categoria de aplicação** que visa otimizar a interação com Grandes Modelos de Linguagem (LLMs) para abordar as complexidades únicas do envelhecimento. O foco é na criação de prompts que considerem fatores como polifarmácia, fragilidade, comorbidades múltiplas, declínio cognitivo e a necessidade de comunicação adaptada. A meta é utilizar a IA como uma ferramenta de suporte à decisão clínica, educação de profissionais e pacientes, e na personalização de planos de cuidado, como demonstrado pelo desenvolvimento de modelos específicos como o *Geriatric Care LLaMA*.

## Examples
```
## 1. Suporte à Decisão Clínica (Polifarmácia)
**Prompt:**
"Você é um farmacêutico clínico especializado em geriatria. Analise a seguinte lista de medicamentos para um paciente de 82 anos com insuficiência cardíaca (IC) e doença de Parkinson: [Medicamento A, Dose], [Medicamento B, Dose], [Medicamento C, Dose], [Medicamento D, Dose]. Identifique potenciais interações medicamentosas perigosas, avalie se algum medicamento está na lista BEERS e sugira uma alternativa mais segura para o medicamento de maior risco, justificando a escolha com base em evidências."

## 2. Educação e Treinamento (Simulação de Caso)
**Prompt:**
"Crie um cenário de simulação de alta fidelidade para um residente de anestesiologia. O paciente é um homem de 78 anos, frágil (índice de fragilidade 5/9), submetido a uma cirurgia de fratura de fêmur. O cenário deve começar no intraoperatório, com uma queda súbita da pressão arterial (PA) e um quadro de delirium emergente. Inclua os sinais vitais iniciais, o histórico médico relevante (incluindo polifarmácia) e as três primeiras perguntas que o residente deve fazer para estabilizar o paciente e gerenciar o delirium."

## 3. Comunicação Adaptada (Simplificação de Informação)
**Prompt:**
"Simplifique o seguinte trecho de um relatório médico sobre 'Fibrilação Atrial' para um paciente de 90 anos com baixa alfabetização em saúde. O texto simplificado deve ter um tom tranquilizador, usar frases curtas, evitar jargões médicos e explicar o que é a condição e por que o medicamento [Nome do Anticoagulante] é importante. O nível de leitura deve ser equivalente ao da 4ª série."

## 4. Pesquisa e Desenvolvimento (Inferência Causal)
**Prompt:**
"Com base no seu conhecimento sobre o modelo Geriatric Care LLaMA, formule um prompt causal para investigar a relação entre 'uso de benzodiazepínicos' e 'risco de quedas' em pacientes geriátricos com demência leve. O prompt deve solicitar a identificação de variáveis de confusão (ex: comorbidades, dose, tempo de uso) e sugerir um protocolo de intervenção para mitigar o risco, estruturado em etapas."

## 5. Plano de Cuidados Personalizado
**Prompt:**
"Elabore um plano de cuidados domiciliares de 7 dias para uma paciente de 88 anos que recebeu alta após um episódio de pneumonia. O plano deve focar em: 1) Prevenção de quedas (com 3 ações concretas), 2) Nutrição (com 2 sugestões de refeições ricas em proteínas) e 3) Gerenciamento da medicação (com um cronograma simplificado). O tom deve ser encorajador e dirigido ao cuidador principal."

## 6. Geração de Conteúdo Educacional para Cuidadores
**Prompt:**
"Crie um guia rápido (máximo 500 palavras) para cuidadores sobre como identificar os sinais precoces de desidratação em idosos. O guia deve incluir uma lista de verificação de 5 sinais de alerta e 3 dicas práticas para incentivar a ingestão de líquidos, formatado para ser impresso em uma fonte grande e legível."

## 7. Assistência Administrativa (Nota de Progresso)
**Prompt:**
"Gere uma nota de progresso SOAP (Subjetivo, Objetivo, Avaliação, Plano) para um paciente de 95 anos em uma casa de repouso.
- **Subjetivo:** Queixa de dor 4/10 na perna esquerda, relata 'sono ruim' na noite passada.
- **Objetivo:** Sinais vitais estáveis. Edema 1+/4+ na perna esquerda. Deambulando com auxílio de andador, mas com marcha instável.
- **Avaliação:** Piora da dor crônica na perna esquerda, risco de queda elevado.
- **Plano:** Solicite a nota de progresso completa, focando na necessidade de reavaliação da medicação para dor e fisioterapia imediata para estabilização da marcha."
```

## Best Practices
1. **Especificidade Geriátrica:** Incluir explicitamente no prompt as condições específicas do paciente idoso (ex: "Paciente com 85 anos, frágil, com histórico de IC e polifarmácia").
2. **Foco no Suporte, Não na Substituição:** Estruturar prompts para que a IA atue como um "assistente" que gera rascunhos, análises ou sugestões, mantendo a supervisão e o julgamento clínico humano como etapa final.
3. **Ênfase na Usabilidade:** Ao gerar conteúdo para idosos, instruir a IA a usar linguagem simples, frases curtas e formatação clara (ex: "Use uma linguagem de nível de leitura da 4ª série e evite jargões médicos").
4. **Consideração Ética e de Privacidade:** Incluir instruções de segurança e privacidade no prompt, lembrando a IA sobre a confidencialidade dos dados do paciente (embora a entrada de dados reais deva ser evitada).
5. **Validação de Fontes:** Sempre que possível, pedir à IA para citar as diretrizes clínicas ou fontes de evidência que embasam suas recomendações, especialmente em contextos de tratamento.

## Use Cases
1. **Suporte à Decisão Clínica:** Geração de diagnósticos diferenciais, planos de cuidado personalizados e recomendações de tratamento que considerem a polifarmácia e as interações medicamentosas em pacientes idosos.
2. **Educação e Treinamento:** Criação de cenários de simulação de alta fidelidade (ex: anestesia geriátrica) e módulos de aprendizado adaptativo para profissionais de saúde, focados em síndromes geriátricas (delirium, quedas, fragilidade).
3. **Comunicação Adaptada:** Simplificação de informações médicas complexas em linguagem acessível para pacientes idosos e seus cuidadores, incluindo a geração de materiais educativos e lembretes de saúde.
4. **Pesquisa e Desenvolvimento:** Utilização de prompts causais para explorar relações de causa e efeito em grandes conjuntos de dados clínicos, identificando fatores de risco e otimizando a gestão de doenças crônicas.
5. **Assistência Administrativa:** Geração de relatórios de alta, notas de progresso e documentação clínica que atendam aos requisitos regulatórios, liberando tempo dos profissionais para o cuidado direto.

## Pitfalls
1. **Substituição do Julgamento Clínico:** Confiar cegamente nas saídas da IA sem a devida validação por um profissional de saúde, ignorando nuances clínicas e a complexidade do paciente idoso.
2. **Ignorar a Usabilidade:** Gerar conteúdo para idosos que é muito complexo, técnico ou mal formatado, levando à baixa aceitação e confusão.
3. **Violação de Privacidade:** Inserir informações de saúde protegidas (PHI) em LLMs públicos ou não seguros.
4. **Generalização Excessiva:** Usar prompts genéricos que não consideram a fragilidade, as comorbidades e a heterogeneidade da população idosa, resultando em recomendações inadequadas.
5. **"Alucinações" Clínicas:** A IA gerar informações médicas falsas ou desatualizadas, o que é particularmente perigoso em um campo sensível como a geriatria.

## URL
[https://www.mdpi.com/2227-7390/13/15/2460](https://www.mdpi.com/2227-7390/13/15/2460)
