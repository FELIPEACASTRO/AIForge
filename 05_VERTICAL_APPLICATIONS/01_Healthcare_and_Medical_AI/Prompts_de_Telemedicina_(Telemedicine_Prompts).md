# Prompts de Telemedicina (Telemedicine Prompts)

## Description
**Prompts de Telemedicina** referem-se a comandos e instruções estruturadas, elaboradas por profissionais de saúde ou pacientes, para interagir com Modelos de Linguagem Grande (LLMs) e IA Generativa no contexto da prestação de cuidados de saúde à distância. A Engenharia de Prompt na Telemedicina é crucial para otimizar a precisão, relevância e segurança das respostas da IA, transformando-a em uma ferramenta de suporte à decisão clínica, automação administrativa e comunicação com o paciente [1] [2]. Esses prompts são projetados para lidar com a complexidade e a sensibilidade dos dados de saúde, exigindo alta especificidade e contextualização para garantir a segurança do paciente e a conformidade regulatória (como HIPAA e LGPD) [3]. O uso eficaz de prompts permite que a IA auxilie em tarefas como triagem inicial, geração de resumos de consultas, redação de notas clínicas e criação de materiais educativos personalizados para o paciente [1].

## Examples
```
**1. Triagem e Priorização (Zero-Shot Prompting)**
`**Prompt:** "Você é um assistente de triagem de telemedicina. Analise os seguintes sintomas e determine o nível de urgência (Emergência, Urgente, Rotina) e a especialidade médica mais apropriada. **Sintomas:** Paciente de 45 anos, sexo feminino, relata dor de cabeça súbita e intensa (pior dor da vida), acompanhada de rigidez na nuca e vômitos há 2 horas. Histórico de hipertensão controlada. **Formato de Saída:** Tabela com Urgência, Especialidade e Justificativa."`

**2. Geração de Resumo de Consulta (One-Shot Prompting)**
`**Prompt:** "Gere um resumo conciso da teleconsulta para o prontuário eletrônico. **Exemplo de Formato:** [Data], [Nome do Paciente], [Motivo da Consulta], [Achados Principais], [Plano de Tratamento]. **Dados da Consulta:** 08/11/2025, João Silva, 68 anos. Motivo: Acompanhamento de Diabetes Mellitus Tipo 2. Achados: Glicemia em jejum média de 150 mg/dL nas últimas 2 semanas. Plano: Ajuste de Metformina para 1000mg 2x/dia e solicitação de HbA1c."`

**3. Suporte à Decisão Clínica (Chain-of-Thought Prompting)**
`**Prompt:** "Você é um consultor clínico de IA. Para o caso a seguir, liste 3 diagnósticos diferenciais mais prováveis e, em seguida, justifique o raciocínio para cada um, citando a fonte (diretriz ou artigo). **Caso:** Paciente de 72 anos, sexo masculino, com dispneia progressiva e edema de membros inferiores. Exame físico: estertores crepitantes em bases pulmonares e turgência jugular. Histórico de infarto do miocárdio há 5 anos. **Instrução:** Primeiro, raciocine sobre a fisiopatologia. Depois, liste os diferenciais."`

**4. Comunicação com o Paciente (Linguagem Leiga)**
`**Prompt:** "Explique a condição médica 'Fibrilação Atrial' para um paciente de 55 anos com ensino médio completo, utilizando analogias simples e evitando jargões médicos. Inclua a importância da medicação anticoagulante e os sinais de alerta. **Tom de Voz:** Empático e educativo."`

**5. Criação de Roteiro de Atendimento (Meta-Prompting)**
`**Prompt:** "Crie um roteiro de atendimento estruturado para uma teleconsulta de acompanhamento de pacientes com ansiedade generalizada. O roteiro deve incluir: 1. Perguntas de triagem inicial (escala GAD-7). 2. Avaliação da adesão ao tratamento. 3. Discussão de estratégias de coping (ex: técnica de respiração). 4. Definição de metas para a próxima sessão. **Formato:** Lista numerada e detalhada."`

**6. Análise de Dados de Monitoramento Remoto**
`**Prompt:** "Analise os dados de monitoramento remoto de um paciente com Insuficiência Cardíaca Congestiva (ICC) e identifique qualquer tendência preocupante. **Dados:** Peso diário (aumento de 2kg em 3 dias), Pressão Arterial (média 135/85 mmHg), Frequência Cardíaca (média 88 bpm). **Instrução:** Foque na relação entre o ganho de peso e a retenção hídrica, sugerindo uma ação imediata para o médico responsável."`

**7. Geração de Material Educativo Personalizado**
`**Prompt:** "Gere um plano alimentar semanal de 7 dias para um paciente vegetariano de 40 anos com diagnóstico recente de hipercolesterolemia (colesterol LDL alto). O plano deve ser rico em fibras solúveis e incluir a contagem aproximada de calorias diárias (máximo 2000 kcal). **Formato:** Tabela com café da manhã, almoço e jantar."`
```

## Best Practices
**1. Seja Específico e Contextualizado:** Inclua o papel da IA (ex: "Você é um assistente de triagem"), o contexto clínico (ex: "paciente de 65 anos com histórico de ICC"), e o objetivo claro (ex: "gerar um resumo de alta"). **2. Utilize a Estrutura de Raciocínio (Chain-of-Thought):** Peça à IA para detalhar o processo de raciocínio antes da resposta final, especialmente para diagnósticos ou planos de tratamento complexos. Isso aumenta a transparência e a confiabilidade. **3. Referencie Diretrizes:** Sempre que possível, solicite que a IA baseie sua resposta em diretrizes clínicas atualizadas (ex: "de acordo com as diretrizes da AHA 2023"). **4. Defina o Formato de Saída:** Especifique o formato desejado (ex: "lista com marcadores", "tabela", "texto em linguagem leiga") para garantir a usabilidade. **5. Priorize a Segurança e a Ética:** Lembre a IA sobre a confidencialidade dos dados (HIPAA/LGPD) e a necessidade de que todas as sugestões sejam revisadas por um profissional humano.

## Use Cases
**1. Suporte à Decisão Clínica Remota:** Auxílio na formulação de diagnósticos diferenciais, sugestão de planos de tratamento e interpretação de exames laboratoriais ou de imagem (teleradiologia) [2]. **2. Automação de Tarefas Administrativas:** Geração de resumos de alta, notas de progresso, cartas de encaminhamento e preenchimento de formulários de seguro, liberando tempo do profissional de saúde [1]. **3. Comunicação e Educação do Paciente:** Criação de materiais educativos personalizados, respostas a perguntas frequentes (FAQ) e elaboração de mensagens empáticas e claras para pacientes, superando barreiras de letramento em saúde [1]. **4. Triagem e Priorização de Pacientes:** Uso de chatbots e assistentes de IA para coletar sintomas, avaliar a urgência e direcionar o paciente para o nível de cuidado apropriado (teleconsulta, pronto-socorro, etc.) [4]. **5. Otimização de Fluxos de Trabalho:** Criação de fluxogramas detalhamento da jornada do paciente em telemedicina, desde o agendamento até o acompanhamento pós-consulta.

## Pitfalls
**1. Alucinações Clínicas:** A IA pode gerar informações falsas, clinicamente incorretas ou desatualizadas (alucinações), o que é perigoso em um contexto de saúde. **Mitigação:** Exigir citações de fontes confiáveis e sempre ter a revisão final por um profissional humano. **2. Viés e Iniquidade:** Se os dados de treinamento da IA forem enviesados, os prompts podem levar a recomendações que perpetuam disparidades de saúde em grupos minoritários. **Mitigação:** Incluir dados demográficos e clínicos detalhados no prompt para garantir a relevância e a equidade. **3. Violação de Privacidade (HIPAA/LGPD):** O uso de dados sensíveis de pacientes em prompts pode violar regulamentações de privacidade. **Mitigação:** Utilizar apenas dados anonimizados ou prompts que manipulem informações genéricas ou sintéticas, e nunca inserir Informações de Saúde Protegidas (PHI) em modelos de IA não validados para tal. **4. Falta de Contexto Humano:** A IA não substitui a empatia e o julgamento clínico. Prompts que buscam um "diagnóstico final" sem a interação humana ignoram a complexidade da medicina. **Mitigação:** Usar a IA como um **assistente** (ferramenta de suporte), e não como um decisor final. **5. Prompts Vagos:** Prompts genéricos levam a respostas genéricas e inúteis. **Mitigação:** Ser o mais específico possível, definindo o papel da IA, o público-alvo, o formato de saída e as restrições de conteúdo.

## URL
[https://pmc.ncbi.nlm.nih.gov/articles/PMC12439060/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12439060/)
