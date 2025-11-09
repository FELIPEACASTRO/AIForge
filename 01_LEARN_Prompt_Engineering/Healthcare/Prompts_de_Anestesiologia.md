# Prompts de Anestesiologia

## Description
A Engenharia de Prompts em Anestesiologia refere-se à criação e otimização de instruções (prompts) para Modelos de Linguagem Grande (LLMs) com o objetivo de auxiliar em tarefas clínicas, administrativas e educacionais no contexto perioperatório. Isso inclui a estratificação de risco (como a classificação ASA-PS), a geração de planos anestésicos preliminares, a simplificação de materiais educativos para pacientes e o suporte à pesquisa e análise de literatura médica. A técnica visa mitigar os riscos de alucinações e vieses, garantindo que as respostas sejam clinicamente relevantes, precisas e seguras. O uso de prompts complexos, como o Chain-of-Thought (CoT) e a inclusão de dados clínicos não estruturados (notas pré-operatórias) via RAG (Retrieval-Augmented Generation), tem demonstrado melhorar significativamente a acurácia do modelo em tarefas de classificação de risco e predição de desfechos.

## Examples
```
**1. Classificação ASA-PS com RAG (Few-Shot):**
`Você é um Anestesiologista Sênior. Sua tarefa é classificar o paciente de acordo com a Escala ASA-PS (I a VI) e justificar sua decisão com base nas informações fornecidas.
**Informações do Paciente:** [Inserir notas clínicas não estruturadas e histórico médico].
**Informações de Referência (RAG):** [Inserir trechos relevantes das diretrizes ASA-PS].
**Exemplo Few-Shot:** [Paciente 1: Histórico, Classificação, Justificativa].
**Formato de Saída:** {"ASA_PS": "Classe [I-VI]", "Justificativa": "[Texto da justificativa]"}`

**2. Geração de Plano Anestésico Preliminar (Zero-Shot):**
`Gere um plano anestésico pré-operatório completo para o seguinte cenário clínico.
**Cenário:** Paciente de 68 anos, sexo masculino, ASA II, com histórico de hipertensão controlada, agendado para colecistectomia laparoscópica eletiva.
**Inclua:** Avaliação de risco cardiovascular e pulmonar (Normal/Intermediário/Alto), Técnica anestésica sugerida (Geral/Regional/Combinada), Plano de monitorização (Invasiva/Não-invasiva), e Plano de manejo da dor pós-operatória (Fármacos e doses sugeridas).`

**3. Simplificação de Material Educativo para Pacientes:**
`Reescreva o seguinte texto sobre o risco de Hipertermia Maligna para um nível de leitura de 6ª série, usando linguagem simples e evitando jargões médicos.
**Texto Original:** [Inserir texto complexo sobre Hipertermia Maligna].
**Instrução Adicional:** Mantenha a precisão clínica e inclua três pontos de ação claros para o paciente.`

**4. Suporte à Decisão Clínica (Chain-of-Thought):**
`Analise o caso a seguir e forneça um diagnóstico diferencial para a hipotensão intraoperatória. Use o método Chain-of-Thought (CoT) para listar as etapas de raciocínio que levam à sua conclusão.
**Dados:** Paciente em cirurgia de coluna, 45 minutos de procedimento, FC 55 bpm, PA 80/45 mmHg, PVC 8 mmHg, EtCO2 32 mmHg. Anestesia geral com Sevoflurano 1.5 MAC.
**Formato de Saída:** 1. Análise de Dados. 2. Hipóteses (CoT). 3. Diagnóstico Mais Provável. 4. Plano de Ação Imediato.`

**5. Resumo de Literatura para Pesquisa:**
`Atue como um pesquisador de Anestesiologia. Revise os 5 artigos mais recentes (últimos 2 anos) sobre o uso de Dexmedetomidina em cirurgia cardíaca pediátrica.
**Tarefa:** Gere um resumo conciso (máx. 300 palavras) destacando as principais conclusões, a dosagem média utilizada e os efeitos adversos mais comuns. Forneça as referências em formato Vancouver.`
```

## Best Practices
**1. Especificação de Papel e Contexto (Role Assignment):** Comece o prompt definindo o LLM como um "Anestesiologista Sênior", "Residente de Anestesiologia" ou "Especialista em Risco Perioperatório" para garantir uma resposta com a perspectiva e o vocabulário adequados.
**2. Prompting Few-Shot e RAG (Retrieval-Augmented Generation):** Para tarefas críticas como classificação ASA-PS ou predição de mortalidade, inclua exemplos de casos anteriores (Few-Shot) e utilize a Geração Aumentada por Recuperação (RAG) para fundamentar a resposta em dados clínicos ou diretrizes específicas, reduzindo alucinações.
**3. Formato de Saída Estruturado (Output Formatting):** Exija que a saída seja em um formato estruturado (e.g., JSON, tabela Markdown) para facilitar a integração com sistemas de Prontuário Eletrônico (EHR) e a análise rápida pelo clínico.
**4. Ênfase na Segurança e Risco:** Inclua no prompt a instrução explícita para "priorizar a segurança do paciente" e "destacar quaisquer recomendações não-padrão ou de alto risco" para forçar o modelo a uma análise crítica.
**5. Validação Humana Obrigatória:** Sempre trate a saída do LLM como um **suporte à decisão** e não como uma decisão final. A validação por um profissional de saúde qualificado é a melhor prática fundamental.

## Use Cases
**1. Suporte à Decisão Clínica:** Auxílio na estratificação de risco pré-operatório (e.g., classificação ASA-PS), predição de desfechos pós-operatórios (e.g., mortalidade em 30 dias) e sugestão de planos anestésicos preliminares.
**2. Educação de Pacientes:** Geração de materiais educativos personalizados e simplificados sobre procedimentos anestésicos, riscos e cuidados pós-operatórios, adaptados ao nível de alfabetização do paciente.
**3. Automação Administrativa:** Geração eficiente de códigos de faturamento (billing codes), resumo de prontuários, e otimização de agendamento de salas de cirurgia e alocação de recursos.
**4. Pesquisa e Análise de Literatura:** Resumo rápido de artigos científicos, identificação de tendências em grandes volumes de literatura médica e auxílio na redação de manuscritos e perguntas de exames de residência.
**5. Chatbots de Suporte ao Paciente:** Fornecimento de suporte 24/7 para responder a perguntas comuns de pacientes sobre agendamento, preparo e recuperação, reduzindo a ansiedade e as chamadas para a equipe clínica.

## Pitfalls
**1. Alucinações (Hallucinations):** O modelo pode gerar informações plausíveis, mas clinicamente incorretas ou inventadas (e.g., doses de medicamentos erradas, procedimentos inadequados). Isso é especialmente perigoso em um campo de alto risco como a Anestesiologia.
**2. Viés de Treinamento:** Os LLMs podem refletir vieses presentes nos dados de treinamento, levando a recomendações que podem ser menos adequadas para populações específicas (e.g., minorias raciais, pacientes com comorbidades raras).
**3. Falta de Raciocínio Clínico Genuíno:** O modelo pode ser excelente em gerar texto coerente, mas a saída não é resultado de um raciocínio clínico complexo e contextualizado como o de um especialista humano.
**4. Inconsistência (Stochasticity):** O mesmo prompt pode gerar respostas diferentes, o que é inaceitável para protocolos clínicos. A falta de reprodutibilidade exige cautela.
**5. Ignorar o Contexto Não-Verbal:** LLMs baseados apenas em texto não podem interpretar dados visuais (como imagens de ultrassom ou gráficos de monitorização), limitando sua utilidade em cenários que exigem análise multimodal.
**6. Risco de Privacidade de Dados:** A inserção de dados de pacientes (mesmo que desidentificados) em modelos de uso geral pode violar regulamentações de privacidade (como HIPAA ou LGPD), exigindo o uso de modelos locais ou com fortes garantias de segurança.

## URL
[https://pmc.ncbi.nlm.nih.gov/articles/PMC12228656/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12228656/)
