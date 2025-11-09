# Patient Education Prompts (Prompts de Educação de Pacientes)

## Description
Os **Prompts de Educação de Pacientes** (Patient Education Prompts) são instruções estruturadas e detalhadas fornecidas a Modelos de Linguagem Grande (LLMs) para gerar materiais informativos, personalizados e acessíveis sobre condições de saúde, tratamentos, procedimentos ou cuidados preventivos. No contexto clínico, esses prompts são uma ferramenta essencial para aprimorar a comunicação entre profissionais de saúde e pacientes, garantindo que a informação seja clinicamente precisa, eticamente responsável e adaptada ao nível de compreensão do indivíduo [1]. A técnica exige a inclusão de elementos cruciais como o **objetivo clínico** (ex: explicar o uso de um novo medicamento), o **público-alvo** (ex: paciente leigo, cuidador), o **formato de saída** (ex: folheto, analogia, FAQ) e a **base de evidências** (ex: diretrizes de 2023 da AHA) [1] [2]. O uso eficaz desses prompts transforma a IA em uma assistente poderosa na criação de conteúdo de saúde personalizado em escala.

## Examples
```
**1. Criação de Folheto Informativo (Nível Leigo):**
```
Aja como um educador de saúde. Crie um folheto informativo de uma página sobre "Hipertensão Arterial" para um paciente adulto com nível de alfabetização em saúde baixo. O folheto deve incluir:
1. Uma analogia simples para explicar o que é pressão alta.
2. Três mudanças de estilo de vida fáceis de implementar.
3. O que fazer se esquecer de tomar o medicamento.
4. Linguagem clara, encorajadora e sem jargões médicos.
```

**2. Explicação de Procedimento (Personalizado):**
```
Crie uma explicação passo a passo para um paciente de 65 anos com diabetes tipo 2 que fará uma colonoscopia. O texto deve ser escrito em um tom calmo e tranquilizador. Inclua instruções detalhadas sobre a dieta de preparo (dia anterior) e o que esperar durante e após o procedimento. O paciente tem ansiedade moderada em relação a procedimentos médicos.
```

**3. Comparação de Opções de Tratamento:**
```
Para um paciente de 40 anos diagnosticado com Doença de Crohn leve a moderada, gere um resumo comparativo entre as opções de tratamento com Mesalazina oral e Biológicos (anti-TNF). O resumo deve ser apresentado em formato de tabela, destacando eficácia, via de administração (oral vs. injeção) e potenciais efeitos colaterais mais comuns. Use uma linguagem de fácil compreensão.
```

**4. Roteiro de Conversa para o Médico:**
```
Gere um roteiro de 5 perguntas-chave que um paciente recém-diagnosticado com Lúpus Eritematoso Sistêmico (LES) deve fazer ao seu reumatologista na próxima consulta. As perguntas devem focar em prognóstico, monitoramento da doença e manejo de surtos.
```

**5. Instruções de Alta Hospitalar (Pós-Cirurgia):**
```
Elabore instruções de alta para um paciente que acabou de passar por uma cirurgia de substituição total do joelho. O documento deve ser uma lista de verificação (checklist) fácil de seguir, cobrindo:
1. Cuidados com a incisão.
2. Sinais de alerta para infecção (o que procurar e quando ligar).
3. Cronograma de medicação para dor (com nomes genéricos e comerciais).
4. Restrições de atividade e exercícios iniciais.
```

**6. Resposta a Dúvidas Comuns (Formato FAQ):**
```
Crie uma seção de Perguntas Frequentes (FAQ) para pais de crianças que receberão a vacina contra o HPV. Responda a pelo menos 5 mitos comuns sobre a vacina (ex: causa infertilidade, é desnecessária) com base em evidências do CDC (Centros de Controle e Prevenção de Doenças).
```

**7. Explicação por Analogia:**
```
Explique o mecanismo de ação da Insulina para um adolescente recém-diagnosticado com Diabetes Tipo 1. Use a analogia de uma "chave" que abre a "porta" da célula para que a "energia" (glicose) possa entrar.
```
```

## Best Practices
**1. Especificidade Clínica e Contextual:** Inclua detalhes do paciente (idade, comorbidades, estágio da doença) e referências a diretrizes clínicas atualizadas (ex: ADA, NCCN) para reduzir a ambiguidade e aumentar a validade clínica da resposta [1].
**2. Definição Clara do Público-Alvo:** Especifique o nível de alfabetização em saúde do paciente (ex: "leigo", "adolescente", "idoso com baixa visão") para que a IA ajuste a linguagem, o tom e o formato (ex: folheto, lista de verificação, analogia) [2] [3].
**3. Iteração e Refinamento:** O primeiro resultado da IA pode ser genérico. Use um ciclo de feedback estruturado para refinar o prompt, solicitando ajustes de tom, inclusão de informações específicas ou simplificação de termos complexos [1].
**4. Ênfase na Ética e Privacidade:** Ao formular prompts, priorize a desidentificação de dados sensíveis. Use a IA para gerar conteúdo educacional baseado em condições, não em prontuários eletrônicos não seguros [1].
**5. Verificação de Evidências:** Sempre solicite que a IA baseie o material em evidências e diretrizes atuais. **Cruze a informação gerada com fontes autorizadas** (ex: UpToDate, PubMed) para mitigar o risco de "alucinações" ou informações desatualizadas [1].

## Use Cases
**1. Criação de Materiais de Saúde Personalizados:** Geração rápida de folhetos, infográficos, FAQs e vídeos explicativos adaptados à condição e ao perfil demográfico de um paciente específico [2].
**2. Otimização da Comunicação Médico-Paciente:** Criação de roteiros de conversas para médicos ou pacientes, garantindo que todas as informações críticas sejam abordadas durante as consultas [3].
**3. Simplificação de Documentos Complexos:** Transformação de termos médicos complexos, resultados de exames ou planos de tratamento em linguagem acessível e fácil de entender para o paciente leigo [2].
**4. Suporte à Adesão ao Tratamento:** Desenvolvimento de lembretes de medicação, planos de dieta ou rotinas de exercícios pós-operatórios em formatos envolventes e motivacionais.
**5. Treinamento de Profissionais de Saúde:** Criação de cenários de simulação ou casos clínicos virtuais para treinar novos profissionais sobre como explicar condições complexas de forma eficaz aos pacientes [4].
**6. Geração de Conteúdo Multilíngue:** Tradução e adaptação cultural de materiais educacionais para pacientes que falam diferentes idiomas, garantindo a precisão clínica [2].

## Pitfalls
**1. Geração de "Alucinações" Clínicas:** A IA pode gerar informações falsas ou desatualizadas (alucinações) que, no contexto de saúde, são perigosas. **Mitigação:** Sempre exija a citação de fontes e verifique o conteúdo com diretrizes oficiais [1].
**2. Falha na Personalização:** Prompts genéricos resultam em materiais genéricos. A falta de detalhes do paciente (idade, comorbidades) pode levar a conselhos inadequados ou contraindicados [1].
**3. Linguagem Inapropriada:** O material gerado pode usar jargões médicos complexos, tornando-o inacessível para o público-alvo, especialmente aqueles com baixa alfabetização em saúde [2]. **Mitigação:** Especifique o nível de leitura desejado (ex: "nível de 6ª série").
**4. Viés e Inequidade:** A IA pode perpetuar vieses de dados, resultando em materiais que não são culturalmente sensíveis ou que negligenciam as barreiras de acesso de certos grupos populacionais [1]. **Mitigação:** Inclua restrições éticas e de equidade no prompt.
**5. Violação de Privacidade (HIPAA/LGPD):** Usar dados de pacientes não desidentificados em prompts de LLMs não seguros viola as leis de privacidade. **Mitigação:** Use apenas informações clínicas genéricas ou desidentificadas [1].

## URL
[https://pmc.ncbi.nlm.nih.gov/articles/PMC12439060/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12439060/)
