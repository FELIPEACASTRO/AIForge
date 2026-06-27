# Obstetrics & Gynecology Prompts

## Description
A Engenharia de Prompt para Obstetrícia e Ginecologia (OB/GYN) refere-se à arte e ciência de estruturar entradas de linguagem natural (prompts) para modelos de linguagem grande (LLMs) e chatbots de Inteligência Artificial (IA) para obter resultados precisos, clinicamente relevantes e eticamente responsáveis no campo da saúde da mulher. Esta técnica é crucial para alavancar o potencial da IA em diversas áreas, incluindo **educação médica**, **suporte à decisão clínica**, **comunicação com pacientes** e **pesquisa**. A eficácia dos prompts em OB/GYN depende diretamente da **especificidade**, da **inclusão de contexto clínico** e da **exigência de fontes ou diretrizes** para mitigar riscos como "alucinações" e vieses inerentes aos modelos [1] [2].

## Examples
```
1.  **Suporte à Decisão Clínica (Diagnóstico Diferencial):**
    `"Aja como um residente sênior de OB/GYN. Uma paciente de 32 anos, G2P1, apresenta sangramento vaginal no primeiro trimestre. O beta-hCG é de 1500 mIU/mL. Liste 5 diagnósticos diferenciais mais prováveis, a probabilidade relativa de cada um e o próximo passo diagnóstico crucial para cada cenário. Formate a resposta em uma tabela."`

2.  **Educação Médica (Cenário de Simulação):**
    `"Crie um cenário de simulação de 10 minutos para um estudante de medicina do 3º ano sobre o manejo da pré-eclâmpsia grave. O cenário deve incluir: 1) Um breve histórico do paciente, 2) Três perguntas críticas que o estudante deve fazer, e 3) O tratamento inicial de primeira linha baseado nas diretrizes do ACOG (2023)."`

3.  **Comunicação com o Paciente (Resposta a Mensagem):**
    `"Reescreva a seguinte mensagem clínica em linguagem simples e empática para uma paciente com baixo letramento em saúde. A paciente tem 45 anos e foi diagnosticada com miomas uterinos. A mensagem original é: 'Seus miomas são intramurais e submucosos, medindo 5 cm e 3 cm, respectivamente. Discutiremos as opções de tratamento, incluindo embolização e miomectomia, na sua próxima consulta.'"`

4.  **Revisão de Literatura e Diretrizes:**
    `"Compare as recomendações de rastreamento de câncer de colo de útero para pacientes imunocomprometidas (HIV positivo) versus a população geral, de acordo com as diretrizes mais recentes (2023-2025) do USPSTF e ACOG. Cite a fonte primária para cada recomendação."`

5.  **Otimização de Fluxo de Trabalho (Documentação):**
    `"Gere um modelo de nota SOAP (Subjetivo, Objetivo, Avaliação, Plano) para uma consulta de rotina de planejamento familiar, focando na discussão de métodos contraceptivos reversíveis de longa ação (LARC). Inclua seções para consentimento informado e acompanhamento."`

6.  **Pesquisa e Análise de Dados (Estatística):**
    `"Explique o conceito de 'Number Needed to Treat' (NNT) e como ele se aplica à decisão de prescrever progesterona para prevenção de parto prematuro em pacientes com colo uterino curto. Use um tom didático e forneça um exemplo numérico hipotético."`
```

## Best Practices
*   **Definir a Persona e o Público-Alvo:** Comece o prompt instruindo a IA a assumir um papel específico (ex: "Aja como um especialista em ultrassonografia fetal", "Aja como um educador de pacientes") e defina o público-alvo (ex: "para um residente", "para uma paciente leiga") [1].
*   **Especificidade Clínica:** Inclua detalhes clínicos cruciais (idade, paridade, resultados de exames, comorbidades) e o contexto (ex: "primeiro trimestre", "pós-menopausa") para evitar respostas genéricas [1].
*   **Exigir Referências e Diretrizes:** Peça explicitamente que a IA cite a fonte de suas informações (ex: "baseado nas diretrizes do ACOG 2023", "cite o estudo original") para facilitar a validação [3].
*   **Estrutura de Saída Clara:** Especifique o formato desejado (ex: "em uma tabela", "em formato de nota SOAP", "lista de 5 pontos") para garantir a usabilidade clínica do resultado [1].
*   **Loop de Feedback Estruturado:** O prompt inicial deve ser seguido por prompts de refinamento (ex: "Refine a resposta para incluir a dosagem de sulfato de magnésio", "Corrija o erro de dosagem para 4g IV") para iterar e validar a precisão clínica [3].

## Use Cases
*   **Suporte à Decisão Clínica:** Geração de diagnósticos diferenciais, planos de manejo iniciais e comparação de protocolos de tratamento (ex: manejo de hemorragia pós-parto, triagem de câncer ginecológico) [1].
*   **Educação e Treinamento:** Criação de cenários de simulação, questões de múltipla escolha, resumos de artigos complexos e planos de aula para residentes e estudantes [2].
*   **Comunicação com o Paciente:** Redação de respostas a mensagens de pacientes, criação de materiais educativos em linguagem acessível e otimização da legibilidade de documentos de consentimento [1].
*   **Pesquisa e Revisão de Literatura:** Síntese rápida de grandes volumes de literatura, identificação de lacunas de pesquisa e comparação de resultados de ensaios clínicos [1].

## Pitfalls
*   **Alucinações Clínicas:** A IA pode gerar informações factualmente incorretas ou clinicamente implausíveis. A validação humana por um profissional de saúde é **obrigatória** [1].
*   **Vieses e Iniquidade:** Os modelos podem perpetuar vieses presentes nos dados de treinamento, levando a recomendações que podem ser inadequadas para grupos populacionais minoritários ou sub-representados [1].
*   **Falta de Consciência em Tempo Real:** A IA não tem acesso a dados clínicos em tempo real (ex: prontuários eletrônicos) ou às últimas atualizações de diretrizes que surgiram após seu treinamento [1].
*   **Violação de Privacidade de Dados (HIPAA/LGPD):** A inserção de informações de saúde protegidas (PHI) nos prompts viola as regulamentações de privacidade. Os prompts devem ser **anonimizados** e **desidentificados** [1].

## URL
[https://www.sciencedirect.com/science/article/pii/S0002937825002285](https://www.sciencedirect.com/science/article/pii/S0002937825002285)
