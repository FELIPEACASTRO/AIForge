# Assessment Creation Prompts

## Description
**Assessment Creation Prompts** (Prompts para Criação de Avaliações) são instruções estruturadas e detalhadas fornecidas a modelos de Linguagem Grande (LLMs) para gerar, modificar ou refinar materiais de avaliação, como testes, questionários, tarefas e rubricas. O objetivo principal é otimizar o tempo de educadores e criadores de conteúdo, permitindo que a IA atue como um assistente na elaboração de itens de avaliação que sejam válidos, confiáveis e alinhados aos objetivos de aprendizagem específicos. A eficácia desses prompts reside na sua capacidade de especificar o formato da questão, o nível de dificuldade, o contexto do conteúdo e, idealmente, a inclusão de respostas e justificativas.

## Examples
```
1.  **Criação de Banco de Questões de Múltipla Escolha:**
    ```
    Aja como um professor universitário de [Área: Ciência da Computação].
    Crie 10 questões de múltipla escolha sobre o tópico "[Tópico: Estruturas de Dados - Listas Encadeadas]".
    As questões devem ser de nível de dificuldade [Nível: Intermediário/Avançado] e testar a [Habilidade: Aplicação e Análise].
    Para cada questão, forneça 4 alternativas, sendo uma correta. Inclua a resposta correta e uma breve justificativa para a resposta.
    Formato de saída: Questão, Alternativas (A, B, C, D), Resposta Correta, Justificativa.
    ```
2.  **Geração de Estudo de Caso para Avaliação Dissertativa:**
    ```
    Aja como um consultor de RH sênior.
    Crie um estudo de caso detalhado para uma avaliação dissertativa sobre "[Tema: Liderança em Crises]".
    O estudo de caso deve incluir: 1) Um cenário de crise empresarial (máximo 300 palavras); 2) O papel do aluno (ex: CEO, Gerente de Comunicação); 3) Uma pergunta de avaliação que exija uma resposta de 500 palavras, focada em [Foco: Estratégia de Comunicação e Tomada de Decisão Ética].
    ```
3.  **Variação de Nível de Dificuldade (Taxonomia de Bloom):**
    ```
    Pegue a seguinte questão de múltipla escolha: "[Questão Original]".
    Reescreva esta questão para que ela avalie a [Nível da Taxonomia de Bloom: Avaliação/Criação], em vez de apenas [Nível Original: Conhecimento/Compreensão].
    Mantenha o tópico central e o formato de múltipla escolha, mas crie um novo cenário ou exija uma análise mais profunda.
    ```
4.  **Criação de Distratores para Questão Existente:**
    ```
    A questão de múltipla escolha é: "Qual é a principal função do protocolo HTTP em uma comunicação web?" (Resposta Correta: C).
    As alternativas atuais são: A, B, C (Correta), D.
    Minha alternativa D é fraca. Gere uma nova alternativa D que seja um distrator plausível, mas incorreto, que se relacione com [Conceito Relacionado: Segurança de Rede] para confundir o aluno que não domina o tema.
    ```
5.  **Elaboração de Rubrica de Avaliação:**
    ```
    Aja como um especialista em avaliação educacional.
    Crie uma rubrica analítica de 4 níveis (Exemplar, Proficiente, Em Desenvolvimento, Insuficiente) para avaliar um projeto de [Tipo de Projeto: Design Thinking].
    A rubrica deve ter 4 critérios principais: [Critérios: 1. Definição do Problema, 2. Geração de Ideias, 3. Prototipagem, 4. Apresentação].
    Descreva detalhadamente o que constitui cada nível para cada critério.
    ```
6.  **Simulação de Prova Específica:**
    ```
    Crie um simulado de 15 questões de múltipla escolha sobre "[Tema: História do Brasil - Período Regencial]".
    O estilo das questões deve imitar fielmente o padrão de formulação e a complexidade da [Banca Examinadora: ENEM].
    Inclua a fonte de cada questão (ex: Texto 1, Imagem 1) e forneça a resposta correta com a resolução comentada.
    ```
```

## Best Practices
*   **Contextualização Completa:** Sempre forneça o contexto de aprendizagem (objetivos do curso, material de referência, público-alvo) antes de solicitar a criação da avaliação.
*   **Definição de Papel (Role Prompting):** Atribua à IA um papel específico (ex: "Especialista em Avaliação", "Elaborador de Provas Certificadas") para refinar o tom e a qualidade das questões.
*   **Especificação de Formato e Saída:** Exija um formato de saída claro (ex: JSON, Markdown, CSV) e o tipo de questão (múltipla escolha, dissertativa, etc.).
*   **Controle de Dificuldade:** Use a Taxonomia de Bloom ou termos como "nível de aplicação", "nível de análise" ou "nível de criação" para controlar a profundidade cognitiva da avaliação.
*   **Revisão Crítica:** Nunca use o conteúdo gerado sem uma revisão humana minuciosa para verificar a precisão factual, a validade dos distratores e o alinhamento pedagógico.

## Use Cases
*   **Educação Formal:** Criação rápida de testes, quizzes e exames para escolas, universidades e cursos técnicos.
*   **Treinamento Corporativo:** Desenvolvimento de avaliações de proficiência e questionários de feedback para módulos de treinamento e desenvolvimento de funcionários.
*   **Certificações:** Geração de bancos de questões para exames de certificação profissional em diversas áreas (TI, Finanças, Saúde).
*   **Autoavaliação:** Criação de testes práticos para estudantes que desejam simular exames e testar seus conhecimentos.
*   **Pesquisa de Mercado/Opinião:** Elaboração de questionários e enquetes estruturadas para coleta de dados.

## Pitfalls
*   **Vagueza na Instrução:** Solicitar apenas "Crie um teste sobre X" resulta em questões superficiais que testam apenas a memorização.
*   **Alucinação Factual:** A IA pode gerar questões ou respostas incorretas, exigindo verificação humana obrigatória.
*   **Questões de Baixa Ordem:** Tendência da IA em gerar questões que se concentram nos níveis mais baixos da Taxonomia de Bloom (lembrar, entender), a menos que explicitamente instruída a ir além.
*   **Viés Involuntário:** O conteúdo gerado pode refletir vieses presentes nos dados de treinamento da IA, o que pode afetar a justiça e a validade da avaliação.
*   **Falta de Contexto Específico:** Sem o material de origem (texto, aula), a IA pode criar questões genéricas ou irrelevantes para o conteúdo exato ensinado.

## URL
[https://cetli.upenn.edu/resources/generative-ai/using-ai-to-create-assessments/](https://cetli.upenn.edu/resources/generative-ai/using-ai-to-create-assessments/)
