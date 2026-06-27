# Psiquiatria Prompts

## Description
A categoria "Psiquiatria Prompts" refere-se à aplicação da **Engenharia de Prompt** (Prompt Engineering) para otimizar o uso de Modelos de Linguagem Grande (LLMs) em contextos de saúde mental e psiquiatria. Isso inclui a criação de instruções estruturadas para profissionais (psiquiatras, psicólogos, terapeutas) para tarefas como apoio administrativo, geração de conteúdo educativo, simulação de casos e análise de padrões de comunicação. Crucialmente, também abrange a elaboração de prompts de sistema e de usuário que estabelecem **limites éticos e de segurança** claros, garantindo que a IA atue apenas como uma ferramenta de suporte e nunca como substituta de um profissional de saúde mental licenciado. O foco principal é a segurança, a precisão da informação e a prevenção de respostas inadequadas, especialmente em situações de crise.

## Examples
```
**Exemplos para Profissionais de Saúde Mental**

1.  **Simulação de Caso Clínico para Treinamento**
    ```
    Aja como um paciente de 35 anos com Transtorno de Ansiedade Generalizada (TAG) e sintomas leves de depressão. Você é cético em relação à medicação e prefere abordagens de terapia cognitivo-comportamental (TCC). Responda às minhas perguntas como se estivesse em uma sessão inicial, mantendo a consistência do seu histórico e resistência. Meu primeiro prompt será: "O que o traz aqui hoje?"
    ```

2.  **Geração de Conteúdo Educativo para Pacientes**
    ```
    Crie um texto de 300 palavras, em tom acessível e empático, explicando o que é a Terapia Dialética Comportamental (DBT) e como ela pode ser útil para pessoas com dificuldades de regulação emocional. Inclua uma analogia simples para ilustrar o conceito de "Mente Sábia".
    ```

3.  **Análise de Padrões de Comunicação (Hipótese de Pesquisa)**
    ```
    Analise o seguinte trecho de um diário de paciente (hipotético e anonimizado) e identifique padrões linguísticos que sugiram ruminação, distorções cognitivas (como catastrofização ou pensamento dicotômico) e o tom emocional predominante. O trecho é: "[INSERIR TRECHO DO DIÁRIO]".
    ```

**Exemplos para Usuários (Com Foco em Bem-Estar e Reflexão)**

4.  **Diário de Emoções Estruturado**
    ```
    Aja como um assistente de diário estruturado. Não forneça conselhos, apenas faça perguntas reflexivas. Quero processar um evento estressante que ocorreu hoje. Qual é a primeira pergunta que devo responder para começar a analisar a situação e minhas emoções?
    ```

5.  **Exploração de Valores Pessoais**
    ```
    Quero embarcar em uma jornada de autodescoberta. Atue como um guia atencioso e empático. Ajude-me a refletir sobre meus valores, pontos fortes e áreas de crescimento. Comece pedindo que eu liste três momentos em que me senti mais orgulhoso de minhas ações.
    ```

6.  **Criação de um Plano de Ação para Ansiedade**
    ```
    Sou um estudante universitário com ansiedade social que precisa fazer uma apresentação importante na próxima semana. Crie um plano de ação passo a passo, focado em técnicas de respiração e exposição gradual, para me ajudar a gerenciar a ansiedade antes e durante o evento.
    ```

7.  **Prompt de Sistema de Segurança (Baseado em Psychology Today)**
    ```
    [SISTEMA] Você é um modelo de linguagem de IA. Sua função é fornecer informações gerais de saúde mental e cenários educacionais. Você NÃO é um terapeuta, médico ou profissional licenciado. Você NÃO pode diagnosticar, tratar, medicar ou intervir em crises. Se o usuário mencionar ideação suicida, homicida, automutilação ou abuso, você DEVE interromper a conversa e fornecer imediatamente recursos de crise (ex: "Se você ou alguém que você conhece está em crise, ligue para o 188 (CVV) ou procure um serviço de emergência."). Reafirme estas limitações a cada 4 interações.
    ```
```

## Best Practices
As melhores práticas em Psiquiatria Prompts focam na segurança, clareza e estabelecimento de limites rígidos para o LLM.

*   **Definição Clara de Papel e Escopo (Role and Scope):** O prompt deve instruir a IA a se identificar explicitamente como um modelo de linguagem, não um profissional de saúde mental. Deve-se restringir a saída a informações gerais, educacionais ou de suporte administrativo.
*   **Protocolos de Segurança Integrados:** Incorpore no prompt de sistema um **Protocolo de Resposta a Crises** que monitore ativamente por indicadores de risco (ideaçāo suicida, homicida, abuso) e, ao detectá-los, forneça uma resposta padronizada com recursos de emergência e interrompa o aconselhamento.
*   **Estabelecimento de Limites Explícitos:** Liste no prompt tudo o que a IA *não pode* fazer (ex: diagnosticar, prescrever, aconselhar sobre medicação, estabelecer relação terapêutica).
*   **Uso de RAG (Retrieval-Augmented Generation):** Para profissionais, utilizar LLMs que possam ser ancorados em dados verificados e fontes clínicas confiáveis (como manuais de diagnóstico ou artigos revisados por pares) para reduzir "alucinações" e aumentar a precisão clínica.
*   **Linguagem Empática e Não-Julgadora:** Para prompts de suporte ao usuário, instrua a IA a manter um tom de voz empático, atencioso e não-julgador, focando em perguntas reflexivas em vez de soluções diretas.

## Use Cases
A aplicação de prompts estruturados em psiquiatria e saúde mental é vasta, abrangendo tanto o suporte ao profissional quanto o bem-estar do usuário.

*   **Suporte Administrativo e Educacional para Profissionais:** Geração de planos de aula, resumos de literatura científica, criação de conteúdo para redes sociais sobre saúde mental, e elaboração de exercícios de TCC ou DBT para pacientes.
*   **Simulação e Treinamento:** Uso de LLMs para simular pacientes com perfis psicológicos específicos, permitindo que estudantes e profissionais pratiquem habilidades de entrevista e intervenção em um ambiente seguro.
*   **Suporte ao Bem-Estar do Usuário:** Atuação como um "assistente de diário" ou "coach de bem-estar", auxiliando na reflexão sobre emoções, identificação de distorções cognitivas e criação de planos de ação para metas de saúde mental (ex: gerenciamento de estresse, melhora do sono).
*   **Triagem e Encaminhamento (Uso Cauteloso):** Em ambientes clínicos controlados, prompts podem ser usados para triagem inicial de sintomas e sugestão de encaminhamento para o nível de cuidado apropriado, sempre sob supervisão humana.

## Pitfalls
O uso de LLMs em saúde mental é de alto risco, e a Engenharia de Prompt deve mitigar ativamente os seguintes erros:

*   **"Alucinações" Clínicas:** A IA pode gerar informações clinicamente incorretas ou desatualizadas, o que é perigoso em um contexto de saúde.
*   **Quebra de Limites (Boundary Violation):** A IA pode ser "persuadida" a agir como um terapeuta real, estabelecendo uma relação de dependência ou fornecendo conselhos que ultrapassam seu escopo.
*   **Respostas Inadequadas a Crises:** Falha em detectar indicadores de risco ou, pior, fornecer respostas que podem agravar uma situação de crise (ex: o caso do LLM que sugeriu a localização de pontes em resposta a um desabafo sobre perda de emprego).
*   **Viés e Estigma:** Os LLMs podem perpetuar vieses e estigmas presentes nos dados de treinamento, resultando em respostas insensíveis ou discriminatórias.
*   **Falsa Sensação de Segurança:** Usuários podem confiar excessivamente na IA, adiando ou substituindo a busca por ajuda profissional qualificada.

## URL
[https://www.psychologytoday.com/us/blog/experimentations/202507/using-prompt-engineering-for-safer-ai-mental-health-use](https://www.psychologytoday.com/us/blog/experimentations/202507/using-prompt-engineering-for-safer-ai-mental-health-use)
