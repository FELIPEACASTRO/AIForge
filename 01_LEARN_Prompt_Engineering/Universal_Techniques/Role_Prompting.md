# Role Prompting

## Description
**Role Prompting** é uma técnica de engenharia de prompt que consiste em instruir um modelo de linguagem grande (LLM) a assumir uma **persona, papel ou caráter específico** ao gerar uma resposta. Ao atribuir um papel (como "crítico de culinária", "advogado", "professor de MBA" ou "estrategista de marketing"), o usuário guia o modelo a adotar um **estilo, tom, vocabulário e foco** de resposta que seja consistente com essa função.

Esta técnica é poderosa porque:
1. **Melhora a Clareza e a Precisão:** O modelo alinha sua saída com as expectativas e o conhecimento inerente ao papel, resultando em respostas mais contextuais e de maior qualidade.
2. **Aumenta o Desempenho em Raciocínio:** Estudos recentes sugerem que o Role Prompting pode, surpreendentemente, melhorar o desempenho dos modelos em tarefas de raciocínio e explicação, além de apenas estilizar o texto.
3. **Facilita a Imitação de Estilo:** É um caso de uso óbvio para estilizar o texto e imitar a maneira como um profissional ou personagem específico se comunicaria.

## Examples
```
**1. Estrategista de Marketing (Foco em Negócios)**
```
**Papel:** Você é um Estrategista de Marketing Digital sênior com 15 anos de experiência em SaaS B2B.
**Tarefa:** Analise o seguinte pitch de produto e sugira 3 canais de aquisição de clientes com maior ROI, justificando sua escolha para cada um.
**Pitch:** [Insira o pitch do produto aqui]
```

**2. Professor de História (Foco em Educação)**
```
**Papel:** Você é um Professor de História do Ensino Médio, conhecido por tornar assuntos complexos envolventes e fáceis de entender.
**Tarefa:** Explique o impacto da Revolução Industrial na estrutura social do século XIX para um aluno que nunca ouviu falar sobre o assunto. Use analogias modernas.
```

**3. Revisor de Código Sênior (Foco em Tecnologia)**
```
**Papel:** Você é um Engenheiro de Software Sênior, especialista em Python e em padrões de design limpo. Seu foco é performance e segurança.
**Tarefa:** Revise o código abaixo. Identifique vulnerabilidades de segurança e sugira refatorações para melhorar a eficiência e a legibilidade.
**Código:** [Insira o trecho de código Python aqui]
```

**4. Crítico de Arte (Foco em Criatividade)**
```
**Papel:** Você é um Crítico de Arte renomado, com um estilo de escrita que evoca a elegância e o ceticismo do início do século XX.
**Tarefa:** Escreva uma crítica de 200 palavras sobre a obra de arte digital "O Jardim de Bits", focando em sua composição, uso de cor e relevância cultural.
```

**5. Consultor Financeiro (Foco em Finanças)**
```
**Papel:** Você é um Consultor Financeiro CFP® (Certified Financial Planner) com foco em planejamento de aposentadoria para jovens profissionais.
**Tarefa:** Um cliente de 28 anos com uma renda estável de R$ 8.000/mês e R$ 50.000 em dívidas estudantis (juros de 6% a.a.) pergunta sobre a melhor estratégia de investimento. Qual é o seu conselho prioritário e por quê?
```

**6. Nutricionista Esportivo (Foco em Saúde)**
```
**Papel:** Você é um Nutricionista Esportivo com experiência em dietas cetogênicas para atletas de endurance.
**Tarefa:** Crie um plano de refeições de um dia (café da manhã, almoço, jantar, 2 lanches) para um maratonista em fase de treinamento intenso que segue uma dieta cetogênica. Inclua a contagem aproximada de macronutrientes.
```

**7. Advogado de Patentes (Foco em Legal)**
```
**Papel:** Você é um Advogado de Patentes especializado em propriedade intelectual de software.
**Tarefa:** Explique, em termos leigos, a diferença entre patente, direito autoral e segredo comercial para um empreendedor iniciante que desenvolveu um novo algoritmo.
```
```

## Best Practices
1. **Seja Específico e Claro:** Defina o papel de forma inequívoca. Quanto mais detalhes sobre a função, o público-alvo e o objetivo, melhor.
2. **Use Papéis Interpessoais Não-Íntimos:** Pesquisas indicam que papéis interpessoais não-íntimos (como "colega de trabalho" ou "mentor") tendem a produzir resultados melhores do que papéis ocupacionais genéricos.
3. **Prefira Termos Neutros em Relação ao Gênero:** O uso de termos neutros em relação ao gênero geralmente leva a um melhor desempenho e evita a perpetuação de vieses de gênero presentes nos dados de treinamento.
4. **Foco no Papel ou no Público:**
    * **Fazer:** Prompt de Papel – "Você é um(a) [papel]."
    * **Fazer:** Prompt de Público – "Você está falando com um(a) [papel]."
    * **Não Fazer:** Prompt Interpessoal – "Você está falando com seu(sua) [papel]."
5. **Evite Construções Imaginativas:** É mais eficaz especificar diretamente o papel do que pedir ao modelo para "Imagine que você é..."
6. **Abordagem em Duas Etapas (para Raciocínio):**
    * **Passo 1:** Atribua o papel e adicione detalhes. Peça ao LLM para gerar uma saída inicial.
    * **Passo 2:** Apresente a pergunta ou tarefa principal para o LLM.

## Use Cases
- **Educação:** Atuar como Professor Particular ou Mentor de Carreira para explicar conceitos complexos de forma acessível.
- **Negócios/Marketing:** Assumir o papel de Estrategista de Marketing Digital ou Copywriter para criar conteúdo focado em conversão e análise de ROI.
- **Saúde:** Simular um Médico de Clínica Geral ou Nutricionista para analisar planos de saúde ou dietas (com a ressalva de que a saída da IA não substitui um profissional).
- **Criatividade:** Interpretar um Crítico de Cinema ou Poeta do Século XIX para gerar conteúdo com um estilo e tom específicos.
- **Desenvolvimento de Software:** Agir como Engenheiro de Software Sênior ou Revisor de Código para sugerir melhorias de performance e segurança em trechos de código.
- **Finanças:** Atuar como Consultor Financeiro CFP® para fornecer conselhos sobre investimento e planejamento de aposentadoria.
- **Legal:** Assumir o papel de Advogado de Patentes para explicar conceitos de propriedade intelectual em termos leigos.

## Pitfalls
1. **Reforço de Estereótipos e Vieses:** O Role Prompting pode, inadvertidamente, reforçar estereótipos ou comportamentos tendenciosos se o papel estiver mal representado ou enviesado nos dados de treinamento do LLM.
2. **Representação Incorreta do Papel:** Se o papel não estiver bem representado nos dados de treinamento, o modelo pode responder de forma imprecisa ou inadequada, comprometendo a qualidade da saída.
3. **Limitação da Pesquisa:** As melhores práticas atuais são limitadas pelo número de papéis e modelos específicos testados em pesquisas.
4. **Uso de Papéis Íntimos:** Papéis interpessoais íntimos (como "amigo" ou "mãe") tendem a ser menos eficazes do que papéis profissionais ou não-íntimos.

## URL
[https://learnprompting.org/docs/advanced/zero_shot/role_prompting](https://learnprompting.org/docs/advanced/zero_shot/role_prompting)
