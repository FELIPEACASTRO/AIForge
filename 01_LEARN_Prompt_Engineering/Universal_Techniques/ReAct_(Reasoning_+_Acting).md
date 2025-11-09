# ReAct (Reasoning + Acting)

## Description
**ReAct (Reasoning + Acting)** é um paradigma de *prompting* que combina o raciocínio em cadeia de pensamento (*Chain-of-Thought* - CoT) com a capacidade de interagir com ferramentas externas (*Acting*). O modelo de linguagem grande (LLM) é solicitado a gerar traços de raciocínio verbais (`Thought`) e ações específicas da tarefa (`Action`) de forma intercalada. O raciocínio ajuda o modelo a induzir, rastrear e atualizar planos de ação, bem como a lidar com exceções, enquanto as ações permitem que ele interaja com fontes externas (como APIs de busca ou bases de conhecimento) para coletar informações adicionais. Isso cria um ciclo iterativo de **Pensamento -> Ação -> Observação -> Pensamento**, que melhora a interpretabilidade, a fidelidade factual (reduzindo alucinações) e a capacidade de resolver tarefas complexas de tomada de decisão e raciocínio [1] [2]. O ReAct é a base para a construção de **Agentes de IA** capazes de planejar e executar tarefas em ambientes dinâmicos.

## Examples
```
**Exemplo 1: Resposta a Perguntas com Busca na Web**

```
Question: Quem foi o primeiro ser humano a pisar na lua e qual é a sua data de nascimento?

Thought: Preciso encontrar o nome do primeiro ser humano a pisar na lua.
Action: Search[primeiro ser humano a pisar na lua]
Observation: O primeiro ser humano a pisar na lua foi Neil Armstrong.
Thought: Agora que tenho o nome, preciso encontrar a data de nascimento de Neil Armstrong.
Action: Search[data de nascimento de Neil Armstrong]
Observation: Neil Armstrong nasceu em 5 de agosto de 1930.
Thought: Tenho todas as informações necessárias para responder à pergunta.
Final Answer: O primeiro ser humano a pisar na lua foi Neil Armstrong, nascido em 5 de agosto de 1930.
```

**Exemplo 2: Resolução de Problemas Matemáticos Complexos**

```
Question: Qual é o resultado de (15 * 3) + (45 / 5)?

Thought: Preciso primeiro calcular a multiplicação (15 * 3) e a divisão (45 / 5) e depois somar os resultados.
Action: Calculator[15 * 3]
Observation: 45
Thought: O resultado da multiplicação é 45. Agora, calcularei a divisão.
Action: Calculator[45 / 5]
Observation: 9
Thought: O resultado da divisão é 9. Agora, somarei os dois resultados.
Action: Calculator[45 + 9]
Observation: 54
Thought: O resultado final é 54.
Final Answer: 54
```

**Exemplo 3: Verificação de Fatos (Fact Verification)**

```
Question: A capital da Austrália é Sydney?

Thought: Preciso verificar qual é a capital oficial da Austrália.
Action: Search[capital da Austrália]
Observation: A capital da Austrália é Canberra.
Thought: A informação inicial está incorreta. A capital é Canberra, não Sydney.
Final Answer: Não, a capital da Austrália é Canberra.
```

**Exemplo 4: Planejamento de Viagem (com Ferramenta de Previsão do Tempo)**

```
Question: Qual é a previsão do tempo para Paris amanhã e devo levar um guarda-chuva?

Thought: Preciso obter a previsão do tempo para Paris para amanhã.
Action: Weather_API[Paris, amanhã]
Observation: Previsão: 15°C, Chuva leve.
Thought: A previsão indica chuva leve. Devo recomendar um guarda-chuva.
Final Answer: A previsão do tempo para Paris amanhã é de 15°C com chuva leve. Sim, você deve levar um guarda-chuva.
```

**Exemplo 5: Interação com Base de Dados (Database Query)**

```
Question: Quantos clientes na base de dados têm mais de 60 anos?

Thought: Preciso executar uma consulta SQL para contar os clientes com idade superior a 60.
Action: SQL_Query[SELECT COUNT(*) FROM clientes WHERE idade > 60]
Observation: 1245
Thought: A consulta retornou 1245 clientes.
Final Answer: Existem 1245 clientes com mais de 60 anos na base de dados.
```

**Exemplo 6: Resolução de Problemas de Lógica (com Ferramenta de Busca)**

```
Question: Se o preço do Bitcoin subir 10% e o Ethereum subir 5%, qual será o valor total de um portfólio com 1 BTC (R$ 300.000) e 10 ETH (R$ 15.000 cada)?

Thought: Preciso calcular o novo valor do Bitcoin e do Ethereum.
Action: Calculator[300000 * 1.10]
Observation: 330000
Thought: O novo valor do Bitcoin é R$ 330.000. Agora, o novo valor do Ethereum.
Action: Calculator[15000 * 1.05]
Observation: 15750
Thought: O novo valor de 1 ETH é R$ 15.750. O portfólio tem 10 ETH.
Action: Calculator[330000 + (10 * 15750)]
Observation: 487500
Thought: O valor total do portfólio será R$ 487.500.
Final Answer: O valor total do portfólio será de R$ 487.500.
```
```

## Best Practices
**1. Defina as Ferramentas (Tools):** O prompt deve listar claramente as ferramentas externas disponíveis para o modelo (ex: `Search`, `Calculator`, `Wikipedia`). **2. Estrutura o Ciclo ReAct:** Use um formato de prompt de poucas tentativas (few-shot) que demonstre o ciclo `Thought`, `Action`, `Observation` e `Final Answer`. **3. Seja Explícito no Raciocínio (`Thought`):** O modelo deve ser instruído a articular seu raciocínio antes de cada ação, explicando por que a ação é necessária e como ela se relaciona com o objetivo final. **4. Use Observações para Iteração:** A saída da ferramenta (`Observation`) deve ser usada como entrada para o próximo `Thought`, permitindo que o modelo corrija erros, colete mais informações ou mude de estratégia. **5. Use um Token de Parada (`Final Answer`):** O modelo deve ser instruído a usar um token de parada claro (ex: `Final Answer:`) para indicar que a tarefa foi concluída e a resposta final está pronta.

## Use Cases
**1. Agentes de IA:** Construção de agentes autônomos capazes de planejar, executar e monitorar tarefas complexas em ambientes digitais (ex: navegação na web, automação de software). **2. Resposta a Perguntas com Fundamentação (Grounded QA):** Responder a perguntas factuais usando fontes externas (ex: Wikipedia, bases de dados) para garantir que a resposta seja precisa e livre de alucinações. **3. Verificação de Fatos (Fact Checking):** Automatizar o processo de verificação de alegações, usando ferramentas de busca para cruzar informações e validar a veracidade de uma declaração. **4. Resolução de Problemas de Raciocínio:** Solucionar problemas que exigem múltiplas etapas lógicas e o uso de ferramentas específicas (ex: matemática, programação, lógica). **5. Interação com APIs e Sistemas:** Permitir que o LLM interaja com sistemas externos (ex: APIs de clima, bases de dados SQL, ferramentas de e-commerce) para realizar ações no mundo real. **6. Jogos e Ambientes de Decisão:** Em ambientes como ALFWorld e WebShop, o ReAct permite que o agente tome decisões sequenciais e interaja com o ambiente para alcançar um objetivo (ex: comprar um produto online).

## Pitfalls
**1. Definição Inadequada de Ferramentas:** Não definir claramente as ferramentas disponíveis ou as suas assinaturas (como chamar a ferramenta e o que esperar de retorno) pode levar o modelo a gerar ações inválidas ou ineficazes. **2. Alucinação de Ações:** O modelo pode "alucinar" ferramentas que não existem ou ações que não são suportadas pelo ambiente, resultando em falhas de execução. **3. Raciocínio Insuficiente:** Um `Thought` muito breve ou genérico pode não guiar o modelo de forma eficaz, levando a um ciclo de Ação-Observação ineficiente ou a um desvio do objetivo principal. **4. Dependência Excessiva de Ferramentas:** Confiar em ferramentas para cada etapa, mesmo para cálculos simples ou informações que o LLM já possui, pode aumentar a latência e o custo. **5. Erro de Propagação:** Se uma `Observation` inicial for incorreta ou ambígua, o raciocínio subsequente pode se basear nessa informação falha, propagando o erro até a `Final Answer`. **6. Limitação de Contexto:** Em tarefas complexas com muitos ciclos de ReAct, o histórico de `Thought`/`Action`/`Observation` pode exceder a janela de contexto do LLM, fazendo com que ele "esqueça" o raciocínio anterior ou o objetivo.

## URL
[https://react-lm.github.io/](https://react-lm.github.io/)
