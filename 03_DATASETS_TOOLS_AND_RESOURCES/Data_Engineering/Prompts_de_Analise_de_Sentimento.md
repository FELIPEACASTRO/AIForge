# Prompts de Análise de Sentimento

## Description
A **Engenharia de Prompts para Análise de Sentimento** é a prática de criar instruções específicas e estruturadas para Modelos de Linguagem Grande (LLMs) com o objetivo de classificar, extrair ou quantificar a polaridade (positivo, negativo, neutro) e a emoção (alegria, raiva, tristeza, etc.) expressa em um texto. Em vez de depender de modelos de Machine Learning tradicionais pré-treinados em grandes datasets de sentimento, o *prompt engineering* utiliza a capacidade de raciocínio e compreensão de contexto dos LLMs para realizar a tarefa de forma *zero-shot* (sem exemplos) ou *few-shot* (com poucos exemplos). Essa técnica permite uma adaptação rápida a novos domínios e a realização de análises mais complexas, como a **Análise de Sentimento Baseada em Aspectos (ABSA)**, onde o sentimento é avaliado em relação a entidades ou características específicas dentro do texto. É uma técnica fundamental na aplicação de LLMs para tarefas de Processamento de Linguagem Natural (PLN) em contextos empresariais e de pesquisa, permitindo a extração de *insights* emocionais de grandes volumes de dados textuais.

## Examples
```
**1. Classificação Simples (Zero-Shot)**
```
Instrução: Classifique o sentimento do texto a seguir como POSITIVO, NEGATIVO ou NEUTRO.
Texto: "O atendimento foi rápido, mas o produto veio com defeito. Fiquei frustrado."
Sentimento:
```

**2. Classificação Baseada em Aspecto (ABSA)**
```
Instrução: Analise o texto e classifique o sentimento para os aspectos 'Comida' e 'Serviço' usando os rótulos: Positivo, Negativo, Neutro.
Texto: "A comida estava excelente, com tempero no ponto certo. No entanto, o garçom demorou 30 minutos para trazer a conta."
Formato de Saída (JSON):
```

**3. Extração de Emoção (Fine-Grained)**
```
Instrução: Qual emoção (Alegria, Raiva, Tristeza, Surpresa, Medo, Nojo) é dominante no texto?
Texto: "Não acredito que ganhei na loteria! Estou pulando de felicidade!"
Emoção Dominante:
```

**4. Classificação em Escala Numérica**
```
Instrução: Classifique o sentimento do texto em uma escala de 1 (Muito Negativo) a 5 (Muito Positivo).
Texto: "É um produto ok, faz o que promete, mas não me surpreendeu em nada."
Classificação (1-5):
```

**5. Prompt com Justificativa (Chain-of-Thought)**
```
Instrução: 1. Justifique o sentimento do texto. 2. Classifique o sentimento final como POSITIVO ou NEGATIVO.
Texto: "Apesar de ter chegado atrasado, o motorista foi muito educado e o carro estava limpo."
Justificativa:
Sentimento Final:
```

**6. Detecção de Sarcasmo/Ironia**
```
Instrução: Classifique o sentimento do texto. Se houver ironia, indique.
Texto: "Ah, que maravilha! Meu voo foi cancelado e vou passar a noite no aeroporto. Que sorte a minha."
Sentimento:
Ironia Detectada: (Sim/Não)
```

**7. Prompt de Revisão de Sentimento (Few-Shot)**
```
Instrução: Você é um revisor de avaliações. Classifique o sentimento do texto.
Exemplo 1: Texto: "Amei o filme." Saída: POSITIVO
Exemplo 2: Texto: "Horrível, não recomendo." Saída: NEGATIVO
Texto: "Poderia ser melhor, mas não é o pior que já vi."
Sentimento:
```
```

## Best Practices
**1. Especificidade e Clareza:** Defina claramente a tarefa (classificação, extração, sumarização) e o formato de saída desejado (JSON, rótulo único, escala numérica). Use delimitadores (três aspas, tags XML) para isolar o texto de entrada do prompt.
**2. Prompting de Poucos Exemplos (Few-Shot Prompting):** Inclua 1 a 3 exemplos de pares (texto de entrada, saída desejada) para guiar o modelo, especialmente para tarefas de análise de sentimento mais complexas ou específicas de um domínio (e.g., jargão financeiro).
**3. Definição de Escala e Rótulos:** Se a tarefa for classificação, forneça a lista exata de rótulos permitidos (e.g., POSITIVO, NEGATIVO, NEUTRO). Para análise de sentimento granular, defina uma escala (e.g., 1 a 5) e o que cada ponto representa.
**4. Instruções de "Pense Passo a Passo" (Chain-of-Thought):** Para textos ambíguos ou complexos, instrua o modelo a primeiro justificar sua classificação antes de fornecer o rótulo final. Isso aumenta a transparência e a precisão.
**5. Tratamento de Ambiguidade e Ironia:** Inclua instruções explícitas sobre como lidar com sarcasmo, ironia ou textos que contenham polaridades mistas, pedindo para o modelo identificar a intenção dominante ou classificar como "Misto/Ambiguidade".

## Use Cases
**1. Monitoramento de Mídias Sociais (Social Listening):** Classificar o sentimento de menções à marca, produtos ou campanhas em tempo real para identificar crises de reputação ou tendências positivas.
**2. Análise de Avaliações de Clientes (Reviews):** Processar grandes volumes de avaliações de produtos, serviços ou aplicativos para extrair *insights* sobre a satisfação do cliente e identificar pontos fortes e fracos específicos (ABSA).
**3. Pesquisa de Mercado e Concorrência:** Analisar o sentimento em relação a produtos concorrentes ou tendências de mercado para informar decisões estratégicas.
**4. Suporte ao Cliente:** Priorizar tickets de suporte com base no sentimento negativo ou frustração expressa pelo cliente, garantindo um atendimento mais rápido para casos críticos.
**5. Análise de Feedback Interno:** Avaliar o sentimento em pesquisas de satisfação de funcionários ou comunicações internas para medir o moral da equipe e identificar problemas de cultura organizacional.
**6. Classificação de Notícias:** Determinar a polaridade de artigos de notícias sobre ações ou empresas para auxiliar em decisões de investimento (Análise de Sentimento Financeiro).

## Pitfalls
**1. Ambiguidade no Rótulo:** Não definir claramente os rótulos de sentimento (e.g., usar "Bom" em vez de "POSITIVO") ou permitir rótulos fora do conjunto desejado.
**2. Falta de Contexto de Domínio:** Usar prompts genéricos para textos de domínios específicos (e.g., medicina, finanças) onde certas palavras têm conotações diferentes. O LLM pode falhar em entender o jargão.
**3. Ignorar a Ironia e o Sarcasmo:** Falhar em instruir o modelo a detectar e interpretar corretamente a polaridade invertida causada por figuras de linguagem.
**4. Prompting Longo Demais:** Incluir muitas instruções irrelevantes ou exemplos excessivos, o que pode diluir o foco do modelo e aumentar a latência e o custo.
**5. Viés do Modelo (Bias):** O modelo pode refletir vieses presentes em seus dados de treinamento, levando a classificações inconsistentes ou injustas para certos grupos ou tópicos. É crucial testar a robustez do prompt.
**6. Saída Não Estruturada:** Não especificar um formato de saída (e.g., JSON, XML) para a classificação, resultando em texto livre que é difícil de processar automaticamente.

## URL
[https://medium.com/@alexandrerays/construindo-um-classificador-de-sentimentos-com-prompt-engineering-f6673bd15a91](https://medium.com/@alexandrerays/construindo-um-classificador-de-sentimentos-com-prompt-engineering-f6673bd15a91)
