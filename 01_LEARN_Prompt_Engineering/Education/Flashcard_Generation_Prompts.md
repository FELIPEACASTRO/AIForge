# Flashcard Generation Prompts

## Description
**Flashcard Generation Prompts** são instruções de engenharia de prompt projetadas para que Modelos de Linguagem Grande (LLMs) transformem textos brutos, notas de aula, artigos ou conceitos complexos em flashcards estruturados (geralmente no formato Pergunta/Resposta ou Oclusão de Lacunas). A técnica visa automatizar a parte tediosa da criação de flashcards, permitindo que o usuário se concentre na revisão e no aprendizado ativo. A eficácia desses prompts é maximizada quando combinados com princípios de aprendizado baseado em ciência, como **Active Recall** (Recuperação Ativa) e **Spaced Repetition** (Repetição Espaçada).

## Examples
```
1. **Prompt Básico (Pergunta/Resposta Estrita)**
```
Aja como um especialista em aprendizado ativo. Analise o texto a seguir e extraia 10 fatos cruciais. Para cada fato, crie um flashcard no formato estrito:
Pergunta: [A pergunta deve exigir a recuperação de um único fato.]
Resposta: [A resposta deve ser concisa e direta.]
Use o seguinte texto: [COLE O TEXTO AQUI]
```

2. **Prompt de Alta Retenção (Concurso/Prova)**
```
Crie 15 flashcards sobre o tema "[TEMA]", no formato pergunta e resposta. Misture conceitos-chave, definições, e inclua 3 "pegadinhas de prova" (perguntas que testam exceções ou detalhes facilmente confundidos).
Formato de Saída:
### Flashcard [NÚMERO]
Pergunta:
Resposta:
```

3. **Prompt com Taxonomia de Bloom (Nível de Aplicação)**
```
Com base no conceito de "[CONCEITO]", gere 5 flashcards.
- 2 devem ser de nível 'Lembrar' (definição).
- 2 devem ser de nível 'Entender' (explicar em suas próprias palavras).
- 1 deve ser de nível 'Aplicar' (apresentar um cenário e perguntar como o conceito seria usado).
Use o formato Pergunta/Resposta.
```

4. **Prompt para Cloze Deletion (Oclusão de Lacunas)**
```
Transforme o parágrafo a seguir em 5 flashcards no formato Cloze Deletion (oclusão de lacunas), ideal para o Anki. A lacuna deve ser colocada em uma palavra ou frase-chave.
Formato de Saída:
[TEXTO COM LACUNA]
Resposta: [PALAVRA/FRASE REMOVIDA]
Parágrafo: [COLE O PARÁGRAFO AQUI]
```

5. **Prompt com Chain-of-Thought (CoT) para Qualidade**
```
Você é um gerador de flashcards de alta qualidade. Antes de gerar o flashcard, siga o processo de Chain-of-Thought (CoT):
1. Identifique o fato principal no texto.
2. Formule uma pergunta concisa que exija a recuperação desse fato.
3. Forneça a resposta direta.
4. Crie o flashcard final.
Gere 8 flashcards a partir do texto: [COLE O TEXTO AQUI]
Formato de Saída:
Processo CoT: [Seu raciocínio]
Flashcard: Pergunta: [X] | Resposta: [Y]
```

6. **Prompt para Flashcards de Linguagem (Vocabulário)**
```
Crie 10 flashcards de vocabulário em inglês para o nível B2, usando as palavras do texto a seguir. Cada flashcard deve incluir:
1. A palavra em inglês.
2. A definição em português.
3. Uma frase de exemplo em inglês.
Texto: [COLE O TEXTO EM INGLÊS AQUI]
```

7. **Prompt para Flashcards de Programação (Conceito/Sintaxe)**
```
Gere 7 flashcards sobre o conceito de "Programação Orientada a Objetos (POO)" em Python. Inclua:
- 3 cartões de definição (Classes, Objetos, Herança).
- 2 cartões de sintaxe (Pergunta: Como você define uma classe em Python? Resposta: [CÓDIGO DE EXEMPLO]).
- 2 cartões de caso de uso (Pergunta: Dê um exemplo de polimorfismo).
```

8. **Prompt de Comparação (Análise)**
```
Crie um flashcard de comparação entre "[CONCEITO A]" e "[CONCEITO B]".
Pergunta: Quais são as 3 principais diferenças entre [CONCEITO A] e [CONCEITO B]?
Resposta: [Lista de 3 diferenças concisas].
Use o texto de referência: [COLE O TEXTO AQUI]
```
```

## Best Practices
1. **Princípio de Um Fato por Cartão:** O prompt deve instruir o LLM a criar flashcards que abordem apenas um único fato, conceito ou habilidade por cartão. Isso reduz a carga cognitiva e melhora a retenção.
2. **Fraseado Ativo e Contextual:** O prompt deve exigir que as perguntas sejam formuladas de maneira ativa (exigindo geração, não reconhecimento) e que forneçam contexto suficiente para serem não ambíguas.
3. **Estrutura de Saída Clara:** Defina um formato de saída estrito (ex: `Pergunta: [X] | Resposta: [Y]`) ou um formato de tabela/CSV para facilitar a importação para sistemas de repetição espaçada (como Anki ou Quizlet).
4. **Mapeamento para a Taxonomia de Bloom:** Peça ao LLM para criar cartões que cubram diferentes níveis cognitivos (Lembrar, Entender, Aplicar, Analisar). Comece com cartões de "Lembrar" e avance para cartões de "Aplicar" ou "Analisar" para aprofundar o domínio.
5. **Uso de Técnicas Avançadas (CoT e Few-Shot):** Para resultados de maior qualidade, incorpore **Chain-of-Thought (CoT)** e **Few-Shot Prompting** (fornecendo 1-2 exemplos de flashcards ideais) para guiar o LLM na extração precisa e concisa de fatos.
6. **Controle de Qualidade Humano:** Sempre revise manualmente uma amostra dos cartões gerados (10-20%) para verificar a precisão, concisão e ambiguidade. A personalização e a edição humana ativam o **Generation Effect**, melhorando a memorização.

## Use Cases
*   **Estudos Acadêmicos:** Transformar notas de aula, resumos de livros didáticos ou artigos científicos em conjuntos de flashcards para exames.
*   **Aprendizado de Idiomas:** Criar cartões de vocabulário, conjugação verbal e frases contextuais.
*   **Preparação para Concursos/Certificações:** Gerar cartões focados em conceitos-chave, "pegadinhas" de prova e jurisprudência.
*   **Aprendizado de Programação/Tópicos Técnicos:** Criar cartões para sintaxe de linguagem, algoritmos, comandos de terminal ou conceitos de arquitetura de software.
*   **Medicina e Ciências:** Gerar cartões de anatomia (com oclusão de imagem), farmacologia (mecanismo vs. uso clínico) e vias bioquímicas.

## Pitfalls
1. **Cartões Longos ou Ambíguos:** O LLM pode gerar cartões com perguntas ou respostas muito longas, violando o princípio de "um fato por cartão", ou perguntas que podem ter múltiplas respostas corretas.
2. **Dependência Excessiva:** Confiar cegamente no conteúdo gerado pela IA sem revisão manual pode levar à memorização de informações incorretas ou mal formuladas, perdendo o benefício do **Generation Effect**.
3. **Foco Apenas em Fatos:** Gerar apenas cartões de nível "Lembrar" (definições, datas) e negligenciar cartões de nível superior (Aplicação, Análise), resultando em memorização superficial.
4. **Formato Incompatível:** O LLM pode não aderir ao formato de saída solicitado, dificultando a importação automatizada para o software de repetição espaçada.
5. **Sobrecarga de Cartões:** Gerar um número excessivo de cartões (mais de 20-30 novos por dia) pode levar ao esgotamento e à incapacidade de manter a rotina de revisão espaçada.

## URL
[https://blog.educate-ai.com/en/flashcards-creation-modern-methods-tips-tools-for-effective-learning](https://blog.educate-ai.com/en/flashcards-creation-modern-methods-tips-tools-for-effective-learning)
