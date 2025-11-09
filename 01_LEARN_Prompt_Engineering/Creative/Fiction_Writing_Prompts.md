# Fiction Writing Prompts

## Description
**Prompts de Escrita de Ficção** (Fiction Writing Prompts) é uma categoria de engenharia de prompt focada em utilizar Modelos de Linguagem Grande (LLMs) para auxiliar em tarefas de escrita criativa, como a geração de ideias, desenvolvimento de personagens, construção de mundos (worldbuilding), criação de diálogos e rascunhos de cenas ou capítulos. A técnica vai além de um simples pedido, exigindo que o usuário defina parâmetros narrativos complexos (gênero, tom, ponto de vista, arco de personagem, conflito) para guiar a IA a produzir textos coesos, estilisticamente consistentes e criativos. A eficácia reside na capacidade de fornecer à IA o máximo de contexto e restrições para evitar clichês e a "escrita genérica de IA" (AI-slop). As abordagens mais avançadas, como o **Few-Shot Prompting** (fornecer exemplos de escrita desejada) e o **Chain-of-Thought** (pedir à IA para raciocinar sobre a estrutura da história antes de escrever), são cruciais para alcançar resultados de alta qualidade e complexidade narrativa.

## Examples
```
**1. Geração de Cena com Estilo Definido (Few-Shot)**
*   **Prompt:** "Assuma o papel de um autor de ficção policial noir, como Raymond Chandler. O tom deve ser cínico, a prosa concisa e o cenário deve ser uma rua chuvosa de Los Angeles à noite. Escreva a cena de abertura onde o detetive particular, Jack Rourke, encontra uma mulher fatal em seu escritório. O texto deve ter no máximo 200 palavras.
    *   **Exemplo de Estilo (Few-Shot):** 'A chuva batia na janela como um milhão de dedos apressados. O cheiro de café velho e desespero era o perfume da minha sala. Ela entrou, e o mundo parou de girar. Tinha pernas que iam até o ano que vem e olhos que prometiam problemas.'
    *   **Tarefa:** Escreva a cena de abertura."

**2. Desenvolvimento de Personagem com Conflito Interno (CoT)**
*   **Prompt:** "Você é um psicólogo e escritor. O protagonista é um ex-soldado chamado Kael, que sofre de estresse pós-traumático e agora trabalha como jardineiro.
    *   **Passo 1 (CoT):** Descreva o conflito interno de Kael em 3 pontos-chave (ex: culpa pela guerra, medo de espaços fechados, desejo de redenção).
    *   **Passo 2:** Escreva um monólogo interno de Kael enquanto ele poda uma roseira, onde o ato de podar desencadeia uma memória de combate. O monólogo deve usar metáforas de jardinagem para descrever a violência."

**3. Construção de Mundo e Regras de Magia (Constraint-Based)**
*   **Prompt:** "Gênero: Fantasia Sombria. Crie um sistema de magia chamado 'Tecelagem de Sombras'.
    *   **Restrições:** A magia deve ser baseada em emoções negativas (medo, inveja, luto). Cada uso deve ter um custo físico (ex: perda de memória, dor crônica).
    *   **Tarefa:** Descreva a primeira vez que a protagonista, uma jovem órfã, usa a Tecelagem de Sombras para se defender de um guarda. Descreva o custo físico imediato."

**4. Diálogo com Subtexto (Subtext Prompting)**
*   **Prompt:** "Escreva um diálogo de 5 falas entre um pai (Sr. Alistair) e sua filha (Lia) em uma cozinha.
    *   **Contexto:** Eles estão discutindo o futuro de Lia na faculdade, mas o subtexto real é que o pai teme que ela o abandone, e Lia teme decepcioná-lo.
    *   **Instrução:** Nenhuma das falas pode mencionar diretamente 'medo' ou 'abandono'. O subtexto deve ser transmitido através de perguntas sobre logística e planos futuros."

**5. Brainstorming de Plot Twist (Iterative Prompting)**
*   **Prompt:** "Gênero: Thriller Psicológico. O protagonista acabou de descobrir que seu vizinho é um assassino em série.
    *   **Tarefa 1:** Liste 5 reviravoltas (plot twists) possíveis para o final do livro.
    *   **Tarefa 2:** Escolha a reviravolta mais chocante (ex: o protagonista é o assassino, mas sofre de amnésia dissociativa).
    *   **Tarefa 3:** Escreva o parágrafo final do livro revelando essa reviravolta, usando um tom de epifania aterrorizante."
```

## Best Practices
**1. Defina o Papel e o Tom (Role and Tone Setting):** Comece o prompt instruindo a IA a assumir um papel específico (ex: "Você é um romancista de ficção científica sombria") e um tom (ex: "O tom deve ser melancólico e descritivo"). Isso restringe o espaço de geração e melhora a coerência estilística.
**2. Use a Estrutura de Prompt (Context-Task-Constraint-Example):** Forneça contexto (gênero, cenário), a tarefa (o que escrever), restrições (limite de palavras, ponto de vista) e, idealmente, um exemplo de escrita (Few-Shot Prompting) para refinar o estilo.
**3. Detalhes Sensoriais e Emocionais:** Inclua detalhes específicos sobre o que os personagens veem, ouvem, cheiram, tocam e sentem. A IA tende a focar em ações; o escritor deve forçá-la a focar na experiência.
**4. Prompting de Cadeia de Pensamento (Chain-of-Thought - CoT):** Para arcos de história complexos, peça à IA para primeiro delinear a lógica da cena ou do desenvolvimento do personagem antes de escrever o texto final. Ex: "Primeiro, liste 3 maneiras pelas quais este evento afeta a motivação do protagonista. Em seguida, escreva a cena."
**5. Iteração e Refinamento:** Não espere a perfeição no primeiro rascunho. Use prompts de acompanhamento para refinar (ex: "Reescreva o parágrafo anterior, aumentando o suspense e mudando o ponto de vista para o do antagonista").

## Use Cases
**1. Superar o Bloqueio de Escritor (Writer's Block):** Gerar ideias iniciais, primeiros parágrafos ou sinopses de enredo quando o autor está estagnado.
**2. Desenvolvimento de Personagens e Diálogos:** Criar perfis detalhados de personagens, explorar suas motivações internas (usando CoT) ou gerar rascunhos de diálogos para testar a voz de um personagem.
**3. Construção de Mundo (Worldbuilding):** Definir regras de sistemas de magia, culturas, história ou geografia de um mundo fictício, garantindo a consistência interna.
**4. Rascunho Rápido (Drafting):** Gerar rascunhos de cenas ou capítulos inteiros para que o autor possa se concentrar na edição e no refinamento, acelerando o processo de escrita.
**5. Exploração de Gênero e Estilo:** Experimentar diferentes gêneros (ex: *steampunk*, *cyberpunk*, realismo mágico) ou imitar o estilo de autores específicos (Few-Shot Prompting) para encontrar a voz ideal para um projeto.
**6. Revisão e Edição:** Usar a IA para identificar clichês, sugerir alternativas para frases fracas ou reescrever um texto em um tom diferente (ex: de passivo para ativo).

## Pitfalls
**1. A Armadilha da Vaguidão (The Vagueness Trap):** Prompts genéricos como "Escreva uma história de amor" resultam em clichês e falta de originalidade. A IA preenche as lacunas com o que é estatisticamente mais provável.
**2. Sobrecarga de Informação (Information Overload):** Tentar incluir muitos detalhes não essenciais em um único prompt longo pode confundir a IA, diluindo as instruções cruciais. O ideal é usar prompts curtos e iterativos.
**3. Confiança Cega (Blind Trust):** Aceitar o resultado da IA sem revisão crítica. A IA pode introduzir inconsistências de enredo, erros de continuidade ou diálogos que soam "robóticos" (o chamado *AI-slop*).
**4. O Vácuo de Contexto (The Context Vacuum):** Não definir o papel da IA (ex: "Você é um editor", "Você é um escritor de terror") ou o público-alvo. A falta de contexto leva a um tom e estilo inconsistentes.
**5. Falha em Iterar:** Tratar o prompt como uma única interação. A escrita criativa com IA é um processo iterativo. Não usar prompts de acompanhamento para refinar, expandir ou corrigir o texto gerado é um erro comum.

## URL
[https://www.prompthub.us/blog/the-few-shot-prompting-guide](https://www.prompthub.us/blog/the-few-shot-prompting-guide)
