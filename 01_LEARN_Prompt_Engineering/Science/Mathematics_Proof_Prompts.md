# Mathematics Proof Prompts

## Description
A técnica de **Mathematics Proof Prompts** (Prompts de Prova Matemática) refere-se à engenharia de instruções específicas para modelos de linguagem grande (LLMs) com o objetivo de gerar, verificar ou refutar **provas matemáticas formais e rigorosas**. Diferentemente de prompts que buscam apenas a resposta numérica final, esta técnica foca na **cadeia de raciocínio lógico** e na **estrutura formal** da demonstração. O objetivo é mitigar a tendência dos LLMs de cometerem erros lógicos, fazerem suposições injustificadas ou supergeneralizarem padrões observados em casos menores [1]. A eficácia desta técnica está em forçar o modelo a emular o processo de raciocínio de um matemático, exigindo clareza, coerência e o uso de notação formal (como LaTeX) para garantir a precisão da saída. Estudos recentes, como os que avaliam LLMs em problemas de Olimpíadas de Matemática, demonstram que prompts bem estruturados são cruciais para alcançar um desempenho aceitável em tarefas de raciocínio de alto nível [1].

## Examples
```
1. **Geração de Prova Formal (Prompt Principal do Artigo):**
```
Give a thorough answer to the following question. Your answer will be graded by human judges based on accuracy, correctness, and your ability to prove the result. You should include all steps of the proof. Do not skip important steps, as this will reduce your grade. It does not suffice to merely state the result. Use LaTeX to format your answer.

**Problema:** Prove que, para todo número natural $n$, a soma dos primeiros $n$ números ímpares é igual a $n^2$.
```

2. **Verificação e Avaliação de Prova (Prompt de Juiz):**
```
# Instruction
You are an expert mathematician that grades solutions of high-school olympiad-level problems. You will be given a mathematical problem, as well as a grading scheme that you should adhere to. Your task is to accurately grade a solution according to that grading scheme.

# Problem and Scheme
## Problema: Prove que $\sqrt{2}$ é irracional.
## Esquema de Avaliação:
- 7 pontos: Prova completa e rigorosa por contradição.
- 4 pontos: Prova com falha lógica menor ou erro de cálculo.
- 2 pontos: Tentativa correta de contradição, mas incompleta.
- 0 pontos: Resposta incorreta ou sem tentativa de prova.

# Solution to Grade
## Solução: [Insira a solução do aluno aqui]
```

3. **Prova por Indução (Passo a Passo Explícito):**
```
Utilize o método de **Prova por Indução** para demonstrar a seguinte proposição. Apresente sua resposta em três seções claras: Base da Indução, Hipótese de Indução e Passo de Indução. Use o formato LaTeX para todas as expressões matemáticas.

**Proposição:** $\sum_{i=1}^{n} i^3 = \left(\frac{n(n+1)}{2}\right)^2$
```

4. **Prova por Contradição (Instrução de Raciocínio CoT):**
```
Você deve provar a afirmação abaixo usando o método de **Prova por Contradição**. Antes de apresentar a prova final, use a técnica Chain-of-Thought (CoT) para detalhar seu raciocínio.

1. **Assunção de Contradição:** Declare a negação da afirmação.
2. **Desenvolvimento Lógico (CoT):** Mostre a sequência de passos lógicos que levam a uma contradição.
3. **Conclusão Formal:** Declare a conclusão final.

**Afirmação:** Não existe um maior número primo.
```

5. **Geração de Lema e Prova Auxiliar (Few-Shot):**
```
**Instrução:** Dada a afirmação principal, primeiro sugira um lema auxiliar que possa simplificar a prova. Em seguida, prove o lema e use-o para provar a afirmação principal.

**Afirmação Principal:** Se $n$ é um inteiro, então $n^2 + n$ é sempre par.

**Exemplo de Lema (Few-Shot):**
*Lema:* Um inteiro $n$ é par se e somente se $n=2k$ para algum inteiro $k$. Um inteiro $n$ é ímpar se e somente se $n=2k+1$ para algum inteiro $k$.

**Sua Tarefa:**
1. Sugerir Lema Auxiliar.
2. Prova do Lema.
3. Prova da Afirmação Principal usando o Lema.
```

6. **Refutação de Prova (Identificação de Falha Lógica):**
```
Analise a prova apresentada abaixo para a afirmação "Todo triângulo é isósceles". Se a prova estiver incorreta, identifique o **primeiro erro lógico** ou a **suposição inválida** e explique por que ela invalida a prova.

**Prova a ser Refutada:** [Insira aqui a prova falha clássica, como a que usa a intersecção da bissetriz e da mediatriz.]
```

7. **Tradução para Linguagem Formal (Lean/Isabelle):**
```
Traduza a seguinte prova informal para uma sequência de passos que possa ser verificada por um sistema de prova formal (como Lean ou Isabelle). Concentre-se na precisão e na sintaxe lógica.

**Prova Informal:** "A composição de duas funções injetoras é injetora. Sejam $f: A \to B$ e $g: B \to C$ injetoras. Para provar que $g \circ f$ é injetora, assuma que $(g \circ f)(x_1) = (g \circ f)(x_2)$. Isso significa $g(f(x_1)) = g(f(x_2))$. Como $g$ é injetora, temos $f(x_1) = f(x_2)$. Como $f$ é injetora, temos $x_1 = x_2$. Portanto, $g \circ f$ é injetora."
```

8. **Geração de Prova com Restrição de Método:**
```
Gere uma prova para a seguinte afirmação, mas **proíba estritamente** o uso do Teorema Fundamental do Cálculo. A prova deve ser baseada apenas em somas de Riemann e limites.

**Afirmação:** Calcule a integral definida $\int_{0}^{1} x^2 dx$.
```
```

## Best Practices
**1. Exigir Prova Completa e Rigorosa:** Inclua explicitamente no prompt a necessidade de apresentar **todos os passos da prova** e a proibição de pular etapas, garantindo o rigor lógico [1].
**2. Especificar o Formato de Saída:** Exija o uso de linguagens de formatação matemática, como **LaTeX**, para garantir a clareza e a precisão das expressões e símbolos matemáticos [1].
**3. Usar Contexto de Avaliação (Judge Prompt):** Adicione um contexto de sistema (ou no prompt) que simule uma avaliação por um "juiz humano" ou "matemático especialista". Isso eleva o padrão de raciocínio do LLM, incentivando-o a ser mais cauteloso e rigoroso [1].
**4. Aplicar Chain-of-Thought (CoT) Estruturado:** Para problemas complexos, instrua o modelo a detalhar seu raciocínio em etapas lógicas (e.g., "Assunção de Contradição", "Desenvolvimento Lógico", "Conclusão Formal") antes de apresentar a prova final.
**5. Prova Auxiliar (Few-Shot):** Forneça exemplos de provas ou lemas auxiliares (Few-Shot) para guiar o modelo na estratégia de prova desejada (e.g., prova por indução, contradição) [1].
**6. Restrição de Método:** Para testar a flexibilidade e o conhecimento fundamental do modelo, proíba explicitamente o uso de teoremas ou métodos avançados, forçando-o a construir a prova a partir de princípios básicos.

## Use Cases
**1. Geração de Provas Formais:** Criação de soluções rigorosas e detalhadas para problemas de matemática de nível avançado (e.g., Olimpíadas de Matemática, Teoremas Fundamentais) [1].
**2. Avaliação Automatizada de Soluções:** Uso de LLMs como "juízes" (Judge Prompts) para avaliar a correção, o rigor e a clareza de provas submetidas por alunos ou outros modelos [1].
**3. Tradução para Sistemas de Prova Formal:** Conversão de provas informais em linguagem natural para linguagens formais (e.g., Lean, Isabelle), facilitando a verificação por computador.
**4. Pesquisa em Raciocínio Matemático:** Estudo das capacidades e falhas de raciocínio de LLMs, identificando áreas onde o modelo falha (lógica, suposição, criatividade) para melhorias futuras [1].
**5. Tutoria e Educação:** Geração de provas passo a passo para fins didáticos, ajudando estudantes a entenderem a estrutura e a lógica por trás das demonstrações matemáticas.

## Pitfalls
**1. Generalização Excessiva de Padrões:** A tendência de LLMs de supergeneralizar padrões observados em casos numéricos menores para casos maiores, sem fornecer uma prova formal que sustente a afirmação [1].
**2. Citações Inexistentes (Non-Existent Citation):** Fabricação de referências, teoremas ou lemas que parecem plausíveis, mas são falsos ou não verificáveis, para dar credibilidade à prova [1].
**3. Falhas Lógicas Ocultas:** A prova pode parecer correta superficialmente, mas conter falácias lógicas ou suposições injustificadas que invalidam a conclusão.
**4. Falta de Clareza Estrutural:** Variação na coerência e na estrutura da solução, dificultando a verificação do raciocínio passo a passo.
**5. Answer Boxing:** O modelo pode focar em fornecer a resposta final em um formato "encaixotado" (boxed answer), um artefato de treinamento, em vez de se concentrar no rigor da prova [1].

## URL
[https://arxiv.org/pdf/2503.21934](https://arxiv.org/pdf/2503.21934)
