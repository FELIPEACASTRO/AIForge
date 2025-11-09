# Prompts de Programação Python

## Description
**Prompts de Programação Python** são instruções estruturadas e detalhadas fornecidas a um Large Language Model (LLM) com o objetivo de gerar, depurar, refatorar, documentar ou explicar código na linguagem Python. A eficácia desses prompts depende da clareza, do contexto fornecido e da aplicação de técnicas de engenharia de prompt específicas para tarefas de codificação. Eles transformam o LLM de um gerador de texto genérico em um assistente de programação altamente especializado, capaz de lidar com tarefas que variam desde a criação de pequenos scripts até a arquitetura de sistemas complexos, sempre com foco na aderência a padrões de código como o PEP 8. A pesquisa recente (2023-2025) enfatiza a importância de prompts iterativos, a inclusão de testes e a definição clara de restrições de ambiente e desempenho.

## Examples
```
**1. Geração de Função com Teste Unitário (Function Generation with Unit Test):**
```
**Role:** You are a Python software engineer.
**Task:** Write a Python function `calculate_median(data_list)` that takes a list of numbers and returns the median.
**Constraints:** The function must handle both even and odd-sized lists. Include a unit test using the `unittest` module that asserts the median of `[1, 2, 3, 4, 5]` is `3` and the median of `[1, 2, 3, 4]` is `2.5`.
**Output Format:** Only the Python code block.
```

**2. Refatoração para Otimização (Refactoring for Optimization):**
```
**Task:** Refactor the following Python code to improve its performance and adhere to PEP 8. The goal is to replace the nested loops with a more Pythonic and efficient approach, preferably using a dictionary or set for O(1) lookups.
**Code to Refactor:**
```python
def find_duplicates(list1, list2):
    duplicates = []
    for item1 in list1:
        for item2 in list2:
            if item1 == item2 and item1 not in duplicates:
                duplicates.append(item1)
    return duplicates
```
**Output Format:** Only the refactored Python code block.
```

**3. Geração de Script de Análise de Dados (Data Analysis Script Generation):**
```
**Role:** You are a Data Scientist.
**Task:** Write a Python script using the Pandas library to perform the following steps:
1. Load the CSV file named 'sales_data.csv' into a DataFrame.
2. Calculate the mean of the 'Revenue' column, grouped by the 'Region' column.
3. Print the resulting Series.
**Constraints:** Assume the CSV file exists in the current directory. Do not use any external functions.
**Output Format:** Only the complete Python script.
```

**4. Depuração de Erro (Error Debugging):**
```
**Task:** The following Python code is raising a `KeyError: 'city'`. Analyze the code and the traceback, identify the bug, and provide the corrected code.
**Code:**
```python
data = [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25, 'city': 'New York'}]
for item in data:
    print(item['city'])
```
**Traceback:**
```
Traceback (most recent call last):
  File "script.py", line 3, in <module>
    print(item['city'])
KeyError: 'city'
```
**Output Format:** First, a brief explanation of the bug, then the corrected Python code block using a `try-except` block or `.get()`.
```

**5. Explicação de Código Complexo (Complex Code Explanation):**
```
**Task:** Explain the following Python code snippet line by line, focusing on the use of the `__call__` method and the concept of a callable class instance.
**Code:**
```python
class Multiplier:
    def __init__(self, factor):
        self.factor = factor
    def __call__(self, number):
        return number * self.factor
m = Multiplier(5)
result = m(10)
```
**Output Format:** A detailed, didactic explanation in Portuguese, formatted as a numbered list.
```

**6. Geração de Documentação (Documentation Generation):**
```
**Task:** Generate a comprehensive docstring in the Google Python Style Guide format for the following function. The docstring must include a description, arguments, return value, and an example of usage.
**Function:**
```python
def connect_to_database(host, port=5432, timeout=5):
    """Connects to the PostgreSQL database."""
    # implementation details...
    return connection_object
```
**Output Format:** Only the docstring content.
```
```

## Best Practices
**1. Seja Específico e Estruturado (Specific and Structured):**
   - **Defina o Papel:** Comece o prompt definindo o papel do LLM (ex: "Você é um engenheiro de software Python sênior...").
   - **Especifique a Versão:** Inclua a versão do Python e bibliotecas (ex: "Use Python 3.11 e a biblioteca Pandas").
   - **Formato de Saída:** Peça o código dentro de blocos de código Markdown (` ```python `) e instrua o modelo a não incluir explicações desnecessárias, a menos que solicitado.

**2. Forneça Contexto e Restrições (Provide Context and Constraints):**
   - **Esquema de Dados:** Se aplicável, forneça o esquema de dados, nomes de variáveis, classes ou funções existentes.
   - **Restrições:** Inclua restrições de desempenho, segurança (ex: "O código deve ser otimizado para O(n)"), ou estilo (ex: "Siga o PEP 8").
   - **Testes:** Peça para o modelo incluir testes unitários (`unittest` ou `pytest`) para o código gerado.

**3. Use Técnicas Avançadas (Use Advanced Techniques):**
   - **Chain-of-Thought (CoT):** Para tarefas complexas, peça ao modelo para "pensar em voz alta" ou descrever o plano de implementação antes de gerar o código.
   - **Few-Shot Learning:** Forneça um ou dois exemplos de pares de problema/solução para guiar o estilo e a complexidade do código.
   - **Iteração e Refinamento:** Em vez de um único prompt longo, use prompts curtos e iterativos para refinar o código (ex: "Refatore a função `process_data` para usar `list comprehension`").

## Use Cases
**1. Geração Rápida de Protótipos (Rapid Prototyping):**
   - Criar rapidamente funções, classes ou scripts de utilidade para testar uma ideia ou conceito, reduzindo o tempo de desenvolvimento inicial.

**2. Depuração e Correção de Erros (Debugging and Error Correction):**
   - Inserir um trecho de código com um erro e a mensagem de traceback para que o LLM identifique a causa e sugira a correção.

**3. Refatoração e Otimização de Código (Code Refactoring and Optimization):**
   - Solicitar a melhoria de código existente para maior eficiência, legibilidade (aderência ao PEP 8) ou modernização (ex: converter loops em `list comprehensions`).

**4. Tradução de Linguagem (Language Translation):**
   - Converter código de outra linguagem (ex: JavaScript, R) para Python, mantendo a lógica e a funcionalidade.

**5. Geração de Documentação e Testes (Documentation and Test Generation):**
   - Criar automaticamente docstrings (no formato Sphinx, NumPy ou Google) e testes unitários (usando `unittest` ou `pytest`) para funções e módulos existentes.

**6. Explicação e Aprendizado (Explanation and Learning):**
   - Pedir ao LLM para explicar o funcionamento de um trecho de código complexo, um conceito específico do Python (ex: *decorators*, *generators*) ou o propósito de uma biblioteca.

## Pitfalls
**1. Falta de Contexto (Lack of Context):**
   - **Erro:** Fornecer prompts vagos (ex: "Escreva um código Python para processar dados") sem especificar o formato de entrada, o resultado esperado ou as bibliotecas a serem usadas.
   - **Consequência:** Geração de código genérico, ineficiente ou que não se integra ao projeto existente.

**2. Confiança Excessiva e Não Verificação (Over-reliance and No Verification):**
   - **Erro:** Assumir que o código gerado pelo LLM está sempre correto e funcional, especialmente para lógica complexa ou segurança.
   - **Consequência:** Introdução de bugs, vulnerabilidades de segurança ou código não otimizado no projeto. **Sempre verifique e teste o código gerado.**

**3. Ignorar Restrições de Token (Ignoring Token Limits):**
   - **Erro:** Fornecer bases de código muito grandes para depuração ou refatoração em um único prompt, excedendo o limite de contexto do modelo.
   - **Consequência:** O modelo ignora partes do código ou gera uma resposta incompleta. **Solução:** Dividir a tarefa em prompts menores e iterativos.

**4. Não Especificar o Estilo (Not Specifying Style):**
   - **Erro:** Não mencionar padrões de codificação (ex: PEP 8) ou convenções de nomenclatura do projeto.
   - **Consequência:** Código funcional, mas inconsistente com o restante do projeto, exigindo refatoração manual posterior.

**5. Prompts de "Caixa Preta" (Black Box Prompts):**
   - **Erro:** Pedir apenas o resultado final sem solicitar o processo de raciocínio (Chain-of-Thought).
   - **Consequência:** Dificuldade em depurar ou entender a lógica por trás de uma solução complexa, perdendo a oportunidade de aprendizado.

## URL
[https://github.com/potpie-ai/potpie/wiki/How-to-write-good-prompts-for-generating-code-from-LLMs](https://github.com/potpie-ai/potpie/wiki/How-to-write-good-prompts-for-generating-code-from-LLMs)
