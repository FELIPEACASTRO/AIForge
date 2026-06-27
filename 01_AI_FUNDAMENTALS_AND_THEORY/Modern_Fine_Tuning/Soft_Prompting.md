# Soft Prompting

## Description
Soft Prompting, ou "Prompting Suave", é uma técnica avançada de Engenharia de Prompt que se diferencia do Hard Prompting (prompts em linguagem natural) por não ser legível por humanos. Em vez de usar texto, o Soft Prompting utiliza **vetores de *embedding* contínuos e treináveis** que são concatenados à entrada do modelo de linguagem grande (LLM) antes do processamento. Esses vetores são otimizados por meio de um processo de ajuste fino (*fine-tuning*) leve, como o **Prompt Tuning** ou **Prefix Tuning**, para codificar o conhecimento da tarefa diretamente no espaço latente do modelo. O objetivo é guiar o comportamento do LLM para uma tarefa específica (como classificação ou sumarização) com alta precisão, sem a necessidade de ajustar todos os milhões de parâmetros do modelo base. É uma forma de adaptação de modelo que oferece um equilíbrio entre o ajuste fino completo (que é caro) e o prompting tradicional (que pode ser menos preciso para tarefas complexas). Sua principal característica é a **opacidade** e a **otimização automatizada** para desempenho em tarefas especializadas.

## Examples
```
O Soft Prompting não é expresso em linguagem natural, mas sim como um conjunto de vetores numéricos (embeddings). Portanto, os exemplos abaixo são **conceituais**, ilustrando a intenção da otimização, e não o prompt em si.

1.  **Classificação de Sentimento:**
    *   **Intenção:** Otimizar o modelo para distinguir entre sarcasmo e ironia em avaliações de produtos.
    *   **Representação Conceitual:** `[Soft Prompt Otimizado para Sarcasmo] + "O produto é 'ótimo', demorou apenas 3 meses para chegar."`

2.  **Sumarização Extrativa:**
    *   **Intenção:** Otimizar o modelo para priorizar a extração de datas e nomes de entidades em relatórios financeiros longos.
    *   **Representação Conceitual:** `[Soft Prompt Focado em Entidades Financeiras] + "Sumarize o relatório anual da empresa X."`

3.  **Tradução Específica de Domínio:**
    *   **Intenção:** Otimizar a tradução de termos médicos complexos (e.g., de inglês para português) com alta fidelidade terminológica.
    *   **Representação Conceitual:** `[Soft Prompt de Tradução Médica] + "Translate 'myocardial infarction' to Portuguese."`

4.  **Geração de Código:**
    *   **Intenção:** Otimizar o modelo para gerar código Python seguindo estritamente o padrão de estilo PEP 8.
    *   **Representação Conceitual:** `[Soft Prompt de Conformidade PEP 8] + "Write a Python function to calculate the factorial of a number."`

5.  **Resposta a Perguntas (QA):**
    *   **Intenção:** Otimizar o modelo para ser mais cauteloso e citar fontes internas ao responder perguntas sobre regulamentações legais.
    *   **Representação Conceitual:** `[Soft Prompt de Cautela Legal e Citação] + "Quais são os requisitos de conformidade para o GDPR?"`
```

## Best Practices
**1. Integração com Prompting Rígido (Hard Prompting):** Combine a precisão do Soft Prompting (para tarefas específicas) com a interpretabilidade e o controle do Hard Prompting (para instruções gerais e restrições de formato). **2. Otimização Contínua:** O Soft Prompt deve ser tratado como um hiperparâmetro que requer otimização contínua e revalidação à medida que o modelo base ou a distribuição dos dados da tarefa mudam. **3. Foco em Tarefas de Alta Precisão:** Reserve o Soft Prompting para tarefas onde a precisão e a otimização de recursos são críticas, como classificação de texto em larga escala ou análise de sentimentos sutil. **4. Validação Cruzada Rigorosa:** Devido à sua natureza não interpretável, é crucial validar o desempenho do Soft Prompt em um conjunto de dados de teste robusto para garantir que a otimização não tenha levado a um *overfitting* (sobreajuste) excessivo.

## Use Cases
**1. Adaptação Rápida a Novas Tarefas (Prompt Tuning):** É o principal caso de uso, permitindo que LLMs sejam rapidamente adaptados a novas tarefas de *downstream* (como classificação de 100 categorias de texto) com um custo computacional muito menor do que o ajuste fino completo. **2. Otimização de Recursos em Produção:** Em ambientes de produção com restrições de latência e memória, o Soft Prompting permite que um único modelo base seja adaptado a múltiplas tarefas sem a necessidade de implantar várias instâncias de modelos totalmente ajustados. **3. Tarefas de Alta Precisão e Nuance:** Usado em tarefas onde a linguagem natural (Hard Prompt) não consegue capturar a nuance necessária, como a detecção de *hate speech* sutil ou a identificação de entidades em textos altamente técnicos. **4. Preservação do Conhecimento Geral:** Ao ajustar apenas os vetores de prompt e não os pesos do modelo, o Soft Prompting ajuda a preservar o conhecimento geral e as capacidades do modelo base, evitando o fenômeno de "catastrophic forgetting" (esquecimento catastrófico) comum no ajuste fino completo.

## Pitfalls
**1. Falta de Interpretabilidade (Opacidade):** A natureza não legível por humanos dos vetores de embedding torna impossível inspecionar ou depurar o prompt diretamente, dificultando a compreensão de por que o modelo falhou. **2. Risco de Overfitting:** O Soft Prompt é ajustado para um conjunto de dados de treinamento específico. Se o conjunto de dados for pequeno ou não representativo, o prompt pode se tornar excessivamente ajustado (*overfitted*), falhando em generalizar para novos dados. **3. Dependência do Modelo Base:** O Soft Prompt é intrinsecamente ligado ao modelo de linguagem para o qual foi treinado. Ele não pode ser transferido para um modelo diferente (mesmo que seja da mesma família) sem um novo processo de ajuste. **4. Necessidade de Infraestrutura de Treinamento:** Ao contrário do Hard Prompting, que requer apenas um editor de texto, o Soft Prompting exige um pipeline de treinamento (hardware, dados rotulados, código de otimização) para gerar os vetores de embedding.

## URL
[https://futureagi.com/blogs/hard-prompt-vs-soft-prompt-2025](https://futureagi.com/blogs/hard-prompt-vs-soft-prompt-2025)
