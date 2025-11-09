# Citation & Bibliography Prompts

## Description
A técnica de **Prompts de Citação e Bibliografia** (Citation & Bibliography Prompts) é uma estratégia de Engenharia de Prompt focada em instruir Modelos de Linguagem Grande (LLMs) a gerar referências bibliográficas e citações no corpo do texto com alta precisão e em formatos específicos (como APA, MLA, Chicago, ABNT). O princípio central é a aplicação do método **RTCF (Role, Task, Context, Format)**, onde o usuário fornece ao LLM um papel especializado (ex: bibliotecário), a tarefa clara (ex: gerar citação), o **Contexto** completo (dados da fonte como autor, título, ano, DOI) e o **Formato** de saída exato (ex: APA 7ª Edição, entrada de lista de referências). Ao fornecer o contexto completo e estruturado, o usuário mitiga a tendência do LLM de "alucinar" dados de fontes, garantindo a acurácia e a conformidade com as normas acadêmicas. É uma técnica essencial para a produção de conteúdo acadêmico e técnico confiável.

## Examples
```
**1. Citação de Artigo de Periódico (APA 7ª Edição - Zero-Shot)**
```
**Papel:** Atue como um bibliotecário especializado em APA 7ª Edição.
**Tarefa:** Gere a entrada completa da lista de referências para o artigo de periódico a seguir.
**Contexto:**
- Autores: Smith, J. A., & Jones, B. C.
- Ano de Publicação: 2024
- Título do Artigo: The Future of Prompt Engineering in LLMs
- Título do Periódico: Journal of AI Research
- Volume: 15
- Número: 2
- Páginas: 112-130
- DOI: 10.1000/jair.2024.15.2.112
**Formato:** Forneça apenas a citação formatada.
```

**2. Citação de Livro (MLA 9ª Edição - Zero-Shot)**
```
**Papel:** Você é um assistente de pesquisa acadêmica.
**Tarefa:** Crie uma entrada de "Works Cited" (Referências) para o livro abaixo.
**Contexto:**
- Autor: Johnson, Emily
- Título do Livro: The Algorithmic Muse
- Editora: Tech Press
- Ano de Publicação: 2023
- Cidade de Publicação: New York
**Formato:** Formate em MLA 9ª Edição.
```

**3. Citação no Corpo do Texto (ABNT NBR 6023 - Citação Direta)**
```
**Papel:** Atue como um editor de textos acadêmicos brasileiro.
**Tarefa:** Gere a citação direta (com aspas e página) no corpo do texto para a frase a seguir, usando o sistema autor-data.
**Frase:** "A inteligência artificial transformará a educação superior."
**Contexto:**
- Autor: Silva, M. R.
- Ano: 2025
- Página: 45
**Formato:** ABNT NBR 6023 (Sobrenome, Ano, p. X).
```

**4. Citação de Website (Chicago 17ª Edição - Notas e Bibliografia)**
```
**Papel:** Você é um especialista em estilo Chicago.
**Tarefa:** Gere a nota de rodapé e a entrada de bibliografia para a página web.
**Contexto:**
- Autor: The Prompting Guide Team
- Título da Página: Advanced Prompting Techniques
- Nome do Site: PromptingGuide.ai
- Data de Publicação: 15 de Outubro de 2023
- URL: https://www.promptingguide.ai/techniques/advanced
- Data de Acesso: 8 de Novembro de 2025
**Formato:** Gere a nota de rodapé completa e a entrada de bibliografia separadamente.
```

**5. Geração de BibTeX para Artigo (Formato Técnico)**
```
**Papel:** Atue como um gerador de metadados para LaTeX.
**Tarefa:** Converta os dados da fonte em uma entrada BibTeX formatada como @article.
**Contexto:**
- Autor: Chen, H., & Li, W.
- Título: Large Language Models as Citation Generators
- Periódico: AI Review
- Ano: 2024
- Volume: 8
- Páginas: 50-65
**Formato:** Gere o código BibTeX completo.
```

**6. Few-Shot para Estilo Personalizado**
```
**Papel:** Você é um formatador de referências.
**Tarefa:** Formate a fonte de Contexto no Estilo X.
**Exemplo (Few-Shot):**
- Fonte Exemplo: Autor: Adams, S. | Título: The Guide | Ano: 2020
- Saída Exemplo: ADAMS, S. (2020). The Guide.
**Contexto:**
- Autor: Baker, L.
- Título: Prompting for Success
- Ano: 2025
**Formato:** Formate a fonte de Contexto no mesmo Estilo X do Exemplo.
```
```

## Best Practices
**1. Forneça Contexto Completo e Estruturado (RTCF):** Use o framework **Role, Task, Context, Format** (Papel, Tarefa, Contexto, Formato). O Contexto é o mais crítico; forneça todos os metadados da fonte (autor, título, ano, editora, DOI, URL) de forma clara e organizada, em vez de apenas um link.
**2. Especifique o Estilo e a Edição:** Seja explícito sobre o estilo de citação (ex: APA, MLA, ABNT) e a edição (ex: 7ª Edição, 9ª Edição).
**3. Use Few-Shot Prompting para Formatos Complexos:** Para estilos menos comuns ou formatos de fonte complexos (ex: patentes, relatórios governamentais), inclua um ou dois exemplos de citações corretas no estilo desejado antes de solicitar a nova citação.
**4. Defina o Tipo de Saída:** Especifique se você precisa da **entrada da lista de referências** (bibliografia completa) ou apenas da **citação no corpo do texto** (citação parentética ou narrativa).
**5. Validação Cruzada:** Sempre verifique a saída do LLM com um gerador de citação tradicional (como Citation Machine ou Scribbr) ou com o manual de estilo oficial, especialmente para trabalhos acadêmicos críticos.

## Use Cases
**1. Produção Acadêmica:** Geração rápida e precisa de listas de referências (bibliografias) e citações no corpo do texto para artigos, teses, dissertações e trabalhos escolares em qualquer estilo (APA, MLA, Chicago, ABNT, Vancouver).
**2. Revisão e Padronização:** Conversão de listas de referências de um estilo para outro (ex: de MLA para APA) ou padronização de metadados de fontes coletadas.
**3. Pesquisa e Desenvolvimento (P&D):** Criação de entradas BibTeX ou RIS para gerenciadores de referências (como Zotero ou Mendeley), facilitando a organização de grandes volumes de literatura.
**4. Jornalismo e Conteúdo Técnico:** Garantir que todas as fontes em um artigo de notícias ou manual técnico sejam referenciadas de forma consistente e profissional.
**5. Educação:** Ferramenta de aprendizado para estudantes entenderem a estrutura e os requisitos de diferentes estilos de citação, usando o LLM como um verificador de formato.

## Pitfalls
**1. Confiar em Links:** O erro mais comum é fornecer apenas uma URL e esperar que o LLM extraia todos os metadados corretamente. O LLM pode "alucinar" o autor, a data ou o título.
**2. Falta de Especificidade no Formato:** Solicitar apenas "uma citação APA" sem especificar a edição (ex: 6ª vs. 7ª) ou o tipo de entrada (referência vs. in-text) leva a resultados inconsistentes.
**3. Ignorar o Contexto:** Não fornecer metadados cruciais (como DOI, número da edição, ou nome da editora) força o LLM a adivinhar, aumentando a taxa de erro.
**4. Prompting de Zero-Shot para Estilos Raros:** Para estilos de citação menos comuns ou formatos de fonte muito específicos (ex: normas de uma universidade), o LLM pode falhar sem um exemplo (Few-Shot) para guiar a formatação.
**5. Não Citar a IA:** Esquecer de citar o próprio LLM (como ChatGPT ou Gemini) quando ele é usado para gerar ou analisar conteúdo, o que é uma exigência crescente em muitas diretrizes acadêmicas.

## URL
[https://www.getpassionfruit.com/blog/blog-ai-prompt-engineering-citations](https://www.getpassionfruit.com/blog/blog-ai-prompt-engineering-citations)
