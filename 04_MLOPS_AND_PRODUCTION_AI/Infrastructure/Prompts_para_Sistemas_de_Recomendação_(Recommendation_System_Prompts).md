# Prompts para Sistemas de Recomendação (Recommendation System Prompts)

## Description
A engenharia de prompts para Sistemas de Recomendação (RS) envolve a criação de instruções em linguagem natural para Large Language Models (LLMs) com o objetivo de gerar **recomendações personalizadas** de itens (como notícias, produtos, filmes, etc.) para usuários específicos. Esta abordagem aproveita a capacidade de raciocínio, compreensão de contexto e geração de linguagem natural dos LLMs para simular o comportamento de um sistema de recomendação tradicional.

Em vez de depender apenas de modelos matriciais complexos, os LLMs podem processar o histórico de interações do usuário, perfis de itens e restrições de tarefa (fornecidos no prompt) para produzir listas de recomendação e, crucialmente, **explicações** para essas recomendações. O framework **RecPrompt** [1] é um exemplo notável que utiliza um ciclo de feedback automatizado para otimizar prompts, demonstrando que prompts gerados por LLMs podem superar modelos neurais profundos tradicionais em certas métricas de recomendação.

## Examples
```
## Exemplo 1: Recomendação de Notícias (Baseado em RecPrompt)
**Contexto (System Prompt):** Você é um sistema de recomendação de notícias personalizado. Sua tarefa é analisar o histórico de leitura do usuário e uma lista de artigos candidatos para sugerir os 5 artigos mais relevantes.

**Input (User Prompt):**
```
# Perfil do Usuário
Histórico de Notícias Lidas:
1. "Avanços na Fusão Nuclear: Novo Reator Quebra Recorde de Energia"
2. "Impacto da IA na Criação de Conteúdo Digital"
3. "Crise Hídrica na Europa: Medidas de Racionamento"

# Artigos Candidatos
1. "Novas Políticas de Privacidade do Google"
2. "Descoberta de Exoplaneta com Potencial de Vida"
3. "O Futuro dos Carros Elétricos e a Sustentabilidade"
4. "Análise da Última Temporada de Série de Ficção Científica"
5. "Tendências de Investimento em Energias Renováveis"

# Tarefa
Com base no histórico do usuário, classifique os 5 Artigos Candidatos por relevância.
Formato de Saída:
<START>
[Artigo mais relevante]
[Segundo mais relevante]
...
[Artigo menos relevante]
<END>
```

## Exemplo 2: Recomendação de Produtos de E-commerce
**Contexto (System Prompt):** Você é um assistente de compras. Recomende 3 produtos da lista de candidatos que melhor se alinhem com o perfil de compra e o objetivo atual do usuário.

**Input (User Prompt):**
```
# Perfil do Usuário
- Compras Recentes: Câmera DSLR, Lente 50mm, Tripé Profissional.
- Navegação: Filtros ND, Mochilas para Equipamento Fotográfico.
- Objetivo Atual: Comprar acessórios para fotografia de paisagem.

# Produtos Candidatos
1. Mochila Ergonômica para Câmera (Capacidade Grande)
2. Kit de Limpeza de Sensores
3. Drone com Câmera 4K
4. Cartão de Memória de Alta Velocidade 128GB
5. Livro: "Dominando a Fotografia de Retrato"

# Tarefa
Recomende os 3 produtos mais adequados para o objetivo de fotografia de paisagem. Inclua uma breve justificativa para cada.
```

## Exemplo 3: Recomendação de Filmes/Séries com Explicação
**Contexto (System Prompt):** Atue como um curador de filmes. Recomende um filme ou série e forneça uma explicação detalhada (máximo 50 palavras) de por que ele se encaixa no gosto do usuário.

**Input (User Prompt):**
```
# Histórico de Visualização
- Filmes Favoritos: "A Origem", "Interestelar", "Blade Runner 2049" (Ficção Científica, Temas Complexos, Direção Visual Forte).
- Gêneros Preferidos: Sci-Fi, Thriller Psicológico.
- Gêneros Evitados: Comédia Romântica, Terror Slasher.

# Tarefa
Recomende um único título que eu provavelmente amarei.
```

## Exemplo 4: Recomendação Baseada em Restrições (Viagem)
**Contexto (System Prompt):** Você é um agente de viagens especializado em roteiros personalizados.

**Input (User Prompt):**
```
# Preferências de Viagem
- Destino: Europa.
- Duração: 10 dias.
- Orçamento: Médio (excluindo passagens aéreas).
- Interesses: História Antiga, Gastronomia Local, Poucas Multidões.

# Tarefa
Sugira 3 cidades europeias que atendam a essas restrições. Para cada cidade, liste um ponto turístico histórico e um prato típico.
```

## Exemplo 5: Recomendação de Código/Ferramenta (Tecnologia)
**Contexto (System Prompt):** Você é um especialista em desenvolvimento de software. Recomende a melhor biblioteca Python para a tarefa descrita.

**Input (User Prompt):**
```
# Tarefa
Preciso implementar um sistema de processamento de linguagem natural (NLP) para classificar sentimentos em grandes volumes de texto (mais de 1 milhão de documentos). O sistema deve ser escalável e rápido.

# Tarefa
Recomende a biblioteca Python mais adequada (excluindo NLTK e SpaCy) e justifique a escolha em termos de escalabilidade e desempenho.
```
```

## Best Practices
1. **Definir a Persona (Role-Playing):** Comece o prompt com uma instrução clara de função (ex: "Você é um curador de filmes", "Você é um especialista em e-commerce"). Isso alinha o estilo e o foco da resposta do LLM.
2. **Fornecer Contexto Detalhado:** Inclua o máximo de dados relevantes do usuário (histórico de interações, preferências, demografia, intenção atual) e dos itens candidatos. A qualidade da recomendação é diretamente proporcional à riqueza do contexto.
3. **Estruturar a Saída (Output Formatting):** Especifique o formato de saída desejado (JSON, lista numerada, tags XML como `<START>` e `<END>`). Isso facilita o parsing e a integração da resposta do LLM em um sistema de recomendação maior.
4. **Solicitar Explicações (Explainability):** Peça ao LLM para justificar suas recomendações. A **explicabilidade** é um dos maiores benefícios dos LLMs em RS, aumentando a confiança do usuário.
5. **Usar Raciocínio em Cadeia (Chain-of-Thought):** Para tarefas complexas, instrua o LLM a "pensar passo a passo" (ex: "1. Analise o perfil. 2. Filtre os candidatos. 3. Classifique e justifique."). Isso melhora a precisão e a rastreabilidade do processo de recomendação.

## Use Cases
*   **Recomendação de Conteúdo:** Sugestão de notícias, artigos, vídeos, músicas ou podcasts.
*   **E-commerce e Varejo:** Recomendação de produtos, sugestões de "próxima compra" ou criação de coleções personalizadas.
*   **Viagens e Turismo:** Sugestão de destinos, hotéis, atividades ou roteiros baseados em restrições e interesses.
*   **Educação (EdTech):** Recomendação de cursos, materiais de estudo ou trilhas de aprendizado personalizadas.
*   **Saúde:** Sugestão de artigos médicos, planos de bem-estar ou profissionais de saúde com base no histórico e sintomas.
*   **Recomendação Fria (Cold Start):** Uso de prompts para inferir preferências de novos usuários com base em informações demográficas ou respostas a perguntas iniciais.

## Pitfalls
*   **Alucinações de Itens:** O LLM pode inventar itens ou produtos que não existem na lista de candidatos fornecida, exigindo uma etapa de validação externa.
*   **Viés de Popularidade:** LLMs tendem a favorecer itens populares ou amplamente discutidos, mesmo que não sejam os mais relevantes para o perfil de nicho do usuário.
*   **Custo e Latência:** A execução de prompts complexos em LLMs grandes (como GPT-4) pode ser lenta e cara, tornando-os inadequados para sistemas de recomendação de alta frequência e baixa latência.
*   **Dependência do Contexto:** A qualidade da recomendação é altamente dependente da quantidade de contexto que o LLM pode processar (limite de tokens), o que pode ser um problema para usuários com históricos de interação muito longos.
*   **Falta de Interação:** LLMs não interagem com o ambiente de forma tradicional (como um modelo de aprendizado por reforço), limitando sua capacidade de aprender com o feedback em tempo real, a menos que um framework como o RecPrompt seja implementado.

## URL
[https://www.prompthub.us/blog/recprompt-a-prompt-engineering-framework-for-llm-recommendations](https://www.prompthub.us/blog/recprompt-a-prompt-engineering-framework-for-llm-recommendations)
