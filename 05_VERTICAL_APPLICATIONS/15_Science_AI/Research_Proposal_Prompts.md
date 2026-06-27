# Research Proposal Prompts

## Description
Os **Research Proposal Prompts** são instruções de engenharia de prompt projetadas para auxiliar pesquisadores, acadêmicos e estudantes na elaboração de propostas de pesquisa, teses, dissertações e pedidos de financiamento (grants). Eles transformam o Large Language Model (LLM) em um assistente de pesquisa e revisor, capaz de gerar ideias de tópicos, estruturar a metodologia, refinar a linguagem, simular a avaliação de revisores e garantir o alinhamento com os critérios de editais e agências de fomento. O foco principal é aprimorar a clareza, a coerência e a persuasão do documento final, cobrindo desde a formulação da pergunta de pesquisa até a justificativa do orçamento.

## Examples
```
**1. Geração de Tópicos e Justificativa (Topic Generation and Justification)**
`"Aja como um especialista em [Área de Pesquisa, ex: Neurociência Computacional]. Gere 5 ideias de tópicos de pesquisa inovadores para um projeto de doutorado, cada um com: 1) Uma pergunta de pesquisa clara, 2) Uma justificativa de por que é relevante em 2025, e 3) Um desafio metodológico principal."`

**2. Definição de Escopo e Metodologia (Scope and Methodology Definition)**
`"Defina o escopo e esboce a metodologia para uma proposta de pesquisa sobre [Tópico, ex: O impacto da IA generativa na produtividade acadêmica]. A metodologia deve incluir: 1) O tipo de pesquisa (quantitativa/qualitativa), 2) As ferramentas/softwares a serem utilizados, e 3) As etapas para garantir a ética e eliminar vieses."`

**3. Simulação de Avaliação de Revisor (Reviewer Evaluation Simulation)**
`"Aja como um revisor cético da [Agência de Fomento, ex: FAPESP]. Avalie o rascunho da minha proposta (texto anexado) com base nos critérios de 'Mérito Científico' e 'Viabilidade Técnica'. Identifique as 3 maiores fraquezas e sugira como fortalecer a argumentação."`

**4. Refinamento da Linguagem e Concisão (Language Refinement and Conciseness)**
`"Revise o seguinte parágrafo da minha seção de 'Resultados Esperados' para torná-lo mais conciso e persuasivo, mantendo a precisão técnica. O público-alvo é um comitê multidisciplinar. [Insira o Parágrafo]." `

**5. Elaboração de Orçamento e Recursos (Budget and Resource Elaboration)**
`"Com base na metodologia descrita (anexada), liste os recursos necessários (hardware, software, pessoal) e sugira uma justificativa de orçamento para cada item, focando na necessidade e no custo-benefício para um projeto de 2 anos."`

**6. Análise de Lacunas na Literatura (Literature Gap Analysis)**
`"Analise o resumo da literatura (texto anexado) e identifique as lacunas de pesquisa que meu projeto pretende preencher. Formule uma declaração de problema clara e concisa (máximo 100 palavras) que posicione meu trabalho como essencial."`

**7. Escrita de Resumo/Abstract (Abstract Writing)**
`"Gere um resumo (abstract) de 250 palavras para minha proposta de pesquisa. Use as seguintes informações: 1) Objetivo principal, 2) Metodologia resumida, 3) Principais resultados esperados e 4) Implicações/Impacto."`
```

## Best Practices
**1. Fornecer Contexto Completo e Estrutura:** Inclua o máximo de detalhes possível sobre o tema, objetivos, público-alvo (revisores, comitê de bolsa) e o formato exigido (BAA, edital, etc.). Use prompts de "cadeia de pensamento" (Chain-of-Thought) para que a IA estruture a resposta em seções lógicas.

**2. Simulação de Revisor (Role-Playing):** A melhor prática é instruir a IA a assumir o papel de um revisor cético ou especialista da área. Peça para ela avaliar o rascunho da proposta com base em critérios específicos (mérito científico, viabilidade, impacto, alinhamento com o edital).

**3. Refinamento Iterativo:** Use a IA para refinar a proposta em etapas. Comece com o tópico, depois a metodologia, o orçamento e, por fim, a revisão. Cada saída da IA deve ser um insumo para o próximo prompt.

**4. Foco na Clareza e Concisão:** Peça à IA para simplificar a linguagem técnica, garantir a clareza dos objetivos e resumir seções longas, pois propostas de pesquisa muitas vezes têm limites de palavras rigorosos.

## Use Cases
**1. Propostas Acadêmicas:** Elaboração de projetos de pesquisa para mestrado, doutorado e pós-doutorado.

**2. Pedidos de Financiamento (Grants):** Redação e refinamento de propostas para agências de fomento (ex: CNPq, FAPESP, NSF, NIH), garantindo o alinhamento com os editais (BAAs - Broad Agency Announcements).

**3. Planejamento de Tese/Dissertação:** Auxílio na estruturação de capítulos, definição de metodologias e análise de lacunas na literatura para o desenvolvimento de trabalhos de conclusão.

**4. Propostas de Projetos Internos:** Criação de documentos para justificar a alocação de recursos em projetos de P&D dentro de empresas ou instituições.

**5. Revisão e Autoavaliação:** Uso da IA para simular a avaliação de pares e identificar falhas lógicas, metodológicas ou de alinhamento antes da submissão formal.

## Pitfalls
**1. Confiança Excessiva na Geração de Conteúdo:** O maior erro é usar o texto gerado pela IA diretamente. A IA pode 'alucinar' referências, metodologias inviáveis ou dados falsos. **Sempre** verifique a precisão científica e a viabilidade.

**2. Falta de Especificidade no Contexto:** Prompts vagos levam a propostas genéricas. É crucial fornecer o máximo de contexto específico (área, subárea, critérios do edital, público-alvo) para obter uma saída relevante.

**3. Ignorar a Voz do Pesquisador:** A proposta deve refletir a voz, a experiência e o conhecimento único do pesquisador. A IA deve ser usada para refinar e estruturar, não para substituir a autoria.

**4. Não Utilizar a Simulação de Revisor:** Deixar de usar a IA para simular a avaliação de um revisor é perder uma das maiores vantagens dessa técnica. A simulação ajuda a identificar pontos fracos antes da submissão.

## URL
[https://stanfordh4d.substack.com/p/tt4d-using-llms-for-research-proposals](https://stanfordh4d.substack.com/p/tt4d-using-llms-for-research-proposals)
