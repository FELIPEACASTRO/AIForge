# Frameworks de Prompt para Revisão Bibliográfica (CO-STAR, POSE, Sandwich)

## Description

Este recurso detalha a aplicação de **Frameworks de Engenharia de Prompt** específicos para otimizar a **Pesquisa Acadêmica e a Revisão Bibliográfica** utilizando Grandes Modelos de Linguagem (LLMs) como o ChatGPT. Os frameworks **CO-STAR**, **POSE** e **Sandwich** fornecem estruturas sistemáticas para a criação de prompts, visando melhorar a qualidade, a relevância e a profundidade das revisões de literatura geradas por IA. O foco está em transformar a interação com a IA de uma simples consulta para um processo estruturado e didático, essencial para o ambiente acadêmico.

## Statistics

Estudos (Islam et al., 2025) demonstram que a exposição a esses frameworks melhora significativamente o comportamento de prompt dos estudantes, resultando em prompts mais eficazes e revisões de literatura de maior qualidade. A melhoria é observada na estrutura e organização dos trabalhos. O framework **Markdown Table + CoT** demonstrou ser altamente eficaz (até 94.35% de precisão em um estudo) para tarefas de extração de dados estruturados de resumos acadêmicos (Lee et al., 2025). O uso de frameworks ajuda a mitigar o risco de alucinações e confabulações, um problema crescente em LLMs.

## Features

**CO-STAR (Contexto, Objetivo, Estilo, Tom, Público, Resposta):** Estrutura abrangente para definir todos os parâmetros de saída da IA. **POSE (Persona, Formato de Saída, Estilo, Exemplo):** Focado em atribuir um papel à IA e fornecer um formato de resposta claro e exemplos. **Sandwich (Iterativo/Defensivo):** Enfatiza a iteração (Rascunho -> Feedback -> Refinamento) e a robustez do prompt (Instrução Inicial -> Conteúdo -> Instrução Final). Capacidade de reduzir alucinações e melhorar a precisão na extração de informações acadêmicas.

## Use Cases

1. **Estruturação de Trabalhos Acadêmicos:** Criação de esboços, sumários e planos de pesquisa para artigos, teses e dissertações. 2. **Extração de Dados Estruturados:** Coleta e organização de informações específicas (métricas, metodologias, resultados) de múltiplos resumos ou artigos. 3. **Análise Crítica e Identificação de Lacunas:** Geração de seções que avaliam criticamente a literatura existente e apontam áreas para pesquisa futura. 4. **Refinamento de Escrita:** Ajuste do estilo, tom e formalidade do texto para atender aos padrões de publicação acadêmica.

## Integration

**Exemplo de Prompt (CO-STAR):** "Contexto: Sou um estudante de mestrado escrevendo uma revisão de literatura sobre 'O impacto da IA na educação superior'. Objetivo: Gerar um esboço detalhado da revisão. Estilo: Acadêmico e formal. Tom: Neutro e analítico. Público: Professores e pares. Resposta: Apresente o esboço em formato Markdown com 5 seções principais. Inclua subseções para lacunas de pesquisa e direções futuras." **Melhores Práticas:** 1. **Definir o Papel (POSE):** Comece o prompt com "Aja como um pesquisador sênior em [sua área]". 2. **Fornecer Exemplos (POSE):** Inclua um pequeno trecho de texto e o formato de saída desejado. 3. **Iterar (Sandwich):** Use o output inicial como rascunho e forneça feedback específico para refinar a análise crítica e a originalidade. 4. **Usar CoT (Chain-of-Thought):** Peça à IA para "Pensar passo a passo" antes de fornecer a resposta final.

## URL

https://arxiv.org/abs/2509.01128