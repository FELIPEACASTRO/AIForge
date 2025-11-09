# Prompts para Design de Ensaios Clínicos (Clinical Trial Design Prompts)

## Description
**Prompts de Design de Ensaios Clínicos** são instruções especializadas e estruturadas, frequentemente utilizando Large Language Models (LLMs), para auxiliar na criação, otimização e análise de protocolos de pesquisa clínica. Esta técnica de Prompt Engineering é aplicada no domínio da saúde e ciências da vida para automatizar tarefas complexas como a definição de critérios de elegibilidade, cálculo de tamanho amostral, seleção de desfechos (endpoints) e identificação de riscos éticos e operacionais [1] [3].

O objetivo principal é aumentar a eficiência, reduzir o tempo de desenvolvimento do protocolo e melhorar a qualidade científica e a conformidade regulatória dos ensaios clínicos. Ao fornecer contexto clínico e estatístico detalhado, os prompts guiam a IA para gerar seções de protocolo que são coerentes, baseadas em evidências e alinhadas com as melhores práticas internacionais [2]. A pesquisa recente (2023-2025) demonstra um foco crescente no uso de LLMs para extração de elementos PICO, otimização de critérios de elegibilidade e geração de versões simplificadas de consentimento informado [1].

## Examples
```
**1. Otimização de Critérios de Elegibilidade (Few-Shot Prompting)**

```
**Papel:** Você é um especialista em recrutamento de pacientes para ensaios de Oncologia de Fase II.
**Contexto:** Estamos desenhando um ensaio para um novo inibidor de PD-1 (Intervenção) em pacientes com Melanoma Metastático (População).
**Instrução:** Analise os seguintes critérios de inclusão/exclusão (Critérios Atuais) e sugira modificações para otimizar o recrutamento em 20%, mantendo a validade científica. Justifique cada alteração com base em dados de recrutamento de ensaios de PD-1 publicados nos últimos 3 anos.
**Critérios Atuais:** [Lista de critérios atuais]
**Formato de Saída:** Tabela com Colunas: Critério Original, Modificação Sugerida, Justificativa/Referência.
```

**2. Geração de Seção de Desfechos (Endpoints) Primários e Secundários**

```
**Papel:** Bioestatístico sênior.
**Contexto:** Ensaio de Fase III, randomizado, duplo-cego, controlado por placebo, para um medicamento que visa reduzir a progressão da Doença de Alzheimer leve a moderada.
**Instrução:** Proponha os desfechos primários e secundários mais apropriados para este ensaio. Para o desfecho primário, especifique a métrica de avaliação (ex: CDR-SB, ADAS-Cog), o ponto de tempo (ex: 52 semanas) e o método de análise estatística (ex: ANCOVA).
**Desfechos Primários Sugeridos:**
**Desfechos Secundários Sugeridos:**
**Requisito:** A resposta deve estar em conformidade com as diretrizes da FDA para ensaios de Alzheimer.
```

**3. Cálculo de Tamanho Amostral (Chain-of-Thought)**

```
**Papel:** Estatístico de ensaios clínicos.
**Contexto:** Ensaio de não-inferioridade comparando um novo antibiótico oral com o padrão-ouro para infecções do trato urinário.
**Instrução:** Calcule o tamanho amostral necessário. Use o seguinte processo (Chain-of-Thought):
1. Defina a margem de não-inferioridade (delta) clinicamente aceitável (justifique).
2. Estime a taxa de sucesso do tratamento padrão (padrão-ouro) com base em literatura (cite a fonte).
3. Defina o poder estatístico (ex: 80%) e o nível de significância (alfa = 0.05).
4. Apresente a fórmula utilizada e o cálculo final do tamanho amostral (N) por grupo.
5. Inclua uma taxa de abandono de 15% e recalcule o N final.
```

**4. Análise de Risco e Mitigação Operacional**

```
**Papel:** Gerente de Projetos Clínicos (CPM).
**Contexto:** Protocolo de ensaio clínico para terapia celular avançada que requer coleta e processamento de amostras complexas em 10 centros na Europa.
**Instrução:** Identifique os 5 principais riscos operacionais (ex: logística de amostras, treinamento de pessoal, conformidade regulatória local) e proponha uma estratégia de mitigação detalhada para cada um.
**Formato de Saída:** Tabela de Risco (Risco, Probabilidade (Alta/Média/Baixa), Impacto (Alto/Médio/Baixo), Estratégia de Mitigação).
```

**5. Geração de Versão Simplificada de Consentimento Informado**

```
**Papel:** Especialista em comunicação com o paciente.
**Contexto:** O texto a seguir é a seção de "Riscos e Efeitos Colaterais" de um Formulário de Consentimento Informado (FCI) para um ensaio de vacina.
**Instrução:** Reescreva esta seção em linguagem de 8ª série (nível de leitura simplificado), mantendo a precisão médica e legal. Use analogias simples para explicar termos complexos como "eventos adversos graves" e "randomização".
**Texto Original:** [Inserir texto complexo do FCI]
**Requisito:** A versão simplificada deve ter um índice de Flesch-Kincaid de 60 ou superior.
```

**6. Extração de Elementos PICO de Artigo Científico**

```
**Papel:** Analista de Evidências.
**Contexto:** [Inserir resumo ou seção de Métodos de um artigo de ensaio clínico]
**Instrução:** Extraia e estruture os elementos PICO (População, Intervenção, Comparação, Desfecho) deste texto.
**Formato de Saída:**
- População (P): [Detalhes]
- Intervenção (I): [Detalhes]
- Comparação (C): [Detalhes]
- Desfecho (O): [Detalhes]
```
```

## Best Practices
**1. Contextualização Clínica e de Papel (Role-Playing):** Comece o prompt definindo o papel da IA (ex: "Você é um bioestatístico sênior especializado em ensaios de fase III") e o contexto clínico (ex: "Desenho de um ensaio para um novo inibidor de SGLT2 para insuficiência cardíaca").
**2. Estrutura PICO/PICOTS:** Utilize frameworks de pesquisa (População, Intervenção, Comparação, Desfecho, Tempo, Setting) para garantir que todos os elementos críticos do protocolo sejam abordados.
**3. Especificidade e Clareza:** Use linguagem médica e estatística precisa. Evite termos vagos. Por exemplo, em vez de "melhorar o recrutamento", use "Otimizar os critérios de inclusão/exclusão para aumentar a taxa de recrutamento em 15% sem comprometer a validade interna".
**4. Iteração e Refinamento (Few-Shot/Chain-of-Thought):** Use prompts de múltiplas etapas. Peça à IA para primeiro delinear a estrutura (Chain-of-Thought) e depois preencher os detalhes. Use exemplos de protocolos bem-sucedidos (Few-Shot) para guiar a resposta.
**5. Validação e Referência:** Peça à IA para citar diretrizes regulatórias (ex: ICH-GCP, FDA) ou artigos de referência para as decisões de design propostas, permitindo a validação humana.
**6. Geração de Saída Estruturada:** Solicite a saída em um formato estruturado (ex: Markdown, JSON, tabela) para facilitar a integração em documentos de protocolo [1] [2].

## Use Cases
**1. Otimização de Protocolos:** Geração de rascunhos de seções de protocolo (ex: Desfechos, Critérios de Elegibilidade, Plano de Análise Estatística) para acelerar a fase de *design* do estudo.
**2. Análise de Viabilidade:** Avaliação rápida da complexidade e do potencial de recrutamento de um protocolo, analisando a literatura e dados de ensaios anteriores.
**3. Conformidade Regulatória:** Geração de listas de verificação (checklists) de conformidade com as diretrizes ICH-GCP, FDA ou EMA, garantindo que o protocolo aborde todos os requisitos legais e éticos.
**4. Comunicação com o Paciente:** Simplificação de documentos complexos, como o Formulário de Consentimento Informado (FCI), para melhorar a compreensão e o engajamento dos participantes [1].
**5. Síntese de Evidências:** Extração estruturada de dados (PICO) de artigos científicos para apoiar a justificativa científica do novo ensaio [1].
**6. Treinamento e Educação:** Criação de cenários de ensaios clínicos para treinamento de novos pesquisadores e coordenadores de estudo.

## Pitfalls
**1. Alucinações e Inconsistência Factual:** A IA pode gerar informações que parecem plausíveis, mas são clinicamente ou regulatoriamente incorretas (alucinações). Isso é crítico em saúde, onde a precisão é vital. **Mitigação:** Sempre validar a saída com diretrizes oficiais (ICH-GCP, FDA, EMA) e revisão por especialistas humanos [3].
**2. Viés e Falta de Diversidade:** A IA pode perpetuar vieses presentes nos dados de treinamento, resultando em critérios de elegibilidade que excluem indevidamente populações minoritárias ou sub-representadas. **Mitigação:** Incluir no prompt uma instrução explícita para considerar a diversidade e a equidade (ex: "Garantir que os critérios de elegibilidade não excluam desnecessariamente minorias étnicas ou de gênero").
**3. Falha em Capturar Nuances Clínicas:** LLMs podem ter dificuldade em integrar nuances complexas de doenças raras ou interações medicamentosas específicas. **Mitigação:** Fornecer contexto clínico extremamente detalhado e usar a técnica de *Retrieval-Augmented Generation (RAG)*, alimentando a IA com documentos de referência específicos do ensaio [2].
**4. Dependência Excessiva:** A confiança cega na saída da IA, sem a devida revisão e *due diligence* humana, pode levar a erros graves no protocolo que comprometem a segurança do paciente e a validade do estudo. **Mitigação:** Tratar a saída da IA como um rascunho avançado ou uma sugestão, e não como um documento final [3].
**5. Inconsistência de Formato:** A IA pode falhar em aderir a formatos regulatórios estritos (ex: *Clinical Study Protocol* - CSP). **Mitigação:** Usar prompts de formatação rigorosos (ex: "Estruturar a saída no formato de seção 3.1 do ICH E6(R2)") [1].

## URL
[https://bmcmedicine.biomedcentral.com/articles/10.1186/s12916-025-04348-9](https://bmcmedicine.biomedcentral.com/articles/10.1186/s12916-025-04348-9)
