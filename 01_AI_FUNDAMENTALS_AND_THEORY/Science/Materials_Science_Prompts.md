# Materials Science Prompts

## Description
**Prompts de Ciência de Materiais** referem-se à aplicação da **Engenharia de Prompt** (Prompt Engineering) para interagir com Grandes Modelos de Linguagem (LLMs) no contexto da pesquisa, descoberta e design de novos materiais. Esta técnica é fundamental para superar a limitação de que os LLMs não são, por natureza, modelos de domínio científico. O objetivo é guiar o LLM a extrair, sintetizar, classificar e gerar informações precisas e relevantes a partir de vastos corpos de literatura científica e bases de dados de materiais.

A principal função desses prompts é atuar como uma ponte entre o conhecimento linguístico geral do LLM e o conhecimento técnico especializado da Ciência de Materiais. Isso é feito através da injeção de contexto de domínio, definição de persona (ex: "Atue como um químico de materiais"), e especificação rigorosa do formato de saída. Pesquisas recentes (2024-2025) demonstram que a engenharia de prompt, muitas vezes combinada com o ajuste fino (fine-tuning) em dados específicos de materiais, pode aumentar drasticamente a precisão na classificação de materiais, extração de parâmetros de síntese e até mesmo na geração de hipóteses para a descoberta de novos compostos [1] [2].

A técnica permite que pesquisadores acelerem tarefas como:
*   **Extração de Dados Estruturados:** Transformar textos não estruturados de artigos científicos em dados tabulares para treinamento de modelos de Machine Learning (ML) [2].
*   **Previsão e Classificação de Propriedades:** Usar o conhecimento do LLM para prever a viabilidade ou as propriedades de um material sob certas condições [1].
*   **Geração de Hipóteses:** Sugerir novas composições ou rotas de síntese para materiais com propriedades específicas [3].

## Examples
```
**1. Extração de Dados Estruturados (Síntese):**
`"Atue como um químico de materiais. Analise o seguinte resumo de artigo e extraia os parâmetros de síntese do material 'LiFePO4'. Retorne a 'Temperatura de Reação', 'Tempo de Reação', 'Precursores' e 'Atmosfera' em formato JSON."`

**2. Classificação de Materiais (Zero-Shot):**
`"Com base em suas propriedades eletrônicas e estrutura cristalina, classifique o material 'BaTiO3' como 'Condutor', 'Semicondutor' ou 'Isolante'. Justifique sua resposta em um parágrafo conciso."`

**3. Geração de Hipóteses (Design de Materiais):**
`"Eu preciso de um material com alta condutividade iônica (acima de 10^-3 S/cm) para aplicação em eletrólitos sólidos de baterias de estado sólido. Sugira 3 famílias de materiais (ex: Perovskitas, NASICONs, LISICONs) e, para cada uma, proponha uma composição específica e uma rota de síntese inicial. Use o formato de lista numerada."`

**4. Análise de Mecanismos de Falha:**
`"Descreva o mecanismo de corrosão sob tensão (SCC) em ligas de alumínio da série 7xxx. Quais são os principais fatores microestruturais que influenciam a suscetibilidade ao SCC? Responda em Português Brasileiro, focando na didática para um estudante de graduação."`

**5. Revisão de Literatura e Comparação:**
`"Crie uma tabela comparativa entre o Silício (Si) e o Arsenieto de Gálio (GaAs) para aplicação em células solares. As colunas devem ser: 'Gap de Energia (eV)', 'Mobilidade de Elétrons (cm²/Vs)', 'Absorção de Luz' e 'Custo Relativo'. Cite a fonte para os valores de Gap de Energia."`

**6. Otimização de Processo:**
`"Para a deposição de filmes finos de óxido de zinco (ZnO) via 'Sputtering por RF', quais parâmetros (Potência de RF, Pressão de Trabalho, Temperatura do Substrato) devo priorizar para maximizar a orientação cristalográfica (002)? Sugira um intervalo de valores para cada parâmetro."`
```

## Best Practices
**1. Especificidade e Contexto de Domínio:** Sempre inclua o máximo de detalhes científicos possível. Especifique o material, a estrutura (cristalina, amorfa), as condições de processamento (temperatura, pressão, atmosfera) e as propriedades desejadas (mecânicas, elétricas, ópticas). Use termos técnicos precisos (ex: "perovskita de haleto", "vidro metálico a granel").
**2. Estrutura de Saída Definida:** Peça explicitamente para o LLM formatar a saída em um formato estruturado, como JSON, tabela Markdown ou CSV. Isso facilita a extração e o uso posterior dos dados (ex: "Retorne os resultados em uma tabela com colunas: Material, Propriedade, Valor, Unidade").
**3. Cadeia de Raciocínio (Chain-of-Thought):** Para tarefas complexas como a previsão de síntese ou a análise de mecanismos de falha, instrua o modelo a detalhar seu processo de raciocínio passo a passo antes de fornecer a resposta final. Isso ajuda a identificar alucinações e a validar a lógica científica.
**4. Referência a Fontes Confiáveis:** Se o LLM tiver acesso a ferramentas de busca ou bases de dados, instrua-o a citar as fontes de onde extraiu a informação, especialmente para dados quantitativos ou descobertas recentes.
**5. Iteração e Refinamento:** Comece com um prompt amplo e refine-o com base nas deficiências da resposta inicial. Por exemplo, se a resposta for muito genérica, adicione restrições de material ou de aplicação.

## Use Cases
nan

## Pitfalls
**1. Alucinações de Dados Quantitativos:** LLMs podem gerar valores numéricos, composições químicas ou parâmetros de síntese que parecem plausíveis, mas são factualmente incorretos ou inexistentes na literatura. **Contramedida:** Sempre solicite a citação da fonte ou a verificação cruzada com bases de dados confiáveis.
**2. Falta de Conhecimento de Domínio Específico:** O LLM pode falhar em tarefas que exigem um raciocínio físico-químico profundo ou a interpretação de diagramas de fase complexos. **Contramedida:** Use prompts de "Chain-of-Thought" para forçar o raciocínio e forneça o máximo de contexto e restrições de domínio no prompt.
**3. Viés de Treinamento (Bias):** O modelo pode favorecer materiais ou rotas de síntese mais comuns na literatura, ignorando abordagens inovadoras ou menos publicadas. **Contramedida:** Peça explicitamente por "abordagens não convencionais" ou "materiais emergentes" para mitigar o viés.
**4. Ambiguidade na Terminologia:** Termos como "alta resistência" ou "bom isolante" são subjetivos. **Contramedida:** Substitua a linguagem vaga por critérios quantitativos e unidades de medida (ex: "resistência à tração > 500 MPa").

## URL
[https://www.nature.com/articles/s41524-025-01554-0](https://www.nature.com/articles/s41524-025-01554-0)
