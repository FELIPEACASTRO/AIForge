# Named Entity Recognition Prompts

## Description
Prompts de Reconhecimento de Entidades Nomeadas (REN) são técnicas de Engenharia de Prompt focadas em guiar Modelos de Linguagem de Grande Escala (LLMs) para identificar e classificar entidades nomeadas (como pessoas, organizações, locais, datas, etc.) em um texto não estruturado. Ao invés de depender de modelos de Machine Learning (ML) tradicionais treinados com grandes volumes de dados rotulados, a abordagem de prompt utiliza a capacidade de raciocínio e compreensão de contexto dos LLMs para realizar a tarefa de REN, muitas vezes especificando o formato de saída desejado (e.g., JSON, XML ou notação BIO). Esta técnica aproveita a capacidade de raciocínio e o conhecimento inerente dos LLMs para realizar tarefas de extração de informação com alta precisão e flexibilidade, sendo uma alternativa poderosa aos modelos de REN tradicionais.

## Examples
```
1. **Zero-Shot Simples (Extração Geral)**
   ```
   Instrução: Extraia todas as entidades nomeadas (PESSOA, ORGANIZAÇÃO, LOCAL) do texto a seguir e liste-as no formato: [Entidade]: [Tipo].
   Texto: "A Dra. Ana Silva, da Universidade de São Paulo (USP), apresentou sua pesquisa em Berlim na semana passada."
   ```

2. **Few-Shot para Domínio Financeiro**
   ```
   Instrução: Você é um analista financeiro. Extraia as entidades (EMPRESA, VALOR, MOEDA, DATA) do texto.
   Exemplo 1: "A Petrobras (EMPRESA) anunciou um lucro de 10 bilhões (VALOR) de reais (MOEDA) no terceiro trimestre de 2023 (DATA)."
   Exemplo 2: "A Apple (EMPRESA) atingiu um valor de mercado de 3 trilhões (VALOR) de dólares (MOEDA) em janeiro de 2024 (DATA)."
   Texto: "O Banco do Brasil reportou um crescimento de 5% em seu balanço de 2024, totalizando 15,5 bilhões de reais."
   ```

3. **Extração Estruturada (JSON)**
   ```
   Instrução: Extraia as entidades (PRODUTO, QUANTIDADE, UNIDADE) da lista de compras e retorne a saída estritamente no formato JSON, seguindo o esquema: [{"entidade": "...", "tipo": "...", "valor": "..."}].
   Texto: "Comprar 3 quilos de arroz, 1 dúzia de ovos e 500 gramas de queijo mussarela."
   ```

4. **Simulação de Function Calling (System Prompt)**
   ```
   System Prompt:
   Você é um assistente de extração de dados. Sua única função é chamar a ferramenta `extract_clinical_entities` com os argumentos corretos.
   Ferramenta: `extract_clinical_entities(doenca: str, sintoma: str, medicamento: str)`
   
   User Prompt:
   "O paciente João foi diagnosticado com pneumonia e está tomando Amoxicilina para tratar a febre e a tosse persistente."
   ```

5. **Prompt de Domínio Específico (Jurídico)**
   ```
   Instrução: Identifique e classifique as entidades (PARTE, TRIBUNAL, LEI, DATA) no trecho legal a seguir. Seja preciso.
   Texto: "A decisão foi proferida pelo Supremo Tribunal Federal (STF) em 15 de maio de 2024, em favor da Requerente Maria da Penha, com base no Artigo 5º da Constituição Federal."
   ```

6. **Chain-of-Thought (CoT) para Desambiguação**
   ```
   Instrução: Analise o texto e extraia as entidades (PESSOA, LOCAL). Antes de fornecer a resposta final, use o raciocínio CoT para justificar a classificação de entidades ambíguas.
   Texto: "Paris, a capital da França, é um nome comum. Paris Hilton, por outro lado, é uma celebridade."
   ```

7. **Extração com Notação BIO**
   ```
   Instrução: Realize o Reconhecimento de Entidades Nomeadas (REN) no texto e use a notação BIO (B-TIPO, I-TIPO, O) para marcar cada token.
   Texto: "Steve Jobs fundou a Apple em Cupertino."
   Saída Esperada: "Steve [B-PESSOA] Jobs [I-PESSOA] fundou [O] a [O] Apple [B-ORGANIZAÇÃO] em [O] Cupertino [B-LOCAL] ."
   ```
```

## Best Practices
1. **Determinismo (Temperatura e Seed):** Use `temperature=0.0` e defina um `seed` (se suportado pela API) para obter resultados mais determinísticos e reprodutíveis, cruciais para tarefas de extração de dados.
2. **Instruções Claras e Papel:** Defina um papel claro para o LLM (ex: "Você é um assistente de IA especialista em REN") e use instruções explícitas sobre a tarefa, o formato de saída e as categorias de entidades a serem extraídas.
3. **Few-Shot Learning:** Forneça exemplos de entrada e saída (Few-Shot Prompting) para demonstrar o formato e o tipo de entidades esperadas, melhorando a precisão e a aderência ao esquema.
4. **Funções/Tools (JSON Schema):** Utilize a funcionalidade de chamada de função (Function Calling) ou forneça um JSON Schema detalhado para forçar o modelo a retornar a saída em um formato JSON válido e estruturado, ideal para integração em pipelines de dados.
5. **Prompts de Domínio:** Adapte os prompts para o domínio específico (ex: médico, financeiro, culinário) para melhorar a precisão na identificação de entidades contextuais.
6. **Chain-of-Thought (CoT):** Peça ao modelo para "pensar em voz alta" ou justificar suas extrações antes de fornecer a saída final, o que pode aumentar a precisão em casos complexos.
7. **Prompt Chaining:** Divida a tarefa de REN em etapas menores (ex: 1. Identificar o limite da entidade, 2. Classificar a entidade) para melhorar a performance.

## Use Cases
1. **Extração de Dados de Documentos:** Automatizar a extração de informações-chave (nomes de partes, datas, valores) de contratos, faturas, relatórios financeiros e documentos legais.
2. **Análise de Mídias Sociais:** Identificar menções a marcas, produtos, pessoas e locais em grandes volumes de texto de redes sociais para monitoramento de marca e análise de sentimento.
3. **Biomedicina e Saúde:** Extrair nomes de doenças, medicamentos, sintomas e procedimentos de prontuários médicos e artigos científicos.
4. **Notícias e Jornalismo:** Resumir e categorizar artigos de notícias, identificando rapidamente os principais atores (pessoas, organizações) e locais.
5. **E-commerce e Catálogos:** Extrair atributos de produtos (marca, modelo, cor, tamanho) de descrições de texto para enriquecimento de catálogo.

## Pitfalls
1. **Alucinações e Imprecisão:** LLMs podem "alucinar" entidades ou classificá-las incorretamente, especialmente em domínios de nicho ou com prompts ambíguos.
2. **Inconsistência de Formato:** Sem um JSON Schema rigoroso ou Function Calling, o modelo pode falhar em retornar o formato de saída exato solicitado, dificultando o processamento downstream.
3. **Custo e Latência:** O uso de LLMs para REN pode ser mais caro e lento do que modelos de ML tradicionais otimizados, especialmente para grandes volumes de dados.
4. **Dependência de Contexto:** A precisão pode cair se o texto de entrada for muito longo e as informações contextuais necessárias para a classificação da entidade estiverem fora da janela de contexto do modelo.
5. **Vieses do Modelo:** O modelo pode refletir vieses presentes em seus dados de treinamento, levando a classificações tendenciosas ou incompletas.

## URL
[https://dswithmac.com/posts/prompt-eng-ner/](https://dswithmac.com/posts/prompt-eng-ner/)
