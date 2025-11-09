# Astrophysics Prompts

## Description
O uso de Large Language Models (LLMs) em astrofísica e ciência espacial envolve a criação de prompts especializados para aproveitar as capacidades dos modelos em tarefas como análise de dados, modelagem teórica, geração de conteúdo educacional e comunicação científica. Esta técnica faz parte do campo mais amplo da "Astrocomputação" na era dos LLMs. Prompts eficazes frequentemente atribuem uma persona específica (por exemplo, "astrofísico experiente") para guiar o detalhe e o estilo da resposta, e utilizam técnicas avançadas como Chain-of-Thought (CoT) ou Few-Shot Learning para lidar com o raciocínio científico complexo e a interpretação de dados. O objetivo é transformar vastos conjuntos de dados astronômicos e teorias complexas em *insights* acionáveis ou explicações acessíveis. O sucesso depende da precisão do prompt em definir o contexto, o papel do modelo e o formato de saída esperado.

## Examples
```
**1. Análise de Dados e Interpretação de FITS:**
`Aja como um cientista de dados astronômicos. Analise o cabeçalho FITS a seguir e resuma os principais parâmetros observacionais (e.g., telescópio, data, exposição, objeto). Em seguida, sugira três potenciais vieses de observação. [INSERIR DADOS DO CABEÇALHO FITS]`

**2. Modelagem Teórica e Simulação:**
`Aja como um astrofísico teórico. Explique o processo de nucleossíntese estelar em estrelas de baixa massa (como o Sol). Em seguida, forneça o código Python para simular a curva de luminosidade de uma supernova Tipo Ia, utilizando a biblioteca Astropy.`

**3. Geração de Conteúdo Educacional:**
`Aja como um educador de ciências espaciais. Crie um prompt para um gerador de imagens que visualize a acreção de matéria em torno de um buraco negro de massa estelar, com um disco de acreção visível em raios-X. O tom deve ser didático e visualmente impactante.`

**4. Revisão e Síntese de Literatura:**
`Aja como um revisor de literatura. Sintetize os principais argumentos e descobertas dos últimos 5 anos sobre a controvérsia da Constante de Hubble, citando as fontes primárias (se possível) e destacando a diferença entre os métodos de medição de distância. Use o formato de resumo executivo.`

**5. Resolução de Problemas de Astrofísica:**
`Aja como um astrofísico experiente. Calcule a distância de um aglomerado globular cuja magnitude aparente média das estrelas RR Lyrae é m=18.5, sabendo que a magnitude absoluta média é M=0.6. Mostre o cálculo passo a passo (Chain-of-Thought) e expresse o resultado em parsecs e anos-luz.`

**6. Prompt para Pesquisa de Exoplanetas:**
`Aja como um pesquisador de exoplanetas. Dada a massa de uma estrela (0.8 M☉) e o período orbital de um exoplaneta (15 dias), calcule o raio orbital do exoplaneta em Unidades Astronômicas (UA). Assuma uma órbita circular e use a Terceira Lei de Kepler. Explique a relevância desse raio para a zona habitável.`

**7. Criação de Roteiro para Divulgação Científica:**
`Aja como um roteirista de documentários científicos. Crie um roteiro de 3 minutos para um vídeo sobre a formação da Via Láctea, focando na teoria do Big Bang e na matéria escura. O roteiro deve ser envolvente, acessível ao público leigo e incluir sugestões de imagens de arquivo (e.g., Hubble, JWST).`
```

## Best Practices
**1. Atribuição de Papel (Role Assignment):** Sempre comece o prompt atribuindo um papel específico e experiente ao LLM, como "Aja como um astrofísico especializado em buracos negros" ou "Aja como um cientista de dados astronômicos". Isso direciona o tom, o nível de detalhe e a precisão técnica da resposta.
**2. Contextualização de Dados:** Ao analisar dados (como FITS, CSV ou tabelas), inclua um trecho representativo dos dados ou o cabeçalho, e especifique o formato de saída desejado (e.g., JSON, Python Pandas DataFrame).
**3. Uso de Técnicas Avançadas:** Para problemas complexos de raciocínio científico ou modelagem, utilize técnicas como **Chain-of-Thought (CoT)**, pedindo ao LLM para "pensar passo a passo" antes de dar a resposta final, ou **Few-Shot Learning**, fornecendo exemplos de problemas e soluções.
**4. Especificidade e Limitação:** Seja o mais específico possível sobre o fenômeno, a teoria ou o objeto astronômico. Limite o escopo da resposta para evitar generalizações imprecisas.
**5. Verificação Cruzada:** Sempre trate a saída do LLM como um ponto de partida ou uma hipótese. A complexidade e a natureza crítica dos dados astrofísicos exigem **verificação cruzada** com fontes primárias, simulações e softwares científicos dedicados.

## Use Cases
**1. Análise Preliminar de Dados:** Auxiliar astrofísicos na interpretação rápida de cabeçalhos FITS, logs de observação ou resultados de simulações, identificando parâmetros-chave e potenciais anomalias.
**2. Geração de Hipóteses:** Utilizar o LLM para explorar rapidamente as implicações de novas descobertas ou dados, gerando hipóteses testáveis para a pesquisa.
**3. Educação e Divulgação Científica:** Criar materiais didáticos, resumos de artigos complexos ou roteiros de vídeos, traduzindo conceitos astrofísicos complexos para uma linguagem acessível a diferentes públicos.
**4. Revisão de Código e Documentação:** Gerar documentação para códigos de simulação (e.g., Fortran, Python) ou revisar trechos de código para otimização e correção de *bugs* em rotinas de análise de dados.
**5. Síntese de Literatura:** Realizar a mineração e síntese de grandes volumes de artigos científicos (se o LLM tiver acesso a essa base de dados), identificando tendências e lacunas na pesquisa atual.
**6. Modelagem Conceitual:** Auxiliar na formulação de modelos conceituais para fenômenos astrofísicos, como a evolução de galáxias ou a dinâmica de sistemas estelares, antes de iniciar simulações computacionais intensivas.

## Pitfalls
**1. Alucinações Científicas:** O LLM pode gerar informações factualmente incorretas ou teorias obsoletas. **Armadilha:** Confiar cegamente na saída sem verificação cruzada.
**2. Falta de Contexto de Dados:** Fornecer dados brutos sem especificar o formato, as unidades ou o contexto observacional. **Armadilha:** O LLM pode interpretar mal os valores ou aplicar fórmulas incorretas.
**3. Prompts Vagos:** Pedir "fale sobre buracos negros" sem um objetivo claro. **Armadilha:** Receber uma resposta genérica e inútil para pesquisa ou educação especializada.
**4. Limitações de Cálculo:** Embora LLMs possam realizar cálculos, eles não são calculadoras. **Armadilha:** Usar o LLM para cálculos complexos que exigem alta precisão e que são melhor executados por software científico (e.g., NumPy, Astropy).
**5. Confusão de Unidades:** A astrofísica lida com unidades complexas (parsecs, anos-luz, magnitudes, etc.). **Armadilha:** Não especificar as unidades de entrada e saída, levando a erros de ordem de magnitude.

## URL
[https://www.nature.com/articles/s41597-025-04613-9](https://www.nature.com/articles/s41597-025-04613-9)
