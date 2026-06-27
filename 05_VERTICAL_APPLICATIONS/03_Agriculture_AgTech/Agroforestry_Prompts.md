# Agroforestry Prompts

## Description
"Agroforestry Prompts" são instruções de engenharia de prompt (prompt engineering) especificamente formuladas para interagir com Modelos de Linguagem Grande (LLMs) e sistemas de Inteligência Artificial (IA) com o objetivo de obter informações, análises, recomendações e planos de ação relacionados a Sistemas Agroflorestais (SAFs). Os SAFs são sistemas de uso da terra que integram árvores com culturas agrícolas e/ou pecuária de forma espacial e temporal, buscando benefícios ecológicos e econômicos. Os prompts são projetados para lidar com a complexidade estrutural e a diversidade de espécies inerentes aos SAFs, solicitando análises de dados multimodais (sensoriamento remoto, dados de solo, clima) e a geração de conhecimento prático.

## Examples
```
1. **Planejamento de SAF:** "Desenvolva um plano de implementação de um Sistema Agroflorestal sucessional para uma área de 5 hectares no bioma [Nome do Bioma, ex: Mata Atlântica]. O foco principal é a produção de [Cultura Principal, ex: cacau] e [Espécie de Árvore, ex: ingá]. Inclua a lista de espécies pioneiras, secundárias e clímax, o espaçamento recomendado e um cronograma de manejo para os primeiros 5 anos."
2. **Análise de Dados:** "Analise os seguintes dados de sensoriamento remoto (NDVI e temperatura de superfície) da minha área agroflorestal (anexar dados ou descrever o período) e identifique áreas com estresse hídrico ou nutricional. Sugira intervenções de manejo específicas para as coordenadas [X, Y]."
3. **Otimização de Espécies:** "Para um SAF com culturas de [Culturas, ex: café e banana] e árvores de sombra de [Espécies de Sombra, ex: eritrina], qual é o arranjo espacial ideal para maximizar a interceptação de luz pela cultura principal sem comprometer o crescimento das árvores? Apresente a resposta em um formato de tabela comparativa."
4. **Manejo de Pragas e Doenças:** "Com base nos sintomas de [Sintoma, ex: amarelecimento das folhas e presença de cochonilhas] na cultura de [Cultura, ex: mandioca] dentro do meu SAF, forneça um protocolo de Manejo Integrado de Pragas (MIP) que utilize apenas métodos biológicos e naturais, sem o uso de agroquímicos."
5. **Recomendação de Colheita:** "Considerando a previsão climática para os próximos 30 dias na região de [Nome da Região] e o estágio de maturação da cultura de [Cultura, ex: açaí], qual é a janela de colheita ideal para maximizar o rendimento e a qualidade do produto? Justifique a resposta com base em dados de umidade e temperatura."
6. **Cálculo de Sequestro de Carbono:** "Calcule o potencial de sequestro de carbono acima do solo (em toneladas de CO2 equivalente por hectare/ano) para um SAF maduro com densidade de [Número] árvores/hectare, composto pelas espécies [Espécies, ex: mogno, ipê e seringueira]. Cite a metodologia de cálculo utilizada."
7. **Integração com Pecuária (Silvipastoril):** "Desenhe um sistema silvipastoril para uma fazenda de gado de corte em [Região]. Recomende espécies forrageiras e arbóreas que ofereçam sombra e forragem suplementar, e detalhe a taxa de lotação animal sustentável para o sistema."
```

## Best Practices
* **Especificidade Contextual:** Incluir o máximo de detalhes sobre o contexto (bioma, tipo de solo, clima, culturas e espécies arbóreas existentes, objetivos do agricultor) para que o LLM possa fornecer recomendações hiper-localizadas e relevantes.
* **Solicitação de Dados Estruturados:** Pedir a saída em formatos estruturados (tabelas, listas, JSON) para facilitar a análise e a integração com outras ferramentas de gestão agrícola.
* **Foco Multimodal:** Em prompts avançados, referenciar a necessidade de análise de dados multimodais (imagens de satélite, dados de sensores de solo, modelos climáticos) para simular a capacidade de sistemas de IA mais complexos.
* **Ênfase em Sustentabilidade:** Direcionar o LLM para soluções que priorizem princípios agroecológicos, como biodiversidade, ciclagem de nutrientes e controle biológico de pragas.
* **Validação e Iteração:** Tratar a saída do LLM como uma recomendação inicial e usar prompts de acompanhamento para refinar a análise, questionar suposições e validar a viabilidade das sugestões.

## Use Cases
* **Planejamento e Design de SAFs:** Geração de layouts, seleção de espécies e cronogramas de plantio otimizados para diferentes condições edafoclimáticas e objetivos de produção.
* **Diagnóstico e Monitoramento:** Identificação precoce de estresse em plantas, doenças ou deficiências nutricionais através da análise de dados de sensoriamento remoto e sensores de campo.
* **Tomada de Decisão de Manejo:** Obtenção de recomendações sobre poda, irrigação, adubação e controle de pragas em tempo real.
* **Educação e Extensão Rural:** Criação de materiais didáticos, guias de melhores práticas e respostas a perguntas frequentes para agricultores e técnicos.
* **Pesquisa e Modelagem:** Geração de hipóteses, simulação de cenários de mudança climática e cálculo de serviços ecossistêmicos (ex: sequestro de carbono).

## Pitfalls
* **Simplificação Excessiva:** O LLM pode simplificar a complexidade inerente aos SAFs (interações entre espécies, microclimas), levando a recomendações genéricas ou inadequadas.
* **Viés de Dados:** A qualidade e o viés dos dados de treinamento do LLM podem resultar em recomendações que favorecem práticas de agricultura convencional em detrimento de abordagens agroecológicas.
* **Falta de Contexto Local:** Sem dados de entrada precisos sobre o local (solo, microclima), as sugestões podem ser impraticáveis ou ineficazes.
* **Alucinações Técnicas:** O LLM pode gerar nomes de espécies, métodos de manejo ou dados científicos que parecem plausíveis, mas são factualmente incorretos ou inexistentes.
* **Dependência de Dados Multimodais:** A eficácia dos prompts mais avançados depende da capacidade do LLM de processar e interpretar dados não textuais (imagens, mapas), o que nem sempre é garantido em modelos de acesso público.

## URL
[https://www.researchgate.net/publication/393576880_Artificial_Intelligence_for_Agroforestry_A_Review](https://www.researchgate.net/publication/393576880_Artificial_Intelligence_for_Agroforestry_A_Review)
