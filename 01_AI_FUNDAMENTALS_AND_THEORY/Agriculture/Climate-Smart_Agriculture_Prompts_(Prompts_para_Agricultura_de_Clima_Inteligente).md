# Climate-Smart Agriculture Prompts (Prompts para Agricultura de Clima Inteligente)

## Description
**Prompts para Agricultura de Clima Inteligente (Climate-Smart Agriculture - CSA)** são instruções de engenharia de prompt desenhadas para alavancar Modelos de Linguagem Grande (LLMs) e outras IAs generativas na solução de desafios complexos da agricultura moderna, que está sob crescente pressão das mudanças climáticas. O objetivo principal é otimizar as práticas agrícolas para alcançar a **segurança alimentar** e o **desenvolvimento sustentável** em um clima em constante mudança.

Estes prompts são caracterizados pela sua necessidade de **contexto altamente específico** (dados de solo, clima, cultura, localização) e pela solicitação de **análises preditivas e prescritivas** que abordam os três pilares da CSA:

1.  **Aumento Sustentável da Produtividade e Renda:** Otimização de insumos, irrigação e manejo de culturas.
2.  **Adaptação e Construção de Resiliência:** Seleção de culturas resistentes à seca/calor, manejo de água e solo.
3.  **Redução/Remoção de Gases de Efeito Estufa (Mitigação):** Estratégias de sequestro de carbono no solo (Carbon Farming) e redução de emissões.

A eficácia desses prompts reside na capacidade de transformar dados complexos de sensoriamento remoto, meteorologia e análise de solo em **recomendações acionáveis** para o agricultor, promovendo a tomada de decisão baseada em evidências para um futuro agrícola mais resiliente e sustentável.

## Examples
```
**1. Otimização de Irrigação e Resiliência Hídrica**
```
Aja como um especialista em resiliência hídrica. Minha fazenda de milho (100 hectares, solo argiloso, coordenadas [LAT, LONG]) está prevista para enfrentar uma seca de 30 dias. Com base nos dados de umidade do solo (média de 45% na zona radicular) e na evapotranspiração histórica (5mm/dia), forneça um cronograma de irrigação de emergência (gotejamento) para maximizar a sobrevivência da cultura, minimizando o uso de água. Apresente a resposta em uma tabela com 'Dia', 'Volume de Água (m³/ha)' e 'Justificativa'.
```

**2. Plano de Sequestro de Carbono (Carbon Farming)**
```
Assuma o papel de um consultor de Carbon Farming. Desenvolva um plano de 5 anos para maximizar o sequestro de carbono em uma fazenda de 500 hectares de gado de corte (pastagem degradada, clima semiárido). O plano deve incluir a implementação de pastoreio rotacionado de alta densidade, plantio de culturas de cobertura (espécies recomendadas) e técnicas de plantio direto. Estime o potencial de captura de CO2e por hectare/ano e sugira métricas de monitoramento.
```

**3. Adaptação de Culturas a Eventos Climáticos Extremos**
```
Sou um agricultor na região de [Região/Estado] e o modelo climático prevê um aumento de 2°C na temperatura média e 15% de redução na precipitação nos próximos 10 anos. Minha cultura atual é [Cultura Atual]. Recomende 3 culturas alternativas ou variedades geneticamente adaptadas que demonstrem maior tolerância ao estresse hídrico e térmico. Para cada recomendação, liste os requisitos de solo e o potencial de mercado.
```

**4. Manejo Integrado de Pragas e Doenças (IPM) com Foco Climático**
```
Aja como um fitopatologista. O aumento da umidade e temperatura devido a um evento El Niño está elevando o risco de [Nome da Doença/Praga] na minha plantação de [Cultura]. Descreva um protocolo de Manejo Integrado de Pragas (MIP) preventivo e de baixo impacto ambiental. O protocolo deve priorizar o controle biológico e o uso de defensivos naturais, com um plano de ação para a primeira semana.
```

**5. Otimização de Uso de Fertilizantes para Mitigação de N2O**
```
Com base na análise de solo (pH 5.5, N total 0.1%, Matéria Orgânica 2.5%) para uma lavoura de soja, e buscando reduzir as emissões de Óxido Nitroso (N2O), um potente GEE, sugira uma estratégia de aplicação de fertilizantes nitrogenados. Inclua a forma de nitrogênio mais eficiente, o momento ideal de aplicação (timing) e a taxa de aplicação (kg/ha), justificando como a prática contribui para a mitigação climática.
```

**6. Análise de Viabilidade de Energia Renovável na Fazenda**
```
Avalie a viabilidade de instalar um sistema de energia solar fotovoltaica em minha fazenda de 200 hectares ([LAT, LONG]). Meu consumo médio mensal é de 5.000 kWh. Forneça uma estimativa do tamanho do sistema necessário (kWp), o custo aproximado de investimento e o tempo de retorno (payback) esperado, considerando os incentivos fiscais atuais para Agricultura de Clima Inteligente.
```

**7. Desenvolvimento de um Sistema de Alerta Precoce**
```
Crie um prompt para um modelo de IA que monitore dados de satélite (NDVI, EVI) e dados meteorológicos (temperatura, precipitação) para gerar um alerta precoce de estresse hídrico ou nutricional em uma área de 50 hectares de cana-de-açúcar. O alerta deve ser acionado quando o NDVI cair 10% abaixo da média histórica e a precipitação acumulada for 20% abaixo do esperado para o mês. Defina os parâmetros de entrada e o formato de saída do alerta.
```
```

## Best Practices
**1. Especificidade e Contexto Geográfico:** Sempre inclua dados específicos como tipo de solo, cultura, clima local (temperatura, precipitação histórica), e coordenadas geográficas. A Agricultura de Clima Inteligente é inerentemente local. **2. Definição de Papel (Role-Playing):** Comece o prompt definindo o papel da IA (ex: "Aja como um agrônomo especialista em sequestro de carbono" ou "Simule ser um especialista em resiliência hídrica"). **3. Inclusão de Dados de Entrada:** Forneça dados brutos ou sumarizados (ex: resultados de análise de solo, dados de sensoriamento remoto, histórico de pragas) para que a IA possa realizar análises preditivas e prescritivas. **4. Foco em Soluções Triplas (CSA):** Direcione o prompt para abordar os três pilares da CSA: produtividade, adaptação/resiliência e mitigação (redução de emissões/sequestro de carbono). **5. Formato de Saída Estruturado:** Peça a resposta em um formato específico (ex: tabela, lista numerada, plano de ação passo a passo) para facilitar a aplicação prática no campo.

## Use Cases
**1. Otimização de Insumos:** Determinar a quantidade ideal de fertilizantes e pesticidas com base em dados de solo e previsão climática, reduzindo custos e minimizando a poluição ambiental. **2. Planejamento de Rotação de Culturas e Agrofloresta:** Desenvolver planos de longo prazo que aumentem a matéria orgânica do solo (sequestro de carbono) e a resiliência a pragas e doenças. **3. Gestão de Riscos Climáticos:** Criar estratégias de adaptação para eventos climáticos extremos (secas, inundações, ondas de calor), incluindo a seleção de variedades de culturas mais resistentes. **4. Certificação e Mercados de Carbono:** Gerar relatórios e planos de monitoramento para que os agricultores possam participar de programas de crédito de carbono, quantificando o CO2 sequestrado. **5. Treinamento e Extensão Rural:** Criar materiais didáticos e guias de melhores práticas (em linguagem acessível) para disseminar técnicas de CSA entre comunidades agrícolas. **6. Monitoramento de Saúde do Solo:** Analisar dados de sensores e imagens de satélite para diagnosticar deficiências nutricionais ou estresse hídrico em tempo real, permitindo intervenções precisas.

## Pitfalls
**1. Falta de Especificidade Geográfica:** Usar prompts genéricos sem incluir dados de solo, clima e localização exatos. A CSA exige recomendações hiperlocalizadas. **2. Confiança Excessiva em Dados de Entrada:** Assumir que a IA pode compensar a falta de dados de qualidade (Garbage In, Garbage Out). A precisão da saída depende da precisão da análise de solo, sensoriamento remoto e dados meteorológicos fornecidos. **3. Ignorar o Contexto Socioeconômico:** Focar apenas na otimização técnica sem considerar a capacidade financeira do agricultor, a disponibilidade de mão de obra ou as cadeias de suprimentos locais. **4. Solicitar Saídas Não Acionáveis:** Pedir análises teóricas em vez de planos de ação concretos e implementáveis (ex: "Fale sobre CSA" vs. "Crie um plano de rotação de culturas para o próximo ciclo"). **5. Viés de Modelo:** A IA pode favorecer soluções de alta tecnologia (ex: drones, IoT) que podem não ser acessíveis ou apropriadas para agricultores de pequena escala, a menos que o prompt especifique a restrição de recursos.

## URL
[https://weam.ai/blog/prompts/chatgpt-prompts-for-farming/](https://weam.ai/blog/prompts/chatgpt-prompts-for-farming/)
