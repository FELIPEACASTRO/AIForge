# Prompts de Meteorologia (Meteorology Prompts)

## Description
Prompts de Meteorologia são técnicas avançadas de Engenharia de Prompt que utilizam Large Language Models (LLMs) para processar grandes volumes de dados meteorológicos brutos (como dados de modelos numéricos em formato CSV ou JSON) e gerar resumos de previsão do tempo estruturados, concisos e adaptados a um público-alvo específico. A técnica envolve a definição de um papel especializado para o LLM (ex: hidrometeorologista), a imposição de restrições estritas de formato e conteúdo (ex: limite de palavras, tom de risco, uso de markdown) e a integração de dados externos para garantir que a saída seja factualmente precisa e operacionalmente útil. O objetivo principal é transformar dados complexos e técnicos em comunicação clara e acionável, como alertas de inundação ou relatórios de voo [1].

## Examples
```
**Exemplo 1: Resumo de Previsão de Inundação (Baseado em [1])**

```
**Papel:** Você é um hidrometeorologista do Reino Unido preparando uma previsão focada em inundações para **{region_name}**.
**Tom:** Formal, conciso, consciente do risco (orientação de inundação).
**Restrição:** ≤ 300 palavras (clima + costeiro/maré).
**Dados de Entrada:**
## 2 Input Data (for model use only)
- **Weather model:** ECMWF IFS.
{csv_data_com_chuva_e_vento}
- **Tidal series (proxy):** {tidal_txt_com_ciclo_de_mare}
**Instruções:**
1. Comece com 3 frases: Período de cobertura, modelo base, e um título conciso (##### Título: ...).
2. Descreva as características por regiões amplas, não por locais específicos.
3. Foco principal em chuvas moderadas/fortes. Omita temperatura/ventos gerais, a menos que sejam críticos.
4. Inclua ###### Informações Costeiras/Marés. Destaque ventos costeiros fortes coincidindo com marés altas.
5. **Feche com (escolha 1):** "Nada atualmente indica que possa exacerbar ou criar novo risco de inundação." OU "Possibilidade de aumento do risco de inundação com base nas condições previstas."
```

**Exemplo 2: Relatório de Voo para Pilotos**

```
**Papel:** Você é um Oficial de Meteorologia Aeronáutica.
**Público:** Pilotos de aviação geral (VFR).
**Local:** Aeroporto de Congonhas (SBMT) e área de 50nm.
**Dados de Entrada:** METAR/TAF de SBMT, SBGR, SBKP. Imagem de satélite (link/descrição).
**Instruções:**
1. **Formato:** Resumo de 150 palavras. Use abreviações aeronáuticas padrão (ex: OVC, BKN, VMC, IFR).
2. **Foco:** Teto (Ceiling), Visibilidade, Vento (direção e rajadas), e Risco de Trovoadas (TS).
3. **Saída:** Inicie com "RESUMO METAR/TAF SBMT/50NM". Destaque condições IFR ou marginal VFR.
4. **Perigo:** Se houver TS previsto, inclua um alerta em caixa alta: "ALERTA: RISCO DE TROVOADAS ISOLADAS ENTRE 1500Z E 1800Z."
```

**Exemplo 3: Alerta de Onda de Calor para Saúde Pública**

```
**Papel:** Você é um analista de risco climático para a Secretaria de Saúde.
**Local:** Cidade de São Paulo.
**Dados de Entrada:** Previsão de temperatura máxima e umidade relativa para os próximos 5 dias. Limite de alerta: 32°C e UR < 30%.
**Instruções:**
1. **Tom:** Informativo e preventivo.
2. **Saída:** Crie um boletim de 100 palavras.
3. **Conteúdo:** Indique os dias em que o limite de alerta será excedido. Forneça 3 recomendações de saúde pública (ex: hidratação, horários de pico).
4. **Formato:** Use uma tabela simples para os dias de alerta.
```

**Exemplo 4: Análise de Condições para Agricultura**

```
**Papel:** Consultor Agrônomo.
**Cultura:** Soja (fase de enchimento de grãos).
**Local:** Região Oeste da Bahia.
**Dados de Entrada:** Previsão de precipitação acumulada (mm) e evapotranspiração (ET) para 7 dias.
**Instruções:**
1. **Análise:** Avalie o balanço hídrico.
2. **Saída:** Um parágrafo (máx. 80 palavras) sobre a adequação das condições para a fase da cultura.
3. **Recomendação:** Uma recomendação específica (ex: necessidade de irrigação suplementar ou risco de doenças fúngicas devido à umidade).
```

**Exemplo 5: Resumo de Condições de Surf**

```
**Papel:** Especialista em Previsão de Ondas.
**Local:** Praia de Maresias, SP.
**Dados de Entrada:** Previsão de altura de onda (m), período (s), direção do swell, e vento (direção/velocidade) para as próximas 24h (intervalos de 6h).
**Instruções:**
1. **Tom:** Entusiasta e técnico.
2. **Saída:** Um resumo por período (Manhã, Tarde, Noite).
3. **Foco:** Qualidade das ondas para surf (ex: "Condições Clássicas", "Mar mexido").
4. **Detalhe:** Inclua a melhor janela de maré para o surf.
```
```

## Best Practices
**Definição de Papel e Tom:** Atribua ao LLM um papel específico (ex: "hidrometeorologista do Reino Unido") e um tom (ex: "formal, conciso, consciente do risco") para garantir a terminologia e o estilo apropriados [1]. **Restrições de Saída Rígidas:** Use limites de palavras (ex: "≤ 300 palavras") e instruções de formatação (ex: "Use markdown ###### para títulos") para forçar a concisão e a estrutura [1]. **Contextualização Temporal:** Forneça datas de início e fim explícitas para o período de previsão (ex: "Hoje é {today_day} {today_date}. Dia final: {final_day_name}") para ancorar o LLM no tempo real [1]. **Foco na Relevância:** Priorize informações críticas para o caso de uso (ex: chuva moderada/forte para risco de inundação) e omita detalhes não essenciais (ex: temperatura ou ventos gerais, a menos que sejam críticos) [1]. **Estrutura de Dados Clara:** Apresente os dados de entrada (ex: CSV de dados de modelo, lista de locais amostrados, dados de maré) em seções claramente rotuladas para o "uso do modelo apenas" [1]. **Evitar Hipérbole e Extrapolação:** Instrua o LLM a evitar previsões espaciais excessivamente confiantes ou extrapolações (ex: "Evite previsões espaciais excessivamente confiantes") quando os dados de entrada são baseados em pontos [1].

## Use Cases
**Previsão de Risco de Inundação:** Geração de resumos de previsão do tempo focados em parâmetros relevantes para inundações (ex: taxas de precipitação, marés) para equipes de resposta a emergências [1]. **Relatórios Aeronáuticos:** Criação de boletins meteorológicos concisos e padronizados para pilotos, focando em visibilidade, teto e risco de trovoadas. **Boletins de Saúde Pública:** Geração de alertas de ondas de calor, frio extremo ou qualidade do ar, com recomendações de saúde pública. **Suporte à Agricultura:** Análise das condições climáticas (chuva, evapotranspiração, temperatura) para fornecer recomendações de irrigação, plantio ou colheita. **Comunicação de Mídia:** Criação de scripts ou textos para noticiários sobre o tempo, com foco em eventos significativos (ex: tempestades nomeadas). **Análise de Condições Marítimas:** Geração de previsões para atividades náuticas, pesca ou surf, detalhando altura de onda, direção do swell e ventos.

## Pitfalls
**Alucinação de Dados:** O LLM pode "alucinar" dados meteorológicos ou extrapolar previsões de forma excessivamente confiante se os dados de entrada forem ambíguos ou insuficientes. Isso é especialmente perigoso em previsões espaciais baseadas em dados de pontos [1]. **Perda de Contexto Hidrológico:** O LLM, sem dados de rios ou solo, pode fazer alegações não fundamentadas sobre o impacto real de inundações. É crucial restringir o LLM a apenas informações meteorológicas (ex: "Você não tem informações sobre hidrologia atual ou prevista, então não a mencione") [1]. **Violação de Restrições:** LLMs podem ignorar restrições de formato (ex: limite de palavras, uso de markdown) ou incluir chamadas para ação genéricas, a menos que as instruções sejam extremamente explícitas e repetidas [1]. **Interpretação Inconsistente de Dados:** A menos que o prompt inclua tabelas de referência (ex: Escala Beaufort para ventos) ou defina limites claros (ex: "Maré grande se >7.0m"), o LLM pode interpretar dados numéricos de forma inconsistente [1]. **Falta de Gestão de Incerteza:** A maioria dos prompts de meteorologia foca em dados determinísticos, o que pode levar a resumos que não comunicam a incerteza da previsão, especialmente em prazos mais longos [1].

## URL
[https://medium.com/@rob_cowling/%EF%B8%8Fhow-to-prompt-llms-to-craft-weather-summaries-for-flood-forecasting-cd9936828daf](https://medium.com/@rob_cowling/%EF%B8%8Fhow-to-prompt-llms-to-craft-weather-summaries-for-flood-forecasting-cd9936828daf)
