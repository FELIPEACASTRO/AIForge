# Prompts de Análise de Solo (Soil Analysis Prompts)

## Description
A Engenharia de Prompts para Análise de Solo refere-se à criação de instruções estruturadas e detalhadas para Large Language Models (LLMs) e outros sistemas de Inteligência Artificial (IA) com o objetivo de processar, interpretar e gerar recomendações baseadas em dados de análise de solo. Em vez de ser uma técnica de prompt em si, é uma **aplicação especializada** da Engenharia de Prompt no domínio da Agricultura e Ciência do Solo. Os prompts são cruciais para traduzir dados brutos (como resultados de laboratório, leituras de sensores IoT, ou dados de sensoriamento remoto) em informações acionáveis para agricultores, agrônomos e pesquisadores. A eficácia depende da precisão dos dados de entrada e da capacidade do prompt de definir o contexto agronômico, as variáveis de interesse (pH, NPK, matéria orgânica, textura) e o formato de saída desejado (e.g., recomendação de fertilizante, diagnóstico de deficiência).

## Examples
```
1. **Diagnóstico de Deficiência Nutricional:**
```
Aja como um agrônomo especialista. Analise os seguintes resultados de análise de solo para uma cultura de milho na fase V6: [pH: 5.8, Matéria Orgânica: 2.5%, Fósforo (P): 12 ppm (Baixo), Potássio (K): 150 ppm (Médio), Nitrogênio (N) total: 0.1%].
1. Identifique a deficiência nutricional mais crítica.
2. Explique a causa provável.
3. Sugira uma intervenção imediata.
Formato de saída: Tabela com Deficiência, Causa e Recomendação.
```

2. **Recomendação de Calagem e Gessagem:**
```
Com base nos dados de solo (Cultura: Soja, Tipo de Solo: Argiloso, pH atual: 4.9, Saturação de Bases (V%): 35%, CTC a pH 7.0: 15 cmolc/dm³), calcule a necessidade de calagem para elevar o V% para 60%.
1. Qual a quantidade de calcário (PRNT 80%) necessária por hectare?
2. Qual a quantidade de gesso (se necessário) para neutralizar o Alumínio Tóxico (Al³⁺: 1.5 cmolc/dm³)?
3. Forneça o cálculo passo a passo.
```

3. **Interpretação de Dados de Sensor IoT:**
```
Interprete a seguinte série de dados de sensor de solo para uma plantação de café (variedade Catuaí):
[Dia 1: Umidade 45%, Temperatura 22°C, Condutividade Elétrica 0.8 dS/m]
[Dia 2: Umidade 38%, Temperatura 25°C, Condutividade Elétrica 0.9 dS/m]
[Dia 3: Umidade 30%, Temperatura 28°C, Condutividade Elétrica 1.1 dS/m]
O que a tendência de queda na umidade e aumento na CE indica? Qual a recomendação de irrigação para o Dia 4, considerando o ponto de murcha em 25%?
```

4. **Classificação de Textura do Solo (A partir de Dados Brutos):**
```
Classifique a textura do solo com base na seguinte composição granulométrica:
[Areia: 65%, Silte: 20%, Argila: 15%]
1. Use o triângulo textural para determinar a classe.
2. Descreva duas implicações agronômicas dessa textura (e.g., drenagem, retenção de água).
```

5. **Otimização de Fertilização com Restrições:**
```
Crie um plano de fertilização para a cultura de trigo (objetivo de produtividade: 5 ton/ha) em solo com as seguintes características: [P: 18 ppm (Médio), K: 200 ppm (Bom), pH: 6.2].
Restrição: O orçamento permite um máximo de 100 kg/ha de fertilizante NPK (fórmula 10-20-20).
1. Calcule a dose ideal de N, P₂O₅ e K₂O.
2. Ajuste a recomendação para a restrição orçamentária, priorizando o nutriente mais limitante.
3. Justifique a priorização.
```

6. **Análise de Risco de Salinidade:**
```
Avalie o risco de salinidade para um solo irrigado no semiárido.
Dados: [Condutividade Elétrica (CE): 4.5 dS/m, pH: 7.8, Sódio Trocável (PST): 10%].
1. Classifique o solo (Salino, Sódico, Salino-Sódico ou Normal).
2. Descreva o impacto dessa condição na cultura de algodão.
3. Sugira uma medida de manejo para mitigar o problema.
```
```

## Best Practices
**Fornecer Dados Estruturados e Completos:** Sempre inclua o máximo de dados de análise de solo possível (pH, NPK, MO, CTC, Alumínio, etc.), juntamente com o **contexto agronômico** (cultura, fase de desenvolvimento, clima, objetivo de produtividade). **Definir o Papel (Role Prompting):** Inicie o prompt instruindo o LLM a agir como um especialista (e.g., "Aja como um agrônomo especialista em solos tropicais") para ativar o conhecimento especializado do modelo. **Especificar o Formato de Saída:** Peça a saída em um formato fácil de usar (tabela, lista, JSON) para garantir que a informação seja acionável e não apenas um texto corrido. **Incluir Restrições e Variáveis:** Se a recomendação tiver restrições (orçamento, tipo de fertilizante disponível, legislação local), inclua-as explicitamente para que o LLM as considere na otimização. **Solicitar Justificativa:** Peça ao LLM para justificar suas recomendações. Isso ajuda a validar a resposta e a identificar possíveis alucinações ou erros de cálculo.

## Use Cases
**Recomendação de Fertilidade:** Geração de planos de fertilização e calagem otimizados com base em análises de solo e metas de produtividade. **Diagnóstico Rápido de Problemas:** Identificação de deficiências nutricionais, toxicidade ou problemas de pH a partir de dados de laboratório ou sintomas visuais. **Interpretação de Sensores IoT:** Tradução de dados em tempo real (umidade, temperatura, CE) em decisões de manejo (irrigação, lixiviação). **Educação e Treinamento:** Criação de cenários de estudo de caso para estudantes de agronomia ou treinamento de técnicos agrícolas. **Mapeamento de Variabilidade:** Interpretação de múltiplas amostras de solo de diferentes zonas de manejo para identificar padrões de variabilidade e otimizar a aplicação de insumos.

## Pitfalls
**Confiança Excessiva (Alucinação):** O LLM pode "alucinar" dados ou recomendações, especialmente se o prompt for vago ou se os dados de entrada estiverem incompletos. A taxa de acerto em ciência do solo é moderada (máximo de 65% em testes), exigindo **validação humana**. **Ignorar o Contexto Local:** O LLM pode fornecer recomendações genéricas que não se aplicam à legislação, tipo de solo ou práticas agrícolas locais. O prompt deve sempre incluir o contexto geográfico. **Erro de Unidades e Conversões:** A mistura de unidades de medida (e.g., ppm vs. mg/dm³, kg/ha vs. lb/acre) pode levar a erros catastróficos nas recomendações. O prompt deve ser explícito sobre as unidades. **Dados de Entrada Insuficientes:** A falta de dados cruciais (como a CTC ou o teor de Alumínio) resultará em recomendações incompletas ou incorretas. **Interpretação de Imagens/Gráficos:** LLMs baseados apenas em texto não podem interpretar diretamente gráficos de análise de solo ou imagens de microscopia (como as usadas em IAEM), exigindo que os dados sejam primeiro convertidos em texto estruturado.

## URL
[https://www.sciencedirect.com/science/article/pii/S2950289625000028](https://www.sciencedirect.com/science/article/pii/S2950289625000028)
