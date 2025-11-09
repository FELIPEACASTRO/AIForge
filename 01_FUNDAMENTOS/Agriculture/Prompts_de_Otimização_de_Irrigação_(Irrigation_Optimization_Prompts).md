# Prompts de Otimização de Irrigação (Irrigation Optimization Prompts)

## Description
Prompts de Otimização de Irrigação são instruções estruturadas e contextuais fornecidas a Modelos de Linguagem Grande (LLMs) para analisar dados agrícolas em tempo real e históricos (como umidade do solo, previsão do tempo, tipo de cultura, estágio de crescimento e dados de sensores IoT) e gerar recomendações precisas e acionáveis sobre quando, onde e quanto irrigar. Esta técnica de prompt se enquadra na **Agricultura de Precisão**, onde o LLM atua como um assistente de decisão inteligente, traduzindo dados complexos em linguagem natural e comandos específicos para sistemas de irrigação automatizados. A eficácia desses prompts depende da sua capacidade de integrar-se a APIs e funcionalidades de **Function Calling** para acessar e processar dados externos em tempo real.

## Examples
```
1.  **Prompt de Análise de Dados e Recomendação:**
    ```
    Aja como um especialista em irrigação de precisão. Analise os seguintes dados:
    - Cultura: Milho (Estágio V6)
    - Localização: Fazenda Esperança, Coordenadas: [34.0522, -118.2437]
    - Umidade do Solo (Zona de Raiz): 45% (Limite Crítico: 40%, Capacidade de Campo: 65%)
    - Previsão do Tempo (Próximas 48h): 0% chance de chuva, ETc (Evapotranspiração da Cultura): 6 mm/dia
    - Sistema de Irrigação: Pivô Central (Eficiência: 85%)
    
    Com base na necessidade de repor a água consumida e manter a umidade acima do limite crítico, calcule a lâmina de água necessária (em mm) e forneça a recomendação de irrigação em formato de comando JSON para o sistema.
    ```

2.  **Prompt de Diagnóstico e Ajuste de Sistema:**
    ```
    A bomba de irrigação da Seção 3 falhou inesperadamente. O sistema de monitoramento indica que a umidade do solo na zona de raiz da cultura de Soja (Estágio R2) caiu para 38%. A temperatura ambiente é de 35°C.
    
    Qual é o risco imediato para a cultura? Proponha um plano de contingência de 72 horas, incluindo a redistribuição da irrigação das Seções 1 e 2 para compensar a falha, e a lâmina de água extra (em mm) que deve ser aplicada nessas seções para mitigar o estresse hídrico na Seção 3.
    ```

3.  **Prompt de Otimização de Cronograma Semanal:**
    ```
    Gere um cronograma de irrigação otimizado para a próxima semana (7 dias) para a cultura de Trigo (Estágio de Afilhamento).
    - Tipo de Solo: Argiloso (Alta retenção)
    - Umidade Atual: 60%
    - Previsão de Chuva: 10mm no Dia 4
    - ETc Média Diária: 5.5 mm
    
    O objetivo é manter a umidade entre 55% e 70%. Apresente o cronograma dia a dia, indicando a lâmina de água (mm) a ser aplicada ou 'Nenhuma'.
    ```

4.  **Prompt de Simulação de Cenário:**
    ```
    Simule o impacto de uma onda de calor (temperatura média de 40°C, ETc de 8 mm/dia) por 5 dias consecutivos na cultura de Alface (Estágio de Cabeça).
    
    Se a irrigação for mantida no cronograma padrão de 4 mm a cada 2 dias, qual será o nível de estresse hídrico (em porcentagem de depleção) no final do período? Qual seria a lâmina de água ideal para evitar o estresse?
    ```

5.  **Prompt de Interpretação de Dados de Sensor:**
    ```
    Interprete o seguinte conjunto de dados de sensor de umidade do solo (TDR) e gere uma recomendação.
    - Sensor 1 (15cm): 52%
    - Sensor 2 (30cm): 48%
    - Sensor 3 (45cm): 40%
    - Cultura: Uva (Estágio de Maturação)
    - Requisito: Manter a zona de 30-45cm em estresse leve (40-45%) para otimizar a qualidade do fruto.
    
    A irrigação atual está adequada? Se não, qual ajuste (aumento/diminuição percentual na lâmina) você sugere?
    ```
```

## Best Practices
*   **Estrutura de Prompt Clara:** Defina o **papel** do LLM (Ex: "Especialista em Hidrologia Agrícola"), forneça o **contexto** (cultura, solo, sistema) e inclua **dados de entrada** (umidade, clima, ETc) de forma organizada (tabelas, listas ou JSON).
*   **Integração de Dados em Tempo Real:** O prompt deve ser projetado para acionar funções (Function Calling) que busquem dados de APIs externas (sensores IoT, serviços meteorológicos) antes de gerar a resposta.
*   **Saída Acionável e Formatada:** Especifique o formato de saída desejado (Ex: JSON, XML, ou um comando de sistema) para que a resposta possa ser diretamente consumida por um sistema de irrigação automatizado.
*   **Definição de Limites Críticos:** Inclua no prompt os limites de estresse hídrico e capacidade de campo específicos para a cultura e o solo, permitindo que o LLM faça cálculos precisos.
*   **Iteração e Refinamento:** Use prompts de acompanhamento para refinar as recomendações (Ex: "Como essa recomendação muda se a eficiência do pivô for de 90%?").

## Use Cases
*   **Agendamento Dinâmico de Irrigação:** Criação de cronogramas de irrigação que se ajustam automaticamente a mudanças climáticas e estágios de crescimento da cultura.
*   **Diagnóstico de Estresse Hídrico:** Análise de dados de sensores e imagens de satélite (NDVI) para identificar áreas da lavoura com estresse e recomendar intervenções localizadas.
*   **Otimização de Recursos Hídricos:** Cálculo da lâmina de água mínima necessária para maximizar a produtividade, resultando em economia de água e energia.
*   **Simulação de Cenários:** Previsão do impacto de eventos climáticos extremos (secas, ondas de calor) e planejamento de estratégias de mitigação.
*   **Assistência a Produtores:** Fornecer explicações didáticas e baseadas em dados para produtores rurais sobre as decisões de irrigação.

## Pitfalls
*   **Dependência de Dados Imprecisos:** O LLM é tão bom quanto os dados que recebe. Dados de sensores descalibrados ou previsões meteorológicas incorretas levarão a recomendações falhas.
*   **Falta de Contexto Agronômico:** Não fornecer detalhes suficientes sobre o tipo de solo, cultura, profundidade da raiz e estágio fenológico pode resultar em recomendações genéricas e ineficazes.
*   **Saída Não Acionável:** Se o prompt não exigir um formato de saída estruturado (como JSON ou um comando específico), o LLM pode gerar texto descritivo que não pode ser usado para automatizar o sistema.
*   **Ignorar a Latência:** Em sistemas de tempo real, a latência na busca de dados via API e na geração da resposta do LLM pode atrasar a decisão, impactando a cultura.
*   **Superestimação da Capacidade do LLM:** O LLM é uma ferramenta de raciocínio e tradução, não um modelo hidrológico. Ele deve ser alimentado com dados processados e não deve ser solicitado a realizar cálculos complexos de balanço hídrico sem o auxílio de ferramentas externas (Function Calling).

## URL
[https://dr-arsanjani.medium.com/enhancing-agricultural-decision-making-with-function-calling-in-llms-a-vision-for-the-future-cc11960bf5d1](https://dr-arsanjani.medium.com/enhancing-agricultural-decision-making-with-function-calling-in-llms-a-vision-for-the-future-cc11960bf5d1)
