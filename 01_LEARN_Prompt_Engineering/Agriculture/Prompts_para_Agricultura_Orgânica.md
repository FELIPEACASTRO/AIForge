# Prompts para Agricultura Orgânica

## Description
A técnica de "Prompts para Agricultura Orgânica" refere-se ao uso de Large Language Models (LLMs) e outras IAs generativas para auxiliar agricultores, agrônomos e pesquisadores na tomada de decisões e na otimização de práticas agrícolas sustentáveis e orgânicas. Envolve a criação de instruções detalhadas e contextuais para a IA, solicitando planos de rotação de culturas, estratégias de controle biológico de pragas, otimização do uso de água e nutrientes, e análise de dados de solo e clima, tudo dentro dos princípios da agricultura orgânica e regenerativa. O objetivo é democratizar o acesso a informações agronômicas complexas e aumentar a eficiência e a sustentabilidade das operações agrícolas.

## Examples
```
1. **Planejamento de Rotação de Culturas (Few-Shot/CoT):** "Aja como um agrônomo especialista em agricultura orgânica. Meu objetivo é criar um plano de rotação de culturas de 4 anos para um campo de 5 hectares na Zona de Rusticidade 9b (clima mediterrâneo). O solo é argiloso com baixo teor de matéria orgânica (1.5%). As culturas principais que desejo incluir são: Tomate (solanácea), Feijão-de-corda (leguminosa) e Couve (brássica).
   *   **Passo 1:** Analise as necessidades nutricionais e o risco de pragas de cada cultura.
   *   **Passo 2:** Proponha uma sequência de rotação que maximize a fixação de nitrogênio e a supressão de ervas daninhas.
   *   **Passo 3:** Inclua uma cultura de cobertura de inverno (adubo verde) em cada ciclo.
   *   **Passo 4:** Apresente o plano em uma tabela com as colunas: Ano, Estação, Cultura Principal, Cultura de Cobertura, Benefício Agronômico."

2. **Manejo Orgânico de Pragas (Zero-Shot/Restrição):** "Sou um agricultor orgânico e estou enfrentando uma infestação de pulgões (Aphididae) em minhas plantas de morango. Forneça uma lista de 5 métodos de controle biológico e/ou natural que sejam estritamente orgânicos e eficazes contra pulgões em morangos. Para cada método, descreva o mecanismo de ação e a frequência de aplicação recomendada."

3. **Otimização de Fertilidade do Solo (Contextualizado):** "Meu solo tem pH 5.8, teor de fósforo (P) de 10 ppm e potássio (K) de 150 ppm. Quero plantar cenouras orgânicas. Qual é a melhor estratégia de fertilização orgânica para corrigir o pH e fornecer os nutrientes necessários para uma colheita abundante, sem usar fertilizantes sintéticos? Sugira a quantidade e o tipo de emenda orgânica (ex: calcário dolomítico, composto, cinzas de madeira) que devo aplicar por hectare."

4. **Análise de Dados Climáticos e Irrigação (Ação):** "Com base na previsão de que teremos 15 dias consecutivos sem chuva e temperaturas médias de 32°C, e considerando que minhas plantas de alface orgânica estão na fase de formação de cabeça, qual deve ser o volume de água (em mm/dia) e o horário ideal de irrigação para evitar o estresse hídrico e o risco de doenças fúngicas? Justifique sua resposta com base nas necessidades hídricas da alface."

5. **Criação de Conteúdo Educacional (Role-Playing/Output Format):** "Aja como um redator de conteúdo para uma cooperativa de agricultura orgânica. Crie um post de blog de 500 palavras com o título '5 Mitos sobre o Controle de Pragas Orgânico'. O tom deve ser informativo e encorajador. Inclua uma chamada para ação para um curso sobre compostagem no final. Use subtítulos claros e linguagem acessível."

6. **Diagnóstico e Recomendação (CoT/Restrição):** "Minhas folhas de abobrinha orgânica estão ficando amarelas nas bordas e as nervuras permanecem verdes. Não há sinais visíveis de insetos.
   *   **Passo 1:** Liste as 3 deficiências nutricionais mais prováveis que causam este sintoma.
   *   **Passo 2:** Para cada deficiência, sugira um teste de campo simples que eu possa fazer.
   *   **Passo 3:** Se for uma deficiência de Magnésio, qual é a solução orgânica mais rápida e eficaz para aplicação foliar? (APENAS soluções orgânicas certificadas)."
```

## Best Practices
1. **Definir o Papel (Role-Playing):** Começar o prompt definindo a IA como um "Agrônomo Especialista em Agricultura Orgânica" ou "Consultor em Agricultura Regenerativa".
2. **Especificar o Contexto:** Incluir detalhes cruciais como: tipo de solo (pH, textura), zona de rusticidade (hardiness zone), culturas atuais, histórico de pragas e o objetivo específico (ex: aumentar a matéria orgânica em 1% em 3 anos).
3. **Restrições Orgânicas Claras:** Usar termos de restrição como "APENAS métodos orgânicos", "EXCLUIR qualquer insumo químico sintético" para garantir a conformidade.
4. **Solicitar Formato Estruturado:** Pedir a resposta em formato de tabela, lista ou plano de ação passo a passo para facilitar a aplicação no campo.
5. **Iteração e Refinamento (Chain-of-Thought):** Pedir à IA para justificar suas recomendações com base em princípios agronômicos e, em seguida, refinar o plano com base em novas variáveis (ex: "Agora, ajuste o plano considerando uma seca de 3 semanas no meio da estação").

## Use Cases
1. **Planejamento de Culturas:** Criação de planos de rotação de culturas orgânicas, considerando tipo de solo, clima local e demanda de mercado.
2. **Manejo Integrado de Pragas (MIP) Orgânico:** Sugestão de métodos de controle biológico e natural para pragas e doenças específicas, evitando pesticidas químicos.
3. **Otimização de Nutrientes e Solo:** Recomendações para fertilização orgânica (compostagem, adubos verdes) e estratégias para aumentar a matéria orgânica e a saúde do solo.
4. **Gestão de Recursos Hídricos:** Otimização de cronogramas de irrigação com base em previsões climáticas e necessidades específicas da cultura.
5. **Cadeia de Suprimentos e Logística:** Previsão de demanda e otimização da cadeia de suprimentos para reduzir o desperdício de alimentos orgânicos.
6. **Treinamento e Educação:** Geração de material didático e planos de treinamento para agricultores sobre novas técnicas orgânicas e regulamentações.

## Pitfalls
1. **Generalização Excessiva:** Usar prompts vagos como "Como fazer agricultura orgânica?" resulta em respostas genéricas e inúteis. A especificidade é vital.
2. **Ignorar o Contexto Local:** Falhar em fornecer dados locais (clima, solo, regulamentações) leva a recomendações impraticáveis ou inadequadas para a região.
3. **Confiança Cega (Over-reliance):** Tratar a saída da IA como verdade absoluta. As recomendações devem ser sempre validadas por um agrônomo ou conhecimento prático local.
4. **Viés de Dados:** A IA pode ser treinada em dados que favorecem a agricultura convencional, exigindo prompts mais rigorosos para impor as restrições orgânicas.
5. **Falta de Ação:** Gerar planos complexos sem um passo a passo claro para a implementação, tornando o prompt uma ferramenta de pesquisa e não de ação.

## URL
[https://promptsty.com/prompts-for-sustainable-agriculture/](https://promptsty.com/prompts-for-sustainable-agriculture/)
