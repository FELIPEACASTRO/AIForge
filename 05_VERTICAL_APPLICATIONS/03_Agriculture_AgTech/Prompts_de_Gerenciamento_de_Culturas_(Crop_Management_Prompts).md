# Prompts de Gerenciamento de Culturas (Crop Management Prompts)

## Description
Prompts de Gerenciamento de Culturas são instruções especializadas fornecidas a Modelos de Linguagem Grande (LLMs) ou modelos multimodais de Inteligência Artificial (IA) para auxiliar na tomada de decisões agrícolas. Eles são projetados para simular a experiência de um agrônomo ou especialista, fornecendo análises, diagnósticos e recomendações acionáveis para otimizar a produção, gerenciar pragas e doenças, e melhorar a sustentabilidade. A eficácia desses prompts reside na inclusão de contexto específico (tipo de cultura, estágio de crescimento, condições climáticas, tipo de solo) para gerar respostas precisas e relevantes, transformando a IA em uma ferramenta de suporte à decisão no campo.

## Examples
```
**1. Diagnóstico Multimodal de Doença:** 'Aja como um fitopatologista. Analise a imagem anexa de uma folha de soja (variedade M8210, estágio R2). Descreva os sintomas, identifique a doença mais provável e forneça um plano de manejo integrado de 7 dias, incluindo o ingrediente ativo de fungicida recomendado e a justificativa para a escolha.'
**2. Otimização de Fertilização:** 'Com base na análise de solo (pH 5.8, P 12 ppm, K 150 ppm, MO 2.5%) e no alvo de produtividade de 80 sacas/hectare de milho (estágio V6), calcule a dose de nitrogênio (N) e potássio (K) necessária para a cobertura. Recomende a melhor fonte de N (ureia ou nitrato de amônio) e justifique a escolha em termos de eficiência e custo-benefício (considerando o preço atual da ureia de R$ 3.000/ton).'
**3. Gerenciamento de Irrigação:** 'Sou produtor de feijão-caupi (estágio de floração) no semiárido. A evapotranspiração de referência (ETo) dos últimos 3 dias foi de 6.5 mm/dia. O solo é franco-arenoso e a capacidade de campo é de 15%. A lâmina de irrigação atual é de 10 mm. Calcule a nova lâmina e o intervalo de irrigação para manter o estresse hídrico em 20% da água disponível, explicando o cálculo do balanço hídrico simplificado.'
**4. Planejamento de Safra:** 'Aja como um consultor de planejamento agrícola. Para a próxima safra de verão, em uma área com histórico de nematoides de galha (Meloidogyne javanica), sugira 3 opções de rotação de culturas que sejam economicamente viáveis e ajudem a reduzir a população do nematoide. Para cada opção, liste a cultura, o período de plantio ideal e o principal benefício agronômico.'
**5. Controle de Plantas Daninhas:** 'Identifique a planta daninha na imagem (se multimodal) ou descrita como 'folha larga, prostrada, com seiva leitosa' (se textual). A cultura é cana-de-açúcar em soqueira. Recomende uma mistura de herbicidas pós-emergentes eficaz e seletiva, detalhando a dose por hectare e o momento ideal de aplicação (altura da daninha).'
**6. Análise de Dados de Sensores (NDVI):** 'Interprete o seguinte dado de NDVI de uma área de trigo (estágio de enchimento de grãos): 'Zona 1 (10 ha): NDVI 0.85; Zona 2 (5 ha): NDVI 0.60'. Descreva a provável causa da diferença (assumindo que a Zona 2 tem deficiência nutricional) e sugira uma ação de manejo de precisão para a Zona 2, como a aplicação de nitrogênio em taxa variável, especificando a quantidade extra necessária.'
**7. Conformidade Regulatória:** 'Qual é a restrição de uso do ingrediente ativo 'Glifosato' (Roundup) em áreas de preservação permanente (APP) no estado de São Paulo, Brasil? Resuma a legislação federal e estadual aplicável e indique a distância mínima de aplicação em relação a corpos d'água, agindo como um advogado ambiental.'
```

## Best Practices
**Especificidade do Contexto:** Sempre inclua o máximo de detalhes possível, como cultura, variedade, estágio fenológico (ex: V4, floração), tipo de solo, histórico de manejo e dados climáticos recentes.
**Definição de Persona:** Atribua à IA uma persona de especialista (ex: 'Aja como um agrônomo com 20 anos de experiência em soja no Cerrado') para elevar a qualidade e a relevância das recomendações.
**Solicitação de Raciocínio:** Peça à IA para justificar suas recomendações, citando princípios agronômicos ou dados (se o modelo tiver acesso a eles), o que ajuda a mitigar 'alucinações'.
**Uso Multimodal (se aplicável):** Para diagnóstico de pragas ou doenças, inclua imagens e solicite a análise visual, seguida de um plano de manejo.
**Foco na Ação:** Estruture o prompt para que a saída seja um plano de ação claro e sequencial, e não apenas uma descrição do problema.

## Use Cases
**Diagnóstico e Manejo de Pragas/Doenças:** Identificação de problemas a partir de descrições ou imagens e sugestão de tratamentos químicos ou biológicos.
**Otimização de Nutrição de Plantas:** Recomendação de doses e tipos de fertilizantes com base em análises de solo e estágio da cultura.
**Planejamento de Safra:** Auxílio na escolha de variedades, datas de plantio e espaçamento, considerando dados históricos e previsões climáticas.
**Gerenciamento de Irrigação:** Cálculo de necessidades hídricas e sugestão de cronogramas de irrigação.
**Conformidade Regulatória:** Interpretação de regulamentos locais sobre uso de pesticidas e práticas de manejo.
**Análise de Dados de Sensores:** Processamento e interpretação de dados de drones, satélites e sensores de campo para mapeamento de produtividade e saúde da cultura.

## Pitfalls
**Alucinações (Recomendações Incorretas):** A IA pode sugerir produtos ou práticas inexistentes, desatualizadas ou inadequadas para a região. Sempre verifique as recomendações com um agrônomo local.
**Falta de Especificidade Local:** Respostas genéricas que não consideram as microcondições climáticas, tipos de solo ou regulamentações específicas da fazenda.
**Dependência Excessiva:** Usar a IA como única fonte de decisão, ignorando a experiência prática e a observação de campo.
**Inclusão de Dados Sensíveis:** Evitar compartilhar informações confidenciais da fazenda (como localização exata ou dados financeiros) em prompts de modelos públicos.
**Prompts Ambíguos:** Perguntas abertas ou vagas que resultam em respostas igualmente vagas e pouco úteis.

## URL
[https://cropsandsoils.extension.wisc.edu/articles/ai-in-agriculture/](https://cropsandsoils.extension.wisc.edu/articles/ai-in-agriculture/)
