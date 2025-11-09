# Prompts de Detecção de Plágio e Conteúdo Gerado por IA

## Description
Prompts de Detecção de Plágio e Conteúdo Gerado por IA são instruções específicas fornecidas a Large Language Models (LLMs) para auxiliar na análise de textos, identificando padrões que sugerem plágio tradicional (cópia não citada) ou a geração por outras IAs. Embora os LLMs não substituam ferramentas especializadas (como Turnitin ou Copyleaks), eles podem ser usados para: 1) **Análise de Estilo e Padrões**: Pedir ao LLM para identificar trechos com mudanças abruptas de estilo, vocabulário excessivamente formal ou estruturas sintáticas que se assemelham a saídas típicas de IA. 2) **Verificação de Citações**: Checar a conformidade de referências e citações com normas acadêmicas (como ABNT ou APA). 3) **Comparação de Fontes**: Solicitar a comparação de um trecho de texto com uma fonte específica fornecida, buscando similaridades. É crucial entender que a detecção de IA por LLMs é falível e deve ser usada como um indicador inicial, e não como prova definitiva.

## Examples
```
1. **Análise de Estilo para Detecção de IA:** 'Analise o texto a seguir. Identifique e destaque quaisquer parágrafos que apresentem características de escrita típicas de Large Language Models (LLMs), como vocabulário excessivamente formal, falta de voz autoral ou estruturas repetitivas. Justifique sua análise para cada trecho suspeito. [TEXTO AQUI]'

2. **Verificação de Plágio Simples:** 'Compare o seguinte texto com a fonte original que fornecerei. Destaque todas as frases ou trechos com mais de 5 palavras consecutivas que foram copiadas sem aspas ou citação. [TEXTO AQUI] [FONTE ORIGINAL AQUI]'

3. **Checagem de Conformidade ABNT/APA:** 'Atue como um revisor acadêmico. Verifique se todas as citações diretas e indiretas no texto estão formatadas corretamente de acordo com as normas ABNT/APA. Liste as citações incorretas e sugira a correção. [TEXTO AQUI]'

4. **Identificação de Paráfrase Excessiva:** 'Leia o texto e a fonte original. Identifique trechos onde a paráfrase é tão próxima da fonte que pode ser considerada plágio de ideias ou paráfrase indevida. [TEXTO AQUI] [FONTE ORIGINAL AQUI]'

5. **Simulação de Detector de IA:** 'Simule ser um detector de conteúdo de IA. Atribua uma pontuação de 0 a 100% para a probabilidade de o texto ter sido gerado por uma IA. Em seguida, liste 3 a 5 palavras ou frases que mais contribuíram para essa pontuação. [TEXTO AQUI]'

6. **Reescrita para Originalidade (Uso Ético):** 'Reescreva o seguinte parágrafo com a sua própria voz, mantendo o sentido, mas alterando a estrutura e o vocabulário para garantir a originalidade e evitar a detecção de IA. [PARÁGRAFO SUSPEITO AQUI]'

7. **Análise de Coerência e Fatos:** 'O texto abaixo trata do tema [TEMA]. Verifique a coerência lógica e a precisão factual dos argumentos apresentados. Aponte quaisquer 'alucinações' ou informações que pareçam ter sido geradas sem base em dados reais. [TEXTO AQUI]'

8. **Prompt de Refinamento de Detecção:** 'Com base na sua análise anterior, onde você identificou o trecho [TRECHO SUSPEITO], qual seria o prompt mais eficaz para um detector de plágio especializado encontrar a fonte original deste conteúdo?'

9. **Detecção de Manipulação de Caracteres (Homóglifos):** 'Analise o texto em busca de caracteres homóglifos (letras de alfabetos diferentes que se parecem com letras latinas) que possam ter sido inseridos para tentar burlar a detecção de plágio. Se encontrar, destaque-os. [TEXTO AQUI]'

10. **Prompt de Análise Comparativa de Dois Textos:** 'Atue como um especialista em linguística forense. Analise o Texto A e o Texto B. Eles foram escritos pela mesma pessoa ou um é uma cópia/paráfrase do outro? Justifique sua conclusão com base em métricas de estilo, complexidade e vocabulário. [TEXTO A] [TEXTO B]'
```

## Best Practices
1. **Forneça Contexto e Função:** Comece o prompt definindo o papel do LLM (ex: 'Atue como um revisor acadêmico rigoroso') e o objetivo (ex: 'Identificar padrões de escrita de IA').
2. **Use Amostras de Texto Limitadas:** LLMs têm limites de contexto. Para textos longos, divida-o em seções e analise-as separadamente.
3. **Solicite Justificativa (Chain-of-Thought):** Peça ao LLM para explicar *por que* um trecho é suspeito (ex: 'Justifique sua análise para cada trecho suspeito'). Isso aumenta a transparência e a confiabilidade.
4. **Combine com Ferramentas Especializadas:** Use o prompt como um filtro inicial. Se o LLM apontar suspeitas, utilize ferramentas de detecção de plágio e IA dedicadas para a verificação final.
5. **Seja Específico sobre as Normas:** Se o foco for acadêmico, especifique as normas de citação (ABNT, APA, Vancouver, etc.).

## Use Cases
nan

## Pitfalls
1. **Falsos Positivos/Negativos:** O LLM pode rotular conteúdo humano como gerado por IA (Falso Positivo) ou falhar em detectar conteúdo de IA bem 'humanizado' (Falso Negativo).
2. **Dependência Excessiva:** Confiar no LLM como o único detector de plágio, ignorando a necessidade de ferramentas especializadas e a revisão humana.
3. **Viés de Treinamento:** O LLM tende a identificar como 'IA-gerado' o texto que se assemelha ao seu próprio estilo de treinamento, o que pode penalizar a escrita clara e concisa.
4. **Uso Antiético:** Utilizar prompts para 'burlar' detectores de IA, o que é uma prática antiética e pode levar a sanções acadêmicas ou profissionais.
5. **Limitação de Contexto:** A incapacidade de analisar documentos muito longos de uma só vez, levando a análises parciais ou incompletas.

## URL
[https://treinamentosaf.com.br/ia-para-identificacao-de-plagio-em-trabalhos-academicos-guia-pratico-2025/](https://treinamentosaf.com.br/ia-para-identificacao-de-plagio-em-trabalhos-academicos-guia-pratico-2025/)
