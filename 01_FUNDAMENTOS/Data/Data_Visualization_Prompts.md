# Data Visualization Prompts

## Description
**Data Visualization Prompts** (Prompts de Visualização de Dados) são instruções estruturadas e detalhadas fornecidas a modelos de Linguagem Grande (LLMs) ou ferramentas de Business Intelligence (BI) baseadas em IA para gerar gráficos, *dashboards* e outras representações visuais a partir de dados brutos ou análises [2]. A técnica de *prompt engineering* neste contexto visa superar a tendência da IA de gerar visualizações genéricas ou inadequadas, focando na **intenção** e no **contexto** da análise. Em vez de apenas solicitar "um gráfico de vendas", um prompt eficaz especifica o tipo de dado, o objetivo da comunicação, o tipo de gráfico ideal e até mesmo detalhes estéticos, como paleta de cores e anotações, transformando a IA em um co-piloto de *data storytelling* [1]. O uso de *Data Visualization Prompts* é crucial para garantir que o resultado visual seja preciso, didático e alinhado com os princípios de design de visualização de dados [2].

## Examples
```
Os exemplos a seguir demonstram a aplicação de prompts detalhados para visualização de dados, focando em diferentes objetivos e especificações técnicas:

1.  **Comparação Planejado vs. Real (Gráfico Bullet):**
    > "Com base nos dados de despesas trimestrais fornecidos, crie um **Gráfico Bullet** para comparar o **Gasto Planejado** versus o **Gasto Real** para cada um dos quatro trimestres. O objetivo é destacar rapidamente o desempenho. Mantenha o design minimalista, use a cor azul para o gasto real e um cinza claro para o planejado. O eixo Y deve ter o sufixo 'R$' e exibir 10 *ticks* para melhor legibilidade."

2.  **Análise de Tendência (Gráfico de Linha):**
    > "Gere um **Gráfico de Linha** que mostre a **Receita Mensal** ao longo dos últimos 24 meses. O objetivo é identificar a tendência de crescimento. Adicione uma **linha de tendência** (regressão linear) e um **intervalo de confiança** de 95%. Adicione uma anotação em texto destacando o mês com o maior crescimento percentual."

3.  **Composição de Mercado (Gráfico de Pizza/Rosca):**
    > "Crie um **Gráfico de Rosca** (Donut Chart) para visualizar a **Composição de Mercado** dos nossos 5 principais produtos. O objetivo é mostrar a proporção de vendas de cada produto em relação ao total. Use uma paleta de cores *safe* para daltônicos e destaque a fatia do produto 'Alpha' com uma cor de alto contraste. Inclua os valores percentuais diretamente nas fatias."

4.  **Layout de Dashboard (Estrutura):**
    > "Proponha um **Layout de Dashboard** para monitoramento de *e-commerce*. A visualização deve incluir: 1) **KPIs de Sumário** (Vendas Totais, Taxa de Conversão) no topo; 2) **Gráfico de Linha** de Vendas Diárias no centro; 3) **Gráfico de Barras** de Vendas por Região na lateral. O layout deve ser otimizado para uma tela de 1920x1080."

5.  **Detecção de Anomalias (Gráfico de Dispersão):**
    > "Utilize um **Gráfico de Dispersão** para plotar a **Duração da Chamada** (eixo X) versus a **Satisfação do Cliente** (eixo Y) para o último mês. O objetivo é identificar *outliers* (anomalias). Sinalize em vermelho qualquer ponto de dado onde a Duração da Chamada seja superior a 15 minutos E a Satisfação do Cliente seja inferior a 3 (em uma escala de 5). Forneça o código Python com a biblioteca `matplotlib`."

6.  **Segmentação Interativa (Filtros):**
    > "Para o conjunto de dados de *leads* (dados categóricos), sugira **opções de filtro chave** (ex: Região, Fonte do Lead, Tamanho da Empresa) e **segmentos** (ex: Clientes de Alto Valor vs. Clientes Padrão) para criar um visual interativo. O visual deve ser um **Gráfico de Barras** que compare a Taxa de Conversão por Fonte do Lead, permitindo a filtragem por Região."
```

## Best Practices
As melhores práticas para a criação de **Data Visualization Prompts** eficazes envolvem a clareza, a contextualização e a especificação técnica, garantindo que a IA compreenda o objetivo e o formato desejado [1] [2].

1.  **Definir o Objetivo da Visualização:** Comece explicando o *porquê* da visualização. Qual é a principal mensagem ou *insight* que o gráfico deve comunicar? (Ex: "O gráfico deve destacar a diferença entre o orçamento planejado e o gasto real no último trimestre").
2.  **Especificar o Tipo de Dados:** Informe à IA se os dados são **categóricos** (grupos, como nomes de produtos), **contínuos** (valores numéricos, como vendas ao longo do tempo) ou **temporais** (séries temporais). Isso ajuda a IA a selecionar o tipo de gráfico mais apropriado [1].
3.  **Sugestão de Tipo de Gráfico:** Sempre que possível, sugira o tipo de visualização mais adequado para a mensagem (Ex: "Use um gráfico de linha para tendências", "Use um gráfico de barras empilhadas para composição").
4.  **Detalhes de Estilo e Formato:** Inclua especificações de design para garantir a legibilidade e o profissionalismo. Isso inclui paleta de cores (Ex: "Use cores acessíveis e de alto contraste"), rótulos de eixos (Ex: "Adicione o sufixo 'R$' ao eixo Y"), e anotações (Ex: "Adicione anotações para picos e quedas") [2].
5.  **Instruções de Saída:** Peça à IA para fornecer o código (Python, R, Vega-Lite) ou o formato de saída (JSON, CSV) para que a visualização possa ser reproduzida ou integrada em outras ferramentas.

## Use Cases
A aplicação de *Data Visualization Prompts* abrange diversas áreas, otimizando o fluxo de trabalho de análise e comunicação de dados [2].

*   **Business Intelligence (BI) e Relatórios:** Criação rápida de *dashboards* e relatórios sofisticados, definindo o layout, os KPIs e os tipos de gráficos para cada métrica (Ex: Layout de dashboard de vendas com KPIs no topo e tendências no centro).
*   **Análise Exploratória de Dados (EDA):** Geração rápida de visualizações específicas para explorar relações, distribuições e *outliers* em grandes conjuntos de dados (Ex: Gráfico de dispersão para correlacionar duas variáveis e detectar anomalias).
*   **Design de Visualização:** Auxílio na escolha de paletas de cores acessíveis, fontes e estilos visuais que seguem as melhores práticas de design (Ex: Sugestão de paleta de cores *safe* para daltônicos).
*   **Análise de Séries Temporais:** Criação de gráficos de linha com projeções e intervalos de confiança para prever tendências futuras (Ex: Gráfico de previsão de vendas para os próximos 6 meses).
*   **Comunicação de Dados:** Geração de anotações e textos explicativos que transformam o gráfico em uma ferramenta de *storytelling*, destacando os pontos de inflexão e o contexto por trás dos dados (Ex: Anotação de pico de vendas devido a uma campanha de marketing).
*   **Otimização de Gráficos:** Transformação de um tipo de gráfico menos eficaz em um mais adequado para a mensagem (Ex: Transformar um gráfico de barras em um gráfico *bullet* para comparação de metas).

## Pitfalls
Os erros mais comuns ao usar *Data Visualization Prompts* geralmente resultam da falta de especificidade e da confiança excessiva na capacidade da IA de inferir o contexto [1] [2].

*   **Instruções Ambíguas:** Pedir apenas "um gráfico bonito" ou "visualize os dados" sem especificar o tipo de visualização, o foco dos dados ou os aspectos comparativos. A IA pode gerar um gráfico tecnicamente correto, mas visualmente inútil.
*   **Falta de Contexto:** Não informar o objetivo da visualização (o *insight* que se deseja comunicar) ou o público-alvo. Isso leva a gráficos que não contam a história correta ou são muito complexos para o leitor.
*   **Ignorar Escala e Eixos:** A IA pode, por vezes, gerar gráficos que distorcem a realidade ao truncar o eixo Y ou usar escalas inadequadas, o que é uma falha grave em visualização de dados.
*   **Excesso de Dados:** Tentar visualizar muitas variáveis ou categorias em um único gráfico, resultando em poluição visual e dificuldade de leitura.
*   **Confiança Excessiva no Padrão:** Aceitar o primeiro resultado da IA sem verificar se o tipo de gráfico escolhido é o mais adequado para a mensagem (Ex: usar um gráfico de pizza para muitas categorias).
*   **Não Fornecer o Tipo de Dados:** Não informar se os dados são categóricos, contínuos ou temporais, levando a escolhas de gráficos incorretas (Ex: usar um gráfico de dispersão para dados categóricos).

## URL
[https://blacklabel.net/blog/dataviz-x-ai/how-to-write-better-prompts-to-improve-ai-chart-results/](https://blacklabel.net/blog/dataviz-x-ai/how-to-write-better-prompts-to-improve-ai-chart-results/)
