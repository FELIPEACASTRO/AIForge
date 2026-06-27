# Game Analytics Datasets (via GameAnalytics PipelineIQ)

## Description
O termo "Game Analytics Datasets" refere-se primariamente aos dados brutos de eventos de jogo (raw event data) coletados e disponibilizados pela plataforma GameAnalytics, através de seu produto **PipelineIQ** (especificamente a funcionalidade **Data Export**). Estes datasets não são um conjunto de dados estático e público, mas sim um fluxo de dados em tempo real e histórico que os desenvolvedores de jogos podem exportar para seus próprios ambientes de nuvem (como AWS S3, Google Cloud Storage ou Azure Blob Storage) para análise aprofundada e treinamento de modelos de Machine Learning [1].

Os dados exportados incluem todos os eventos de jogabilidade, eventos de design, eventos de negócios, eventos de anúncios e dimensões personalizadas que o desenvolvedor rastreia em seu jogo. A GameAnalytics atua como um pipeline de dados, permitindo que os estúdios de jogos mantenham a propriedade total e o controle sobre seus dados brutos para análises personalizadas e de longo prazo [2].

A plataforma também oferece o **Data Warehouse**, que contém dois conjuntos de dados principais para clientes do PipelineIQ: **Checkpoints Data Sets** (dados agregados diários com retenção de até 1 ano) e **Event Tables** (dados de eventos brutos com retenção de até 30 dias), que podem ser consultados diretamente no BigQuery [3].

## Statistics
Os datasets não possuem um tamanho ou contagem de amostras fixos, pois são gerados continuamente pelo jogo do desenvolvedor.

*   **Tamanho:** Variável, dependendo do volume de eventos gerados pelo jogo (milhões a bilhões de eventos por dia).
*   **Retenção (Data Export):** Ilimitada, pois os dados são armazenados na infraestrutura de nuvem do cliente, sujeitos às suas políticas internas de retenção.
*   **Retenção (Data Warehouse - BigQuery):**
    *   **Checkpoints Data Sets:** Até 1 ano de dados agregados diários.
    *   **Event Tables:** Até 30 dias de dados de eventos brutos [3].
*   **Versões:** Os dados refletem a versão atual do jogo e a estrutura de eventos definida pelo desenvolvedor. A GameAnalytics garante que todos os campos personalizados sejam retidos no fluxo de dados [2].

## Features
Os principais recursos dos datasets de Game Analytics da GameAnalytics incluem:

*   **Dados Brutos em Tempo Real:** Exportação de todos os eventos de jogabilidade em tempo real (streaming) para o ambiente de nuvem do cliente.
*   **Flexibilidade Total:** Não há esquemas rígidos ou predefinidos. Os dados incluem todos os campos, eventos e dimensões personalizados definidos pelo desenvolvedor.
*   **Eventos Detalhados:** Captura de eventos de design (ações do jogador), eventos de negócios (transações), eventos de recursos (economia do jogo), eventos de anúncios (impressões e cliques) e eventos de erro [3].
*   **Propriedade e Controle:** O desenvolvedor mantém a propriedade total dos dados, armazenando-os em sua própria infraestrutura de nuvem para análise de longo prazo.
*   **Compatibilidade com BI:** Facilidade de integração com ferramentas de Business Intelligence (BI) como Tableau, Power BI ou Looker [2].

## Use Cases
Os datasets de Game Analytics são cruciais para aplicações avançadas de dados em jogos:

*   **Business Intelligence (BI) Personalizado:** Criação de dashboards e relatórios de BI sob medida, que vão além das análises padrão da plataforma GameAnalytics [2].
*   **Análise de Comportamento do Jogador:** Análise aprofundada de padrões de interação, progressão diária, funis de conversão e identificação de pontos de abandono (churn) [2].
*   **Otimização de Monetização:** Análise detalhada do engajamento com anúncios e da receita gerada para otimizar a colocação e frequência de anúncios [2].
*   **Machine Learning e IA:** Treinamento de modelos de IA para predição de abandono (churn prediction), segmentação de jogadores, personalização de ofertas e sistemas de recomendação [1].
*   **Análise Histórica de Longo Prazo:** Realização de análises de tendências e mudanças no comportamento do jogador ao longo de anos, sem depender das políticas de retenção de terceiros [2].

## Integration
A integração e o uso dos datasets de Game Analytics da GameAnalytics são realizados através do produto **PipelineIQ** (funcionalidade **Data Export**).

1.  **Assinatura:** É necessário ser um cliente do PipelineIQ (geralmente Pro ou Enterprise) da GameAnalytics.
2.  **Configuração do Destino:** O desenvolvedor configura um destino de armazenamento em nuvem (AWS S3, Google Cloud Storage, Azure Blob Storage) para receber o fluxo de dados.
3.  **Exportação:** A GameAnalytics transmite os dados de jogabilidade em tempo real para o destino configurado, com exportações ocorrendo a cada 15 minutos, desde que haja novos dados [3].
4.  **Análise:** Os dados brutos são então acessíveis diretamente no ambiente de nuvem do cliente, onde podem ser processados, transformados e carregados em data warehouses ou usados para treinar modelos de Machine Learning.

Para clientes que usam o **Data Warehouse**, o acesso é feito via consultas SQL no Google BigQuery, onde os dados de Checkpoints e Event Tables são populados [3].

## URL
[https://www.gameanalytics.com/pipelineiq/data-export](https://www.gameanalytics.com/pipelineiq/data-export)
