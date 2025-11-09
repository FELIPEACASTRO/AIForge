# National Incident-Based Reporting System (NIBRS) Data

## Description
O National Incident-Based Reporting System (NIBRS) é o principal sistema de coleta de dados de crimes do FBI nos Estados Unidos, substituindo o antigo Summary Reporting System (SRS) do Uniform Crime Reporting (UCR) Program. O NIBRS coleta dados detalhados em nível de incidente sobre crimes, incluindo informações sobre vítimas, agressores, relacionamento entre eles, propriedades envolvidas e armas utilizadas. É a fonte de dados mais abrangente e granular sobre estatísticas de crimes nos EUA, sendo fundamental para pesquisas em criminologia e aplicações de inteligência artificial.

## Statistics
O dataset é massivo, com milhões de registros por ano. Por exemplo, o relatório de 2022 continha mais de 11 milhões de ocorrências criminais. Os arquivos de dados mestres anuais são de tamanho considerável (centenas de megabytes ou gigabytes) e são disponibilizados em formato de texto ASCII de comprimento fixo, compactados em WinZip. Os dados são atualizados anualmente, com a versão mais recente disponível (em 2025) sendo a de 2024.

## Features
Dados em nível de incidente (mais detalhados que dados sumários); Coleta informações sobre 81 tipos de crimes (contra 10 do sistema anterior); Inclui detalhes sobre o contexto do crime (hora, local, armas, valor da propriedade); Informações demográficas detalhadas sobre vítimas e agressores (idade, sexo, raça, etnia); Permite a análise de crimes múltiplos dentro de um único incidente.

## Use Cases
Previsão de crimes e 'hotspots' criminais usando Machine Learning; Análise de tendências criminais e alocação de recursos policiais; Pesquisa acadêmica em criminologia e sociologia; Desenvolvimento de modelos de perfil de agressores; Avaliação de políticas públicas de segurança.

## Integration
Os dados podem ser acessados e baixados de várias formas: 1. **Crime Data API:** Serviço web somente leitura que retorna dados em JSON ou CSV. 2. **Downloads Diretos:** Arquivos mestres anuais (Master Files) em formato de texto ASCII compactado (requer conhecimento de programação para extração). 3. **Ferramenta de Descoberta de Dados (Data Discovery Tool):** Permite a criação de consultas personalizadas e download de subconjuntos de dados em CSV. A documentação técnica completa (NIBRS Data Dictionary) está disponível no portal do FBI.

## URL
[https://cde.ucr.cjis.gov/](https://cde.ucr.cjis.gov/)
