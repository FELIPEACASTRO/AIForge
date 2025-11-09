# Yahoo Finance Data

## Description
O Yahoo Finance Data é uma fonte de dados financeiros e de mercado amplamente utilizada, fornecendo informações históricas e em tempo real sobre ações, índices, moedas, commodities e outros ativos financeiros globais. Embora o Yahoo não ofereça uma API oficial e estável para uso público e gratuito, a comunidade de código aberto desenvolveu bibliotecas robustas, como `yfinance` (Python), que fazem a raspagem (scraping) dos dados do site, permitindo o acesso a um vasto repositório de séries temporais financeiras. É a principal fonte de dados para a maioria dos projetos de análise de mercado e aprendizado de máquina em finanças.

## Statistics
Amostras (Exemplo AAPL, Diário, Máximo): 165 pontos de dados. Período de tempo (Exemplo AAPL): 1984-12-01 a 2025-11-07. Estrutura de dados: Séries temporais financeiras (7 colunas principais por ativo). Versões: Acesso contínuo e em tempo real, sem versões estáticas. Cobertura: Milhares de ações, índices, moedas e commodities globais.

## Features
Dados de preços históricos (Open, High, Low, Close, Volume, Fechamento Ajustado), Dividendos e Splits. Cotações em tempo real (via API), Informações fundamentais (balanços, demonstrações de resultados), Notícias e análises de mercado. Suporte a diferentes granularidades (minuto, dia, semana, mês). Cobertura de milhares de ações, índices, moedas e commodities globais.

## Use Cases
Modelagem e previsão de preços de ações (Machine Learning e Deep Learning). Análise técnica e fundamentalista de ativos financeiros. Backtesting de estratégias de negociação. Pesquisa acadêmica em finanças e economia. Criação de dashboards e visualizações de mercado.

## Integration
O acesso mais comum é feito através de bibliotecas de terceiros como `yfinance` (Python), que 'raspam' os dados do site do Yahoo Finance.

**Instalação (Python):**
```bash
pip install yfinance
```

**Uso (Python):**
```python
import yfinance as yf
# Baixa dados históricos da Apple (AAPL)
data = yf.download('AAPL', start='2023-01-01', end='2024-01-01')
print(data.head())
```

Alternativamente, o acesso pode ser feito via APIs REST não oficiais ou a API do Manus Hub. O acesso direto via download de CSV no site é limitado e pode exigir uma assinatura premium.

## URL
[https://finance.yahoo.com/](https://finance.yahoo.com/)
