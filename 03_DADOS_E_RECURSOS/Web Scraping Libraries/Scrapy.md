# Scrapy

## Description

Scrapy é um framework de web scraping e crawling de código aberto, rápido e de alto nível, escrito em Python. Ele é projetado para extrair dados usando seletores XPath ou CSS, processá-los e armazená-los de forma eficiente. Sua arquitetura assíncrona baseada no Twisted (motor de rede assíncrona) permite o processamento paralelo de requisições, tornando-o ideal para projetos de grande escala.

## Statistics

Framework completo e robusto. Sua arquitetura assíncrona o torna significativamente mais rápido e menos intensivo em recursos de CPU/memória do que o Selenium para sites estáticos. É a escolha preferida para projetos de web crawling em larga escala e alta velocidade. Embora tenha menos usuários totais que o Beautiful Soup, é dominante no nicho de frameworks de scraping.

## Features

Arquitetura de aranha (Spider) modular; Suporte a seletores XPath e CSS; Pipelines de Item para processamento e armazenamento de dados; Middlewares de Spider e Downloader para manipulação de requisições e respostas (ex: rotação de proxy, manipulação de cookies); Agendamento de requisições e processamento assíncrono; Suporte a exportação para JSON, CSV e XML.

## Use Cases

Web crawling em larga escala; Monitoramento de preços e dados de e-commerce; Extração de dados para fins de pesquisa e análise de mercado; Criação de motores de busca e agregadores de conteúdo.

## Integration

A integração é feita através da criação de um projeto Scrapy, definindo 'Spiders' para o crawling e 'Pipelines' para o processamento. Exemplo de um Spider básico:\n```python\nimport scrapy\n\nclass QuotesSpider(scrapy.Spider):\n    name = 'quotes'\n    start_urls = [\n        'http://quotes.toscrape.com/',\n    ]\n\n    def parse(self, response):\n        for quote in response.css('div.quote'):\n            yield {\n                'text': quote.css('span.text::text').get(),\n                'author': quote.css('small.author::text').get(),\n            }\n```

## URL

https://scrapy.org/