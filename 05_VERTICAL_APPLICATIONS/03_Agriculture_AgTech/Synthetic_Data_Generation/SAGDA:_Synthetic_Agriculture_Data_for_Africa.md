# SAGDA: Synthetic Agriculture Data for Africa

## Description

SAGDA é uma biblioteca de código aberto baseada em Python, desenvolvida para gerar, aumentar e validar conjuntos de dados agrícolas sintéticos, visando superar a escassez de dados na agricultura africana e em regiões com dados limitados. O toolkit utiliza distribuições estatísticas e modelos generativos (como VAEs e GANs) para simular dados realistas de clima, solo, rendimento de colheitas e uso de fertilizantes.

## Statistics

Artigo publicado em 2025 (arXiv:2506.13123v1). O uso do método 'augment' do SAGDA aumentou o volume de dados de treinamento em 145,2% e reduziu o MAPE (Erro Percentual Absoluto Médio) de um modelo de previsão de rendimento de quase 40% para 29,85% em dados de teste temporalmente disjuntos. A validação mostrou 99,85% de sobreposição estrutural entre dados sintéticos e originais no espaço PCA.

## Features

Geração de Dados (sagda.generate), Aumento de Dados (sagda.augment), Validação de Dados (sagda.validate), Visualização (sagda.visualize), Otimização (sagda.optimize) para cenários agrícolas, Simulação (sagda.simulate) de processos dinâmicos e Módulos de Modelo (sagda.model) pré-treinados.

## Use Cases

Previsão de rendimento de colheitas aprimorada por aumento de dados para mitigar a escassez temporal. Otimização de fertilizantes NPK (nitrogênio, fósforo, potássio) específicos para o local, resultando em um aumento médio de rendimento de mais de 500 kg/ha.

## Integration

Implementado como uma biblioteca Python (disponível no GitHub e PyPI) com dependências mínimas (pandas, numpy, scipy, tensorflow, scikit-learn). Instalação via `pip install sagda`. O código de exemplo para aumento de dados e otimização de fertilizantes é detalhado no artigo.

## URL

https://arxiv.org/abs/2506.13123