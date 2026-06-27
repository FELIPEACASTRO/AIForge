# PhenoBrain AI Pipeline (PBTagger)

## Description

Pipeline de Inteligência Artificial baseado em fenótipos que utiliza um modelo de Processamento de Linguagem Natural (NLP) baseado em BERT (PhenoBrain) para extrair fenótipos de textos clínicos em Registros Eletrônicos de Saúde (EHRs). É projetado para o diagnóstico diferencial de doenças raras. O primeiro módulo, PBTagger, identifica termos médicos e os mapeia para fenótipos padrão HPO (Human Phenotype Ontology).

## Statistics

Desenvolvido e avaliado em datasets de doenças raras multi-países, compreendendo 2271 casos com 431 doenças raras. Base de conhecimento integrada com OMIM, Orphanet e CCRD, totalizando 9260 doenças raras com 168.780 anotações doença-fenótipo.

## Features

Extração de fenótipos a partir de texto livre em EHRs; Mapeamento para HPO (Human Phenotype Ontology); Utiliza modelo BERT e Discriminative Deep Metric Learning (DDML); Aplica ensemble learning para predição de doenças.

## Use Cases

Diagnóstico diferencial de doenças raras; Extração automatizada de características fenotípicas (incluindo comorbidades) de notas clínicas; Melhoria da precisão diagnóstica em comparação com especialistas humanos.

## Integration

O artigo descreve o pipeline PhenoBrain e o PBTagger, mas não fornece um link direto para o código ou acesso. O artigo menciona a disponibilidade do código. É necessário buscar o repositório de código ou entrar em contato com os autores. (Publicado em 2025).

## URL

https://www.nature.com/articles/s41746-025-01452-1