# SHEPHERD

## Description

SHEPHERD é uma abordagem de deep learning para diagnóstico multifacetado de doenças raras, guiada por conhecimento existente de doenças, fenótipos e genes. É o primeiro método de deep learning para diagnóstico individualizado de doenças genéticas raras. Ele supera as limitações de dados escassos (few-shot) através de um treinamento eficiente em rótulos via aprendizado métrico baseado em conhecimento (knowledge-grounded metric learning), projetando fenótipos de pacientes em um espaço de incorporação (embedding space) otimizado pelo conhecimento mais amplo de fenótipos e genes.

## Statistics

Avaliado em uma coorte externa de 465 pacientes representando 299 doenças na Undiagnosed Diseases Network (UDN). 79% dos genes e 83% das doenças estavam representados em apenas um único paciente. Descoberta de genes causais: genes causais previstos em rank = 3.52 em média. Identifica o gene causal correto em 40% dos pacientes em 16 áreas de doenças na UDN.

## Features

Diagnóstico multifacetado; Descoberta de genes causais; Recuperação de 'pacientes-semelhantes-a-mim' (patients-like-me) com o mesmo gene ou doença; Caracterização interpretável de apresentações de doenças novas; Treinamento eficiente em rótulos via aprendizado métrico baseado em conhecimento.

## Use Cases

Acelerar o diagnóstico de pacientes com doenças raras; Auxiliar na identificação de genes candidatos após análise de sequenciamento genômico; Caracterizar doenças e encontrar outros pacientes com a mesma causa genética.

## Integration

O código-fonte e a implementação estão disponíveis no GitHub (mims-harvard/SHEPHERD). A integração envolve a projeção de fenótipos de pacientes em um espaço de incorporação para nomear genes e doenças.

## URL

https://zitniklab.hms.harvard.edu/projects/SHEPHERD/