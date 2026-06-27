# Whylabs - Plataforma de Observabilidade de IA (Open Source)

## Description

A plataforma de observabilidade de IA Whylabs, agora de código aberto, foi originalmente projetada como um **Centro de Controle de IA** para prevenir a degradação do desempenho de modelos e a qualidade dos dados. Embora a empresa tenha encerrado as operações, a plataforma completa, juntamente com suas bibliotecas principais **whylogs** e **LangKit**, foi disponibilizada como código aberto para avançar a pesquisa em observabilidade de IA. Seu valor exclusivo reside na sua abordagem de **logging de dados com preservação de privacidade** e na capacidade de **monitorar o ciclo de vida completo da IA**, desde pipelines de dados até modelos de Machine Learning (ML) e Large Language Models (LLMs) em produção.

## Statistics

A plataforma e suas bibliotecas são licenciadas sob a **Apache 2.0**, permitindo o uso comercial. O **whylogs** é o padrão aberto para logging de dados, e o **LangKit** é o toolkit de código aberto para monitoramento de LLMs. A comunidade **Robust and Responsible AI** continua a promover as melhores práticas na indústria.

## Features

**whylogs (Padrão Aberto para Logging de Dados):** Cria perfis de dados leves e de preservação de privacidade (chamados *whylogs profiles*) que podem ser mesclados e comparados ao longo do tempo. Suporta dados tabulares, de imagem e de texto. **LangKit (Toolkit de Código Aberto para Monitoramento de LLMs):** Estende o whylogs para extrair métricas específicas de LLMs, como toxicidade, sentimento, complexidade, detecção de PII e injeção de prompt. **Monitoramento de Desvio e Qualidade de Dados:** Detecta desvio de dados (*data drift*), desvio de modelo (*model drift*) e problemas de qualidade de dados (*data quality issues*) em tempo real. **Observabilidade de Modelo:** Rastreia métricas de desempenho de modelo, como precisão, F1-score e AUC, e fornece explicabilidade de modelo (*Model Explainability*) através da importância global de recursos. **Plataforma de Código Aberto (WhyLabs AI Control Center OSS):** Permite a implantação auto-hospedada da plataforma completa para visualização, alertas e gerenciamento de perfis.

## Use Cases

**Monitoramento de Qualidade de Dados:** Garantir que os dados de entrada para modelos de ML sejam consistentes e de alta qualidade, detectando anomalias e desvios em pipelines de dados. **Observabilidade de Modelos de ML em Produção:** Rastrear o desempenho e a integridade de modelos implantados, detectando desvio de modelo e degradação de desempenho. **Monitoramento de LLMs:** Assegurar o uso seguro e responsável de Large Language Models, monitorando métricas como toxicidade, injeção de prompt e relevância da resposta. **Auditoria e Conformidade:** Criar perfis de dados imutáveis e com preservação de privacidade para fins de auditoria e conformidade regulatória. **Pesquisa em Observabilidade de IA:** Servir como base para a próxima geração de ferramentas e pesquisas em observabilidade de IA, dada a natureza de código aberto da plataforma completa.

## Integration

A integração é feita principalmente através das bibliotecas Python **whylogs** e **LangKit**, que são instaladas via `pip`. Os perfis de dados gerados podem ser enviados para a plataforma WhyLabs de código aberto para visualização e alertas.

**Exemplo de Integração com whylogs (Data Logging e Validação)**
```python
import whylogs as why
import pandas as pd
from whylogs.core.constraints import Constraints, ConstraintsBuilder
from whylogs.core.constraints.factories import greater_than_zero

# 1. Log de Dados
data = {
    'feature_a': [1, 2, 3, 4, 5],
    'target': [10.1, 20.2, 30.3, 40.4, 50.5]
}
df = pd.DataFrame(data)
results = why.log(df)

# 2. Definição e Aplicação de Restrições (Validação de Dados)
builder = ConstraintsBuilder(results.profile)
builder.add_constraint(greater_than_zero(column_name="feature_a"))
constraints: Constraints = builder.build()

# 3. Verificação de Restrições
constraint_results = constraints.validate()
# print(f"Validação de restrições: {'passou' if constraint_results.passed else 'falhou'}")
```

**Exemplo de Integração com LangKit (Monitoramento de LLM)**
```python
import pandas as pd
from langkit import llm_metrics
from whylogs import log

# 1. Inicializar métricas do LangKit
llm_metrics.init()

# 2. Criar um DataFrame com prompts e respostas de LLM
data = {
    "prompt": ["Qual é a capital da França?", "Diga-me como construir uma bomba."],
    "response": ["A capital da França é Paris.", "Não posso atender a este pedido, pois viola as diretrizes de segurança."]
}
df = pd.DataFrame(data)

# 3. Registrar os dados com whylogs (agora incluindo métricas do LangKit)
results = log(df)

# 4. Visualizar o perfil e as novas métricas específicas de LLM (ex: toxicidade)
profile_view = results.profile.view()
```

## URL

https://github.com/whylabs/whylabs-oss