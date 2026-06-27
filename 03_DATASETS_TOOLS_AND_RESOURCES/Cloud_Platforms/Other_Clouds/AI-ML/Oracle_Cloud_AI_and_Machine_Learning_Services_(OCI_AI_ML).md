# Oracle Cloud AI and Machine Learning Services (OCI AI/ML)

## Description

A Oracle Cloud Infrastructure (OCI) oferece um conjunto abrangente de serviços de Inteligência Artificial (IA) e Machine Learning (ML) projetados para o ambiente corporativo. A proposta de valor central reside na **IA Embarcada em toda a Pilha** (Full-Stack Embedded AI), integrando capacidades de IA/ML em aplicações (Fusion Apps), na plataforma de dados (Autonomous Database, Vector Search) e na infraestrutura (OCI Supercluster). Os serviços são divididos em três categorias principais: **OCI AI Services** (modelos pré-construídos), **OCI Generative AI** (modelos de linguagem grandes e agentes de IA) e **OCI Data Science** (plataforma de ML gerenciada). A OCI enfatiza a **Soberania de IA**, permitindo que as empresas atendam aos requisitos de segurança e privacidade de dados, e oferece uma infraestrutura de alto desempenho para cargas de trabalho de IA de ponta [1] [2] [3].

## Statistics

**Escalabilidade da Infraestrutura:** OCI Supercluster suporta até **131.072 GPUs** (NVIDIA B200) para treinamento de modelos de fronteira [5]. **Latência de Rede:** Rede de cluster RDMA de ultrabaixa latência de **2.5 a 9.1 microssegundos** [5]. **Desempenho:** Resultados de benchmark **MLPerf Inference 5.0** demonstram desempenho excepcional em inferência [7]. **Custo-Benefício:** Alegações de até **220% melhor preço** em VMs com GPU em comparação com outros provedores de nuvem [5]. **Disponibilidade:** A maioria dos OCI AI Services oferece um **nível gratuito** (free pricing tier) [1].

## Features

**OCI Generative AI:** Serviço totalmente gerenciado com modelos de linguagem grandes (LLMs) de código aberto e proprietários (incluindo Cohere e, em breve, Google Gemini) [4]. Suporta ajuste fino (fine-tuning) e o novo **Agent Hub** para criação e implantação de agentes de IA. **OCI AI Services:** Modelos pré-construídos e personalizáveis para tarefas específicas, como OCI Vision (visão computacional), OCI Language (NLP), OCI Speech (transcrição) e OCI Document Understanding (processamento de documentos) [1]. **OCI Data Science:** Plataforma de ML gerenciada baseada em JupyterLab, com suporte a MLOps (pipelines, implantação e monitoramento de modelos), AutoML e bibliotecas de código aberto (TensorFlow, PyTorch, Scikit-learn) [3]. **OCI AI Infrastructure:** Infraestrutura de alto desempenho com OCI Supercluster, GPUs NVIDIA (Blackwell, Hopper) e AMD (MI300X), e rede RDMA de ultrabaixa latência [5].

## Use Cases

**Setor de Saúde:** Previsão de risco de readmissão de pacientes usando OCI Data Science [3]. **Varejo:** Previsão do Valor de Vida Útil do Cliente (CLV) e otimização de campanhas de marketing [3]. **Manufatura:** Manutenção preditiva e detecção de anomalias em dados de sensores [3]. **Finanças:** Detecção de fraudes em tempo real usando modelos de ML [3]. **Aplicações Empresariais:** IA embarcada nas Fusion Cloud Applications para otimizar finanças, RH e cadeia de suprimentos [2]. **Desenvolvimento de Agentes:** Criação de agentes de IA para automação de tarefas e suporte ao cliente via Agent Hub [4].

## Integration

A integração é primariamente realizada via **APIs REST** e **SDKs** (Python, Java, etc.). Para o OCI Generative AI, o SDK Python é o método preferido para interagir com os modelos. O OCI Data Science se integra com o ecossistema Python de código aberto. O **Autonomous Database** se integra com LLMs via **Select AI**, permitindo consultas em linguagem natural (NL2SQL) [6].

**Exemplo de Integração (OCI Generative AI - Python SDK):**
```python
import oci
from oci.generative_ai import GenerativeAiClient
from oci.generative_ai.models import GenerateTextDetails

# Configuração do cliente OCI (assumindo autenticação por Instance Principal ou config file)
config = oci.config.from_file()
generative_ai_client = GenerativeAiClient(config)

# Detalhes da requisição
generate_text_details = GenerateTextDetails(
    model_id="cohere.command", # Exemplo de modelo
    prompt="Escreva um resumo de 50 palavras sobre os serviços de IA da Oracle Cloud.",
    max_tokens=100,
    temperature=0.7
)

# Chamada da API
response = generative_ai_client.generate_text(generate_text_details)

# Imprimir o resultado
print(response.data.generated_texts[0].text)
```

## URL

https://www.oracle.com/artificial-intelligence/