# Weights & Biases (W&B) - Experiment Tracking and Model Management

## Description

Weights & Biases (W&B) é uma plataforma de MLOps (Machine Learning Operations) que fornece um conjunto de ferramentas para rastreamento de experimentos, visualização de métricas, gerenciamento de modelos e colaboração para equipes de Machine Learning. Sua proposta de valor única reside em oferecer uma solução completa e centralizada para o ciclo de vida do desenvolvimento de modelos de IA, desde o treinamento inicial até a implantação e monitoramento em produção. O W&B permite que pesquisadores e engenheiros transformem experimentos caóticos em um fluxo de trabalho organizado e reproduzível, acelerando a iteração e a construção de modelos de maior qualidade de forma mais rápida [1] [2]. A plataforma é amplamente adotada por sua capacidade de escalar insights de um único pesquisador para equipes inteiras e de uma única máquina para milhares de execuções de treinamento [3].

## Statistics

*   **Usuários:** Mais de 900.000 usuários ativos.
*   **Empresas:** Mais de 1.000 empresas, incluindo startups de IA de ponta, instituições de pesquisa e grandes marcas (como Canva, Microsoft, Toyota, OpenAI, IBM, Pinterest, LG AI Research, Siemens, Festo, entre outras) [1].
*   **Aceleração de Experimentos:** Empresas como Festo relataram uma redução no tempo de configuração de novos experimentos de 8 horas para 20-30 minutos com o uso do W&B [1].
*   **Escalabilidade:** Utilizado para escalar insights de um único pesquisador para equipes inteiras e de uma única máquina para milhares de execuções de treinamento [3].

## Features

O W&B oferece um conjunto modular de ferramentas que cobrem todo o ciclo de vida do MLOps:
*   **W&B Experiments (Runs)**: Rastreamento automático e visualização de métricas, hiperparâmetros, gradientes e recursos de sistema (CPU/GPU) para cada execução de treinamento.
*   **W&B Sweeps**: Otimização automatizada de hiperparâmetros (HPO) e busca de arquitetura de modelo (NAS) usando métodos como Grid Search, Random Search e Hyperband.
*   **W&B Artifacts**: Versionamento e gerenciamento de pipelines de dados e modelos, garantindo a rastreabilidade e a reprodutibilidade de todo o fluxo de trabalho.
*   **W&B Tables**: Visualização e exploração de dados, previsões e resultados de avaliação em formato tabular, permitindo análises ricas e interativas.
*   **W&B Reports**: Documentação e compartilhamento de insights de IA de forma colaborativa, transformando experimentos em relatórios narrativos e reproduzíveis.
*   **W&B Registry**: Gerenciamento de modelos (Model Registry) para versionamento, promoção de modelos para produção e manutenção de linhagem completa.
*   **W&B Weave**: Ferramenta para rastreamento, avaliação e depuração de aplicações de LLM (Large Language Model) e sistemas agenticos, incluindo traços (Traces) e avaliações rigorosas.
*   **W&B Inference**: Acesso a modelos de fundação de código aberto através de uma API compatível com OpenAI, facilitando a experimentação e o uso em produção [2] [4].

## Use Cases

*   **Otimização de Hiperparâmetros (HPO)**: Uso do W&B Sweeps para automatizar a busca pelo melhor conjunto de hiperparâmetros, como demonstrado em projetos de pesquisa de doutorado [7].
*   **Desenvolvimento de Veículos Autônomos**: Empresas como a Woven by Toyota utilizam o W&B Weave para rastrear, avaliar e depurar agentes de IA de vídeo, identificando e corrigindo bugs mais rapidamente [1].
*   **Pesquisa e Desenvolvimento de LLMs**: Equipes como a Aleph Alpha e a LG AI Research utilizam o W&B para comparar execuções, agregar resultados e tomar decisões intuitivas sobre o que funciona bem em seus projetos de modelos de linguagem avançados [1].
*   **Gerenciamento de Modelos em Produção**: Empresas como a Canva e a Pinterest utilizam o W&B Registry para simplificar o gerenciamento de modelos, versionamento e promoção para produção, garantindo que apenas modelos prontos sejam considerados [1].
*   **Aceleração de Experimentação em Dados Sintéticos**: A Gretel alcançou uma velocidade de experimentação 10x maior, passando de 5-10 experimentos para 50-100 experimentos por bloco de computação, usando as ferramentas de log e avaliação do W&B [1].
*   **Monitoramento de Robótica e Visão Computacional de Borda**: Empresas como a Siemens e a Captur utilizam o W&B para monitorar métricas de treinamento, funções de perda e uso de GPU em tempo real, garantindo a confiabilidade e o desempenho de robôs de armazém e sistemas de visão de borda [1].

## Integration

A integração com W&B é tipicamente realizada através do SDK Python `wandb`, que é leve e fácil de adicionar a qualquer script de ML. O fluxo básico envolve a inicialização de uma execução (`run`) e o uso de funções de log para registrar dados.

**Exemplo de Integração com PyTorch:**

```python
import wandb
import torch
import torch.nn.functional as F

# 1. Inicializa uma nova execução (run) do W&B
with wandb.init(project="meu-projeto-pytorch", config=args) as run:
    
    model = ...  # Configuração do seu modelo PyTorch
    
    # 2. Rastreia automaticamente gradientes e a arquitetura do modelo
    run.watch(model, log_freq=100)
    
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        # 3. Registra métricas a cada intervalo
        if batch_idx % args.log_interval == 0:
            run.log({"loss": loss.item()})

    # 4. Registra imagens ou tabelas de dados
    images_t = ...  # Tensor de imagens
    run.log({"exemplos": [wandb.Image(im) for im in images_t]})
```

**Integração com LLMs (W&B Weave):**
O W&B Weave permite rastrear e avaliar o desempenho de LLMs. A integração é feita através da biblioteca `weave`, que pode ser usada para envolver chamadas de LLM e registrar traços de execução, custos e avaliações [4].

**Integrações Nativas:**
O W&B possui integrações de primeira classe com os principais frameworks de ML, incluindo PyTorch, TensorFlow, Keras, Scikit-learn, Hugging Face, e plataformas de nuvem como Azure OpenAI [5] [6].

## URL

https://wandb.ai/