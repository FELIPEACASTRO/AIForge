# Horovod

## Description

Horovod é um framework de código aberto para treinamento distribuído de deep learning, desenvolvido pela Uber Engineering. Sua proposta de valor única é tornar o treinamento distribuído de modelos de deep learning (em frameworks como TensorFlow, Keras, PyTorch e Apache MXNet) **rápido, fácil e portátil**. Ele alcança isso minimizando as alterações de código necessárias para migrar um script de treinamento de GPU única para um ambiente distribuído, utilizando o algoritmo **Ring All-reduce** para comunicação eficiente de gradientes entre os nós.

## Statistics

- **Eficiência de Escalonamento:** Alcançou **quase 90% de eficiência de escalonamento** para modelos como Inception V3 e ResNet-101 em 128 GPUs.
- **Algoritmo Central:** Baseado no algoritmo **Ring All-reduce**, que otimiza a comunicação de gradientes, superando em desempenho abordagens mais antigas como o servidor de parâmetros.
- **Origem:** Desenvolvido e mantido pela **Uber Engineering**.

## Features

- **Suporte a Múltiplos Frameworks:** Compatibilidade com TensorFlow, Keras, PyTorch e Apache MXNet.
- **Fácil de Usar:** Requer apenas algumas linhas de código para adaptar um script de treinamento existente para o modo distribuído.
- **Otimização de Comunicação:** Implementa o algoritmo **Ring All-reduce** para agregação de gradientes, o que é crucial para o desempenho em larga escala.
- **Portabilidade:** Baseado em conceitos do MPI (Message Passing Interface), facilitando a execução em diversos ambientes de cluster.
- **HorovodRunner:** API geral para executar cargas de trabalho de deep learning distribuído em clusters Spark (como no Azure Databricks).

## Use Cases

- **Escalonamento de Treinamento:** O caso de uso primário é escalar scripts de treinamento de GPU única para múltiplos nós ou GPUs, reduzindo drasticamente o tempo de treinamento para modelos grandes e datasets massivos.
- **Pesquisa de Alto Desempenho:** Usado em ambientes de pesquisa de alto desempenho (como NASA/HECC) para acelerar a iteração de modelos de deep learning.
- **Integração com Plataformas de ML:** Utilizado em conjunto com plataformas como **Ray Train**, **AWS SageMaker** e **Azure Databricks** para orquestração e gerenciamento de treinamento distribuído em ambientes de produção.

## Integration

A integração com Horovod é direta e requer apenas algumas modificações no código de treinamento existente. O princípio é inicializar o Horovod, fixar o processo à GPU local e envolver o otimizador com `hvd.DistributedOptimizer`.

**Exemplo de Integração com PyTorch:**

```python
import torch
import horovod.torch as hvd

# 1. Inicializar Horovod
hvd.init()

# 2. Fixar a GPU ao processo local
torch.cuda.set_device(hvd.local_rank())

# 3. Definir o modelo e o otimizador (exemplo)
model = ...
optimizer = optim.Adadelta(model.parameters())

# 4. Envolver o otimizador com Horovod
optimizer = hvd.DistributedOptimizer(optimizer,
                                     named_parameters=model.named_parameters())

# 5. Transmitir (Broadcast) os pesos iniciais do rank 0 para todos os outros
hvd.broadcast_parameters(model.state_dict(), root_rank=0)

# O loop de treinamento segue o padrão, onde optimizer.step() fará o All-reduce
```

**Exemplo de Integração com TensorFlow/Keras:**

```python
import tensorflow as tf
import horovod.tensorflow.keras as hvd

# 1. Inicializar Horovod
hvd.init()

# 2. Configurar GPUs visíveis
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

# 3. Envolver o otimizador com Horovod
opt = tf.optimizers.Adadelta(1.0 * hvd.size())
opt = hvd.DistributedOptimizer(opt)

# 4. Compilar e treinar o modelo
model.compile(optimizer=opt, ...)
model.fit(...)
```

## URL

https://horovod.ai/