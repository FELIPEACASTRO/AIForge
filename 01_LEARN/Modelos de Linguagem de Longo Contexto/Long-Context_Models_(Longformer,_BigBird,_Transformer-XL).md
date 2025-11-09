# Long-Context Models (Longformer, BigBird, Transformer-XL)

## Description

O Longformer, BigBird e Transformer-XL são modelos de linguagem de longo contexto projetados para superar a limitação de atenção quadrática dos Transformers tradicionais. O Longformer e o BigBird utilizam mecanismos de atenção esparsa (janela local + global) para processar sequências de até 4096 tokens com complexidade linear. O Transformer-XL, por sua vez, usa um mecanismo de recorrência em nível de segmento e codificação posicional relativa para estender o contexto de forma quase ilimitada, sendo ideal para geração de texto coerente e longo.

## Statistics

Longformer e BigBird: Capacidade de sequência de até 4.096 tokens. Transformer-XL: Aprende dependência 450% mais longa que Transformers vanilla e é até 1.800x mais rápido na avaliação. Todos reduzem a complexidade de atenção de O(n²) para O(n) ou O(n*w).

## Features

Longformer: Atenção esparsa com janela local e global, complexidade O(n*w). BigBird: Atenção esparsa baseada em blocos (local, aleatória, global), complexidade O(n). Transformer-XL: Recorrência em nível de segmento, codificação posicional relativa, sem limite fixo de comprimento de sequência.

## Use Cases

Resposta a perguntas em documentos longos, sumarização de documentos, classificação de texto em documentos longos, modelagem de linguagem e geração de texto coerente em contextos estendidos, aplicações genômicas.

## Integration

Integração via Hugging Face Transformers. O Longformer requer a criação de uma `global_attention_mask`. O BigBird é ideal para tarefas de QA em documentos longos. O Transformer-XL, devido ao modo de manutenção, requer o uso de um checkpoint e revisão específicos com a variável de ambiente `TRUST_REMOTE_CODE` ativada. Exemplos de código detalhados para cada modelo foram fornecidos.

## URL

Diversas (Hugging Face Docs)