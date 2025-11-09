# AlphaFold (DeepMind/Isomorphic Labs)

## Description

AlphaFold é um sistema de inteligência artificial desenvolvido pelo Google DeepMind e Isomorphic Labs que resolveu o 'problema do dobramento de proteínas', prevendo a estrutura 3D de proteínas a partir de sua sequência de aminoácidos com precisão notável. O AlphaFold 3 expandiu essa capacidade para prever a estrutura conjunta e as interações de complexos biomoleculares, incluindo proteínas, DNA, RNA, ligantes e anticorpos, revolucionando a proteômica estrutural e a descoberta de medicamentos.

## Statistics

Mais de 200 milhões de estruturas de proteínas previstas (cobrindo quase todo o proteoma catalogado). Mais de 2 milhões de pesquisadores em mais de 190 países utilizam o AlphaFold Protein Structure Database (AlphaFold DB). Estima-se que tenha economizado até 1 bilhão de anos de pesquisa. O AlphaFold 3 demonstrou uma precisão 50% superior na previsão de interações proteína-ligante em comparação com métodos anteriores.

## Features

Previsão de Estrutura 3D de Proteínas (AlphaFold 2). Previsão de Estrutura Conjunta de Complexos Biomoleculares (AlphaFold 3). Modelagem de Interações Proteína-Proteína (PPI), Proteína-Ácido Nucleico (DNA/RNA) e Proteína-Ligante. Alta precisão na previsão de estruturas, medida pelo score de confiança (pLDDT). Código-fonte e pesos do modelo AlphaFold 3 disponíveis para uso acadêmico.

## Use Cases

Aceleração da Descoberta de Medicamentos e Alvos Terapêuticos. Engenharia de Enzimas para Biocatálise e Reciclagem de Plásticos. Compreensão de Mecanismos de Doenças (e.g., Parkinson, Câncer) através da estrutura de proteínas defeituosas. Design de Vacinas e Anticorpos. Otimização de Culturas Agrícolas e Combate a Patógenos.

## Integration

O AlphaFold DB oferece uma API pública para acesso programático às mais de 200 milhões de estruturas previstas. O código-fonte do AlphaFold 2 e 3 (para uso acadêmico) está disponível no GitHub, permitindo a execução local via Docker ou scripts Python. Ferramentas como o AlphaPulldown facilitam a previsão de interações proteína-proteína (PPI) usando o AlphaFold-Multimer. A integração com o AlphaFold Server (AlphaFold 3) é feita via interface web para uso comercial e acadêmico.

## URL

https://deepmind.google/science/alphafold/