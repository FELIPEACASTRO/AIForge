# DoctorAgent-RL

## Description

DoctorAgent-RL é uma estrutura de aprendizado por reforço (RL) multiagente e colaborativa, projetada para revolucionar o diálogo clínico multi-turn. Ela modela consultas médicas como processos dinâmicos de tomada de decisão sob incerteza, permitindo que o agente médico otimize sua estratégia de questionamento para coleta adaptativa de informações. O objetivo é desenvolver estratégias de interação alinhadas com a lógica de raciocínio clínico, superando as limitações dos modelos estáticos de aprendizado supervisionado.

## Statistics

Precisão Diagnóstica Média: 58.9% (Tabela 5 do artigo). O modelo superou modelos existentes em capacidade de raciocínio multi-turn e desempenho diagnóstico final. O conjunto de dados MTMedDialog, com mais de 10.000 diálogos, foi construído para treinamento e avaliação.

## Features

Colaboração Multiagente (Agente Médico e Agente Paciente); Otimização Dinâmica de Estratégia via RL (Group Relative Policy Optimization - GRPO); Design de Recompensa Abrangente para avaliação de consulta; Integração de Conhecimento Médico; Utiliza o dataset MTMedDialog.

## Use Cases

Otimização de Consultas Médicas Multi-Turn; Melhoria da Precisão Diagnóstica em Ambientes de Diálogo; Redução do Risco de Diagnóstico Incorreto em Cenários de Pressão de Tempo; Alívio da Escassez de Mão de Obra Clínica.

## Integration

O código e os modelos estão disponíveis no GitHub e Huggingface. A integração envolve clonar o repositório, configurar o ambiente (seguindo o script setup_ragen.sh) e utilizar scripts de treinamento e avaliação em Bash. Exemplo de treinamento: `bash scripts_exp/doctor-agent-rl-dynamic.sh` (para Dynamic Turns + SFT Cold Start).

## URL

https://github.com/JarvisUSTC/DoctorAgent-RL