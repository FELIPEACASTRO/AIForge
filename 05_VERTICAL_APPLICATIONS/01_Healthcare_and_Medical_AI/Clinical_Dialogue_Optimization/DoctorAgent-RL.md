# DoctorAgent-RL

## Description

DoctorAgent-RL is a multi-agent, collaborative reinforcement learning (RL) framework designed to revolutionize multi-turn clinical dialogue. It models medical consultations as dynamic decision-making processes under uncertainty, enabling the doctor agent to optimize its questioning strategy for adaptive information gathering. The goal is to develop interaction strategies aligned with clinical reasoning logic, overcoming the limitations of static supervised learning models.

## Statistics

Average Diagnostic Accuracy: 58.9% (Table 5 of the paper). The model outperformed existing models in multi-turn reasoning capability and final diagnostic performance. The MTMedDialog dataset, with more than 10,000 dialogues, was built for training and evaluation.

## Features

Multi-Agent Collaboration (Doctor Agent and Patient Agent); Dynamic Strategy Optimization via RL (Group Relative Policy Optimization - GRPO); Comprehensive Reward Design for consultation evaluation; Medical Knowledge Integration; Uses the MTMedDialog dataset.

## Use Cases

Optimization of Multi-Turn Medical Consultations; Improved Diagnostic Accuracy in Dialogue Settings; Reduced Risk of Misdiagnosis in Time-Pressured Scenarios; Alleviation of Clinical Workforce Shortages.

## Integration

The code and models are available on GitHub and Huggingface. Integration involves cloning the repository, setting up the environment (following the setup_ragen.sh script), and using Bash training and evaluation scripts. Training example: `bash scripts_exp/doctor-agent-rl-dynamic.sh` (for Dynamic Turns + SFT Cold Start).

## URL

https://github.com/JarvisUSTC/DoctorAgent-RL
