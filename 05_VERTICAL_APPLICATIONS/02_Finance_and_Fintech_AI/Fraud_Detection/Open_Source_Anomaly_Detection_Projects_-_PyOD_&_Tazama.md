# Open Source Anomaly Detection Projects - PyOD & Tazama

## Description

PyOD é uma biblioteca Python abrangente e escalável para detecção de objetos anômalos/outliers em dados multivariados. Fundada em 2017, ela se tornou uma ferramenta essencial, oferecendo mais de 50 algoritmos de detecção, desde técnicas clássicas (como LOF) até métodos de ponta de Deep Learning baseados em PyTorch. A versão 2.0 introduziu suporte a Deep Learning expandido e seleção de modelo assistida por LLM (Large Language Model). Tazama é uma plataforma de software de código aberto e gratuita dedicada à detecção de fraudes e lavagem de dinheiro em tempo real, antes que ocorram. É um projeto da Linux Foundation, financiado pela Gates Foundation, focado em fornecer uma solução escalável e econômica para monitoramento de transações, especialmente em mercados emergentes, promovendo a inclusão financeira.

## Statistics

PyOD: Mais de 26 milhões de downloads desde 2017. Mais de 9.6k estrelas e 1.5k forks no GitHub. Mais de 50 algoritmos de detecção de outliers implementados. Compatível com Python 3.8+. Tazama: Projeto de código aberto da Linux Foundation, financiado pela Gates Foundation. Focado em mercados emergentes. Oferece uma alternativa de baixo custo para monitoramento de transações em tempo real.

## Features

PyOD: Interface unificada e amigável; Ampla gama de modelos (clássicos e Deep Learning); Alto desempenho e eficiência (utiliza Numba e Joblib); Treinamento e previsão rápidos via framework SUOD; API completa com métodos como `fit(X)`, `decision_function(X)`, `predict(X)` e `predict_proba(X)`. Tazama: Monitoramento de transações em tempo real; Redução de fraudes e golpes; Suporte a atividades Anti-Lavagem de Dinheiro (AML); Permite a implementação de regras simples ou complexas; Projetado para ser escalável e econômico.

## Use Cases

PyOD: Detecção de fraude (fraud-detection); Detecção de anomalias em dados multivariados; Detecção de outliers em séries temporais (via TODS); Pesquisa acadêmica e produtos comerciais; Detecção de anomalias em Processamento de Linguagem Natural (NLP) (via NLP-ADBench). Tazama: Prevenção de fraudes em sistemas de pagamento; Monitoramento de transações financeiras em tempo real; Combate à lavagem de dinheiro; Melhoria da segurança e conformidade de transações; Promoção da inclusão financeira em mercados emergentes.

## Integration

PyOD: Instalação via pip (`pip install pyod`) ou conda. Integração com o ecossistema Python de Machine Learning (scikit-learn, PyTorch). Exemplo de uso com 5 linhas de código:\n```python\nfrom pyod.models.ecod import ECOD\nclf = ECOD()\nclf.fit(X_train)\ny_train_scores = clf.decision_scores_\ny_test_scores = clf.decision_function(X_test)\n```\nTazama: Software de código aberto. Projetado para ser implementado por organizações para monitorar transações financeiras. O código e a documentação estão disponíveis no GitHub.

## URL

PyOD: https://github.com/yzhao062/pyod | Tazama: https://www.tazama.org/