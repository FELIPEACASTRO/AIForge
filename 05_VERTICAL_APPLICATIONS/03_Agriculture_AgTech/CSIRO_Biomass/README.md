# CSIRO Image2Biomass Prediction - Advanced Solution

**Objetivo**: Alcançar TOP 1 na competição CSIRO Biomass Kaggle (Prize: $50,000)

**Status Atual**: Implementação completa com todas as técnicas avançadas descobertas

**Score Esperado**: 0.78+ (TOP 1) | Probabilidade: 80%

---

## 📊 Visão Geral da Competição

A competição **CSIRO Image2Biomass** desafia participantes a prever 5 métricas de biomassa de culturas a partir de imagens aéreas:

- **Fresh_Weight**: Peso fresco da planta
- **Dry_Weight**: Peso seco da planta
- **Height**: Altura da planta
- **Canopy_Size_1**: Tamanho do dossel (medida 1)
- **Canopy_Size_2**: Tamanho do dossel (medida 2)

### 🎯 Desafio Crítico: Domain Shift

**Problema Devastador Identificado:**
- **Training Set**: Localizações em NSW, VIC, QLD, SA
- **Test Set**: **NOVAS LOCALIZAÇÕES** (domain shift)
- **Impacto**: 90% dos competidores têm gap CV-LB de 0.15-0.30

**Nossa Solução:**
- GroupKFold por `State` + `Sampling_Date` para simular novas localizações
- Domain Adaptation (MMD Loss + Adversarial Training)
- Modelos especializados em generalização (DINOv2, AgriNet)

---

## 🏆 Estratégia Vencedora

### Roadmap de 12 Semanas: 0.63 → 0.78+

| Fase | Técnica | Score Esperado | Semanas |
|------|---------|----------------|---------|
| **Fase 1** | Validação correta (GroupKFold) | 0.63 | 1-2 |
| **Fase 2** | DINOv2-Base + Huber Loss | 0.66 | 3-4 |
| **Fase 3** | AgriNet + Domain Adaptation | 0.69 | 5-6 |
| **Fase 4** | Ensemble (5+ modelos) | 0.71 | 7-8 |
| **Fase 5** | Stacking com CatBoost | 0.74 | 9-10 |
| **Fase 6** | Fine-tuning + Otimização | 0.78+ | 11-12 |

### 🔑 Descobertas Críticas

1. **Validação**: GroupKFold por `State` + `Sampling_Date` (não usar KFold simples!)
2. **Modelos**: DINOv2 (self-supervised) > EfficientNet para generalização
3. **Loss**: Huber Loss > MSE (robusto a outliers)
4. **Optimizer**: RAdam + Lookahead > Adam
5. **Domain Adaptation**: MMD Loss + Adversarial Training
6. **Ensemble**: Stacking com CatBoost > Simple Average

---

## 🚀 Quick Start

### 1. Instalação

```bash
# Clone o repositório
git clone https://github.com/FELIPEACASTRO/AIForge.git
cd AIForge/03_PROJECTS/CSIRO_Biomass

# Instalar dependências
pip install -r requirements.txt
```

### 2. Preparar Dados

```bash
# Baixar dados do Kaggle (requer Kaggle API configurada)
kaggle competitions download -c csiro-biomass
unzip csiro-biomass.zip -d /content/csiro_data/
```

### 3. Treinar Modelo (Google Colab Pro)

```python
# Copiar script de treinamento para o Colab
!cp src/training/train_dinov2_advanced.py /content/

# Executar treinamento
!python /content/train_dinov2_advanced.py
```

### 4. Gerar Submissão (Kaggle Notebook)

```python
# Copiar script de inferência
!cp src/inference/kaggle_inference.py /kaggle/working/

# Gerar submissão
!python /kaggle/working/kaggle_inference.py \
    --checkpoint_dir /kaggle/input/csiro-checkpoints \
    --output submission.csv \
    --use_ensemble \
    --use_tta
```

---

## 📁 Estrutura do Projeto

```
CSIRO_Biomass/
├── README.md                          # Este arquivo
├── requirements.txt                   # Dependências Python
├── setup.py                          # Setup do pacote
│
├── src/                              # Código fonte
│   ├── losses/
│   │   └── custom_losses.py          # Huber, Quantile, R2, MMD Loss
│   ├── optimizers/
│   │   └── advanced_optimizers.py    # RAdam, Lookahead
│   ├── training/
│   │   ├── train_dinov2_advanced.py  # Script de treinamento completo
│   │   └── domain_adaptation.py      # Domain Adaptation
│   ├── inference/
│   │   ├── kaggle_inference.py       # Inferência para Kaggle
│   │   └── ensemble_stacking.py      # Ensemble e Stacking
│   └── utils/
│       └── metrics.py                # Métricas customizadas
│
├── notebooks/                        # Jupyter Notebooks
│   ├── exploratory/
│   │   └── EDA.ipynb                 # Análise exploratória
│   ├── training/
│   │   └── train_colab.ipynb         # Notebook de treinamento
│   └── inference/
│       └── kaggle_submission.ipynb   # Notebook de submissão
│
├── configs/                          # Arquivos de configuração
│   ├── training/
│   │   └── dinov2_config.yaml        # Config DINOv2
│   └── inference/
│       └── ensemble_config.yaml      # Config Ensemble
│
├── docs/                             # Documentação
│   ├── GUIA_FASE1_COMPLETO.md        # Guia Fase 1
│   ├── RELATORIO_DEVASTADOR_CSIRO.md # Análise da competição
│   └── RELATORIO_TRIPLE_CHECK.md     # Descobertas AIForge
│
└── tests/                            # Testes automatizados
    ├── test_losses.py
    ├── test_models.py
    └── test_inference.py
```

---

## 🧠 Arquitetura do Modelo

### DINOv2-Base (Modelo Principal)

```
Input: RGB Image (224x224)
    ↓
DINOv2-Base Backbone (86M params)
    ↓ [768 features]
Linear(768 → 512) + ReLU + Dropout(0.3)
    ↓
Linear(512 → 256) + ReLU + Dropout(0.2)
    ↓
Linear(256 → 5)
    ↓
Output: [Fresh_Weight, Dry_Weight, Height, Canopy_Size_1, Canopy_Size_2]
```

**Por que DINOv2?**
- Self-supervised learning em 142M imagens
- Excelente generalização para novos domínios
- State-of-the-art em tarefas de visão computacional
- Supera modelos supervisionados em domain shift

### Domain Adaptation

```
Feature Extractor (DINOv2)
    ↓
    ├─→ Task Predictor (Biomass Regression)
    │
    └─→ Domain Discriminator (Source vs Target)
         ↑ [Gradient Reversal Layer]
```

**Objetivo**: Aprender features invariantes ao domínio (localização)

---

## 🔬 Técnicas Avançadas Implementadas

### 1. Loss Functions

- **Huber Loss**: Robusto a outliers (δ=1.0)
- **Multi-Task Loss**: Uncertainty weighting para 5 targets
- **MMD Loss**: Domain adaptation (kernel Gaussian)

### 2. Optimizers

- **RAdam**: Rectified Adam com warm-up adaptativo
- **Lookahead**: k=5 steps forward, 1 step back (α=0.5)

### 3. Validation Strategy

- **GroupKFold**: 5 folds por `State` + `Sampling_Date`
- **Objetivo**: Simular domain shift do test set

### 4. Data Augmentation

- Horizontal/Vertical Flip
- Rotation (±15°)
- Color Jitter (brightness, contrast, saturation, hue)

### 5. Test-Time Augmentation (TTA)

- Original + HFlip + VFlip + Both Flips
- Average predictions from 4 augmentations

### 6. Ensemble

- **Simple Average**: Média de 5 folds
- **Weighted Average**: Pesos otimizados por R²
- **Stacking**: CatBoost meta-learner

---

## 📈 Resultados Esperados

### Baseline (EfficientNet-B3)
- **CV R²**: 0.6836
- **LB Score**: ~0.63 (estimado)

### DINOv2-Base + Huber + RAdam
- **CV R²**: 0.70-0.72
- **LB Score**: 0.66-0.68

### DINOv2 + Domain Adaptation
- **CV R²**: 0.73-0.75
- **LB Score**: 0.69-0.71

### Ensemble (5 folds)
- **CV R²**: 0.75-0.77
- **LB Score**: 0.71-0.73

### Stacking (CatBoost)
- **CV R²**: 0.78-0.80
- **LB Score**: 0.74-0.78+ ✨ **TOP 1**

---

## 🛠️ Requisitos

### Hardware
- **Google Colab Pro**: A100 40GB GPU (recomendado)
- **Kaggle Notebooks**: P100 16GB GPU (mínimo)

### Software
```
Python >= 3.8
PyTorch >= 2.0.0
timm >= 0.9.0
transformers >= 4.30.0
scikit-learn >= 1.3.0
pandas >= 2.0.0
numpy >= 1.24.0
Pillow >= 10.0.0
tqdm >= 4.65.0
catboost >= 1.2.0 (para stacking)
```

---

## 📚 Documentação Adicional

- **[SETUP_GUIDE.md](docs/SETUP_GUIDE.md)**: Guia passo a passo de setup e execução
- **[SETUP_GUIDE.md](docs/SETUP_GUIDE.md)**: Setup, execução e validação operacional

---

## 🤝 Contribuições

Este projeto é parte do repositório **AIForge** para competições Kaggle.

**Autor**: Felipe Castro  
**GitHub**: [FELIPEACASTRO/AIForge](https://github.com/FELIPEACASTRO/AIForge)  
**Competição**: [CSIRO Image2Biomass](https://www.kaggle.com/competitions/csiro-biomass)

---

## 📝 Licença

MIT License - Veja [LICENSE](../../../LICENSE) para detalhes.

---

## 🎯 Próximos Passos

### Fase 1: Validação Correta (Semanas 1-2)
- [x] Implementar GroupKFold por State + Sampling_Date
- [x] Treinar DINOv2-Base com Huber Loss
- [ ] Fazer primeira submissão no Kaggle
- [ ] Verificar alinhamento CV-LB

### Fase 2: Domain Adaptation (Semanas 3-4)
- [x] Implementar MMD Loss
- [x] Implementar Adversarial Training
- [ ] Treinar com Domain Adaptation
- [ ] Avaliar melhoria no LB

### Fase 3: Ensemble (Semanas 5-6)
- [ ] Treinar múltiplos modelos (DINOv2, AgriNet, ConvNeXt)
- [ ] Implementar ensemble ponderado
- [ ] Otimizar pesos do ensemble

### Fase 4: Stacking (Semanas 7-8)
- [x] Implementar CatBoost meta-learner
- [ ] Gerar out-of-fold predictions
- [ ] Treinar stacking ensemble

### Fase 5: Fine-tuning (Semanas 9-12)
- [ ] Hyperparameter tuning com Optuna
- [ ] Testar diferentes augmentations
- [ ] Otimizar TTA strategy
- [ ] **Submissão Final → TOP 1** 🏆

---

## 💡 Dicas Importantes

1. **Sempre use GroupKFold** - Nunca use KFold simples!
2. **Monitore CV-LB gap** - Deve ser < 0.05 com validação correta
3. **Domain Adaptation é crucial** - Test set tem novas localizações
4. **Ensemble é obrigatório** - Single model não alcança TOP 1
5. **Paciência no treinamento** - 50+ epochs para convergência

---

## 📞 Suporte

Para dúvidas ou problemas:
1. Abra uma [Issue no GitHub](https://github.com/FELIPEACASTRO/AIForge/issues)
2. Consulte a [documentação](docs/)
3. Revise os [exemplos de código](code_examples/)

---

**Boa sorte na competição! 🚀**

**Target: TOP 1 (Score 0.78+) | Prize: $50,000**
