# Guia de Setup - CSIRO Biomass

Este guia fornece instruções detalhadas para configurar o ambiente e começar a treinar modelos para a competição CSIRO Biomass.

---

## 📋 Pré-requisitos

### 1. Conta Kaggle
- Criar conta em [kaggle.com](https://www.kaggle.com)
- Aceitar as regras da competição: [CSIRO Image2Biomass](https://www.kaggle.com/competitions/csiro-biomass)

### 2. Google Colab Pro (Recomendado)
- Assinar [Google Colab Pro](https://colab.research.google.com/signup) ($9.99/mês)
- Acesso a GPU A100 40GB (essencial para treinar DINOv2)

### 3. Google Drive
- Mínimo 10GB de espaço livre para checkpoints

---

## 🚀 Instalação

### Opção 1: Google Colab (Recomendado para Treinamento)

#### Passo 1: Configurar Google Drive

```python
# Montar Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Criar diretório para checkpoints
!mkdir -p /content/drive/MyDrive/csiro_checkpoints_advanced
```

#### Passo 2: Clonar Repositório

```python
# Clonar AIForge
!git clone https://github.com/FELIPEACASTRO/AIForge.git
%cd AIForge/03_PROJECTS/CSIRO_Biomass
```

#### Passo 3: Instalar Dependências

```python
# Instalar requirements
!pip install -q -r requirements.txt

# Verificar instalação
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

#### Passo 4: Baixar Dados do Kaggle

```python
# Configurar Kaggle API
!mkdir -p ~/.kaggle
!echo '{"username":"SEU_USERNAME","key":"SUA_API_KEY"}' > ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json

# Baixar dados da competição
!kaggle competitions download -c csiro-biomass
!unzip -q csiro-biomass.zip -d /content/csiro_data/

# Verificar estrutura
!ls -lh /content/csiro_data/
```

**Como obter Kaggle API Key:**
1. Ir para [kaggle.com/account](https://www.kaggle.com/account)
2. Clicar em "Create New API Token"
3. Baixar `kaggle.json`
4. Copiar username e key para o comando acima

#### Passo 5: Treinar Modelo

```python
# Copiar script de treinamento
!cp src/training/train_dinov2_advanced.py /content/

# Executar treinamento (5 folds, ~6-8 horas no A100)
!python /content/train_dinov2_advanced.py
```

---

### Opção 2: Kaggle Notebooks (Para Inferência)

#### Passo 1: Criar Novo Notebook

1. Ir para [kaggle.com/code](https://www.kaggle.com/code)
2. Clicar em "New Notebook"
3. Selecionar "GPU P100" ou "GPU T4 x2"

#### Passo 2: Adicionar Datasets

1. Clicar em "Add data" → "Competition Data"
2. Selecionar "CSIRO Image2Biomass"
3. Clicar em "Add data" → "Your Datasets"
4. Upload dos checkpoints treinados

#### Passo 3: Clonar Código

```python
# Clonar repositório
!git clone https://github.com/FELIPEACASTRO/AIForge.git
import sys
sys.path.append('/kaggle/working/AIForge/03_PROJECTS/CSIRO_Biomass')
```

#### Passo 4: Gerar Submissão

```python
# Copiar script de inferência
!cp /kaggle/working/AIForge/03_PROJECTS/CSIRO_Biomass/src/inference/kaggle_inference.py .

# Gerar submissão
!python kaggle_inference.py \
    --test_csv /kaggle/input/csiro-biomass/test.csv \
    --test_images_dir /kaggle/input/csiro-biomass/test_images \
    --checkpoint_dir /kaggle/input/csiro-checkpoints \
    --output submission.csv \
    --use_ensemble \
    --use_tta

# Submeter
!kaggle competitions submit -c csiro-biomass -f submission.csv -m "DINOv2 + Ensemble + TTA"
```

---

### Opção 3: Local (Linux/Mac)

#### Passo 1: Clonar Repositório

```bash
git clone https://github.com/FELIPEACASTRO/AIForge.git
cd AIForge/03_PROJECTS/CSIRO_Biomass
```

#### Passo 2: Criar Ambiente Virtual

```bash
# Criar venv
python3 -m venv venv

# Ativar venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

#### Passo 3: Instalar Dependências

```bash
# Upgrade pip
pip install --upgrade pip

# Instalar requirements
pip install -r requirements.txt
```

#### Passo 4: Verificar GPU

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 📊 Estrutura de Dados Esperada

Após baixar os dados, a estrutura deve ser:

```
/content/csiro_data/
├── train.csv                    # Metadados de treino
├── test.csv                     # Metadados de teste
├── train_images/                # Imagens de treino
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
└── test_images/                 # Imagens de teste
    ├── test_001.jpg
    ├── test_002.jpg
    └── ...
```

**Verificar:**
```python
import pandas as pd

# Carregar CSVs
train_df = pd.read_csv('/content/csiro_data/train.csv')
test_df = pd.read_csv('/content/csiro_data/test.csv')

print(f"Train samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
print(f"\nTrain columns: {list(train_df.columns)}")
print(f"\nTarget columns:")
print(train_df[['Fresh_Weight', 'Dry_Weight', 'Height', 'Canopy_Size_1', 'Canopy_Size_2']].describe())
```

---

## 🔧 Configuração Avançada

### Ajustar Hiperparâmetros

Editar `src/training/train_dinov2_advanced.py`:

```python
class Config:
    # Ajustar batch size de acordo com GPU
    BATCH_SIZE = 32  # A100: 32-64, P100: 16-32, T4: 8-16
    
    # Ajustar learning rate
    LEARNING_RATE = 1e-4  # Padrão: 1e-4, mais rápido: 3e-4, mais estável: 5e-5
    
    # Ajustar número de epochs
    NUM_EPOCHS = 50  # Padrão: 50, rápido: 30, completo: 100
    
    # Ativar/desativar técnicas
    USE_MULTITASK = True
    USE_LOOKAHEAD = True
    USE_AMP = True  # Mixed precision (mais rápido)
```

### Monitorar Treinamento

```python
# Durante o treinamento, monitorar:
# - Train Loss: deve diminuir consistentemente
# - Val Loss: deve diminuir sem overfitting
# - Val R2: deve aumentar (target: > 0.70)
# - Val MAE: deve diminuir
# - Val RMSE: deve diminuir

# Sinais de problemas:
# - Train Loss >> Val Loss: underfitting (aumentar capacidade do modelo)
# - Train Loss << Val Loss: overfitting (aumentar regularização)
# - R2 estagnado: ajustar learning rate ou loss function
```

---

## 🧪 Testar Instalação

### Teste 1: Imports

```python
# Testar imports essenciais
import torch
import torchvision
import timm
import pandas as pd
import numpy as np
from PIL import Image

print("✓ Todos os imports funcionaram!")
```

### Teste 2: Carregar DINOv2

```python
# Testar carregamento do DINOv2
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_base', pretrained=True)
print(f"✓ DINOv2 carregado: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
```

### Teste 3: GPU

```python
# Testar GPU
if torch.cuda.is_available():
    x = torch.randn(1, 3, 224, 224).cuda()
    model = model.cuda()
    with torch.no_grad():
        y = model(x)
    print(f"✓ GPU funcionando: {y.shape}")
else:
    print("⚠ GPU não disponível - treinamento será lento!")
```

### Teste 4: Carregar Dados

```python
# Testar carregamento de dados
from src.training.train_dinov2_advanced import CSIRODataset, get_transforms

train_df = pd.read_csv('/content/csiro_data/train.csv')
dataset = CSIRODataset(
    train_df.head(10),
    '/content/csiro_data/train_images',
    transform=get_transforms(is_train=True)
)

image, targets = dataset[0]
print(f"✓ Dataset funcionando: image {image.shape}, targets {targets.shape}")
```

---

## 🐛 Troubleshooting

### Problema 1: Out of Memory (OOM)

**Sintomas**: `RuntimeError: CUDA out of memory`

**Soluções**:
```python
# 1. Reduzir batch size
Config.BATCH_SIZE = 16  # ou 8

# 2. Reduzir tamanho da imagem
Config.IMG_SIZE = 192  # ao invés de 224

# 3. Usar gradient accumulation
# Adicionar no loop de treinamento:
accumulation_steps = 2
loss = loss / accumulation_steps
loss.backward()
if (step + 1) % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

### Problema 2: Treinamento Muito Lento

**Soluções**:
```python
# 1. Verificar GPU está sendo usada
print(torch.cuda.is_available())

# 2. Aumentar num_workers
Config.NUM_WORKERS = 4  # ou 8

# 3. Ativar mixed precision
Config.USE_AMP = True

# 4. Usar pin_memory
DataLoader(..., pin_memory=True)
```

### Problema 3: Kaggle API Error

**Sintomas**: `403 Forbidden` ou `401 Unauthorized`

**Soluções**:
```bash
# 1. Verificar kaggle.json
cat ~/.kaggle/kaggle.json

# 2. Verificar permissões
chmod 600 ~/.kaggle/kaggle.json

# 3. Aceitar regras da competição
# Ir para: https://www.kaggle.com/competitions/csiro-biomass/rules
# Clicar em "I Understand and Accept"
```

### Problema 4: DINOv2 Não Carrega

**Sintomas**: `RuntimeError: Error loading model from torch.hub`

**Soluções**:
```python
# 1. Limpar cache do torch.hub
import torch
torch.hub.set_dir('/tmp/torch_hub')

# 2. Forçar reload
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_base', 
                       pretrained=True, force_reload=True)

# 3. Baixar manualmente
!git clone https://github.com/facebookresearch/dinov2.git
# Importar localmente
```

---

## 📞 Suporte

Se encontrar problemas:

1. **Verificar logs**: Sempre salvar e revisar logs de erro completos
2. **GitHub Issues**: [AIForge Issues](https://github.com/FELIPEACASTRO/AIForge/issues)
3. **Kaggle Discussion**: [Competition Discussion](https://www.kaggle.com/competitions/csiro-biomass/discussion)

---

## ✅ Checklist de Setup

- [ ] Conta Kaggle criada e regras aceitas
- [ ] Google Colab Pro assinado (ou GPU local disponível)
- [ ] Google Drive com 10GB+ livre
- [ ] Repositório AIForge clonado
- [ ] Dependências instaladas (`requirements.txt`)
- [ ] Kaggle API configurada (`kaggle.json`)
- [ ] Dados baixados e extraídos
- [ ] GPU funcionando (teste com `torch.cuda.is_available()`)
- [ ] DINOv2 carrega corretamente
- [ ] Dataset carrega imagens sem erros
- [ ] Pronto para treinar! 🚀

---

**Próximo passo**: revise os exemplos em [code_examples](../code_examples/).
