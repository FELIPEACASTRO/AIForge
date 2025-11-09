# AgriSync: AI-Powered Smart Farming & Marketplace System

## Description

O **AgriSync** é uma plataforma de agricultura inteligente de código aberto que integra Inteligência Artificial (IA), Machine Learning (ML), Deep Learning e tecnologia Blockchain para fornecer aos agricultores insights em tempo real, automatizar o monitoramento da saúde das colheitas e garantir transações transparentes. O projeto se destaca por ser uma solução full-stack que aborda múltiplos desafios da agricultura moderna, desde a detecção precoce de doenças até a comercialização segura de produtos. Sua proposta de valor única reside na combinação de modelos preditivos avançados com a rastreabilidade e segurança da Blockchain, capacitando os agricultores a tomar decisões baseadas em dados e a reduzir perdas.

## Statistics

**Tecnologias Chave:** Python (41.6%), JavaScript (56.0%), Solidity (1.6%). **Modelos de IA/ML:** EfficientNetB4 (Deep Learning para detecção de doenças), Random Forest e XGBoost (Previsão de Preços), Modelos de Séries Temporais (Previsão do Tempo). **Frameworks:** FastAPI, React.js, TensorFlow, Keras, Scikit-learn, Web3.js, Solidity. **Status do Repositório:** 7 estrelas, 6 forks (em 14 de março de 2025). **Desenvolvimento:** Sistema full-stack com 100 commits (em 4 meses atrás).

## Features

**Detecção de Doenças em Plantas:** Utiliza um modelo de Deep Learning (EfficientNetB4) para classificar doenças foliares a partir de imagens. **Previsão de Preços de Colheitas:** Emprega modelos de Machine Learning (Random Forest, XGBoost) treinados em dados históricos para prever preços de mercado. **Previsão do Tempo e Alertas:** Usa modelos de ML de séries temporais para prever padrões climáticos futuros. **Previsão da Qualidade do Solo:** Classifica o tipo de solo com base em valores de NPK e outros parâmetros. **Marketplace Blockchain:** Um contrato inteligente (Solidity) permite transações seguras e transparentes para a venda de colheitas, com rastreabilidade garantida pelo Ethereum testnet. **Interface de Usuário (UI) Moderna:** Frontend construído com React e Tailwind CSS, permitindo que os agricultores visualizem alertas, previsões e listem itens para venda.

## Use Cases

**Monitoramento de Saúde da Colheita:** Agricultores podem tirar fotos de folhas de plantas para obter um diagnóstico instantâneo de doenças, permitindo intervenção precoce e redução de perdas. **Otimização de Recursos:** A previsão da qualidade do solo e do tempo ajuda a otimizar o uso de fertilizantes, água e pesticidas. **Planejamento Financeiro:** A previsão de preços de colheitas permite que os agricultores planejem a venda de seus produtos no momento mais lucrativo. **Rastreabilidade e Confiança:** O Marketplace Blockchain permite que os consumidores rastreiem a origem dos produtos e garante transações justas e transparentes entre agricultores e compradores. **Desenvolvimento Comunitário:** O código aberto serve como base para que desenvolvedores e pesquisadores criem novas funcionalidades ou adaptem os modelos para diferentes regiões e culturas.

## Integration

A integração com o AgriSync é feita através de sua arquitetura full-stack, que utiliza **FastAPI** no backend para servir os modelos de ML/IA e **React.js** no frontend para a interface do usuário. A parte de Blockchain é integrada via **Web3.js** para interagir com o contrato inteligente (Solidity).

**Exemplo de Integração (Python - Backend FastAPI):**
Para interagir com o modelo de detecção de doenças, o backend expõe um endpoint. O código Python para o modelo de previsão de doenças (`predict_plantdoc.py`) utiliza TensorFlow/Keras.

```python
# Exemplo de código no backend (FastAPI)
from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = FastAPI()
model = load_model('path/to/efficientnetb4_model.h5') # Carrega o modelo

@app.post("/predict/disease")
async def predict_disease(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    
    # Mapear predicted_class para o nome da doença
    disease_name = "Doença X" # Lógica de mapeamento
    
    return {"prediction": disease_name, "confidence": float(predictions[0][predicted_class])}

# Para rodar: uvicorn main:app --reload
```

**Exemplo de Integração (Blockchain - Web3.js no Frontend):**
O frontend React usa Web3.js para interagir com o contrato inteligente de venda de colheitas.

```javascript
// Exemplo de código no frontend (React/Web3.js)
import Web3 from 'web3';
import CropMarketplace from './contracts/CropMarketplace.json'; // ABI do contrato

const web3 = new Web3(Web3.givenProvider || "http://localhost:7545");
const networkId = await web3.eth.net.getId();
const deployedNetwork = CropMarketplace.networks[networkId];
const contract = new web3.eth.Contract(
    CropMarketplace.abi,
    deployedNetwork && deployedNetwork.address,
);

// Função para listar uma colheita para venda
async function listCropForSale(price, quantity) {
    const accounts = await web3.eth.getAccounts();
    await contract.methods.listCrop(price, quantity).send({ from: accounts[0] });
    console.log("Colheita listada com sucesso!");
}
```

## URL

https://github.com/bunnysunny24/AgriSync