# AgriSync: AI-Powered Smart Farming & Marketplace System

## Description

**AgriSync** is an open-source smart farming platform that integrates Artificial Intelligence (AI), Machine Learning (ML), Deep Learning, and Blockchain technology to provide farmers with real-time insights, automate crop health monitoring, and ensure transparent transactions. The project stands out as a full-stack solution that addresses multiple challenges of modern agriculture, from early disease detection to secure product commercialization. Its unique value proposition lies in combining advanced predictive models with the traceability and security of Blockchain, empowering farmers to make data-driven decisions and reduce losses.

## Statistics

**Key Technologies:** Python (41.6%), JavaScript (56.0%), Solidity (1.6%). **AI/ML Models:** EfficientNetB4 (Deep Learning for disease detection), Random Forest and XGBoost (Price Prediction), Time Series Models (Weather Forecasting). **Frameworks:** FastAPI, React.js, TensorFlow, Keras, Scikit-learn, Web3.js, Solidity. **Repository Status:** 7 stars, 6 forks (as of March 14, 2025). **Development:** Full-stack system with 100 commits (as of 4 months ago).

## Features

**Plant Disease Detection:** Uses a Deep Learning model (EfficientNetB4) to classify leaf diseases from images. **Crop Price Prediction:** Employs Machine Learning models (Random Forest, XGBoost) trained on historical data to forecast market prices. **Weather Forecasting and Alerts:** Uses time-series ML models to predict future weather patterns. **Soil Quality Prediction:** Classifies soil type based on NPK values and other parameters. **Blockchain Marketplace:** A smart contract (Solidity) enables secure and transparent transactions for selling crops, with traceability guaranteed by the Ethereum testnet. **Modern User Interface (UI):** Frontend built with React and Tailwind CSS, allowing farmers to view alerts and forecasts and list items for sale.

## Use Cases

**Crop Health Monitoring:** Farmers can take photos of plant leaves to get an instant disease diagnosis, enabling early intervention and reducing losses. **Resource Optimization:** Soil quality and weather forecasting help optimize the use of fertilizers, water, and pesticides. **Financial Planning:** Crop price forecasting allows farmers to plan the sale of their products at the most profitable time. **Traceability and Trust:** The Blockchain Marketplace allows consumers to trace the origin of products and ensures fair, transparent transactions between farmers and buyers. **Community Development:** The open-source codebase serves as a foundation for developers and researchers to build new features or adapt the models to different regions and crops.

## Integration

Integration with AgriSync is achieved through its full-stack architecture, which uses **FastAPI** on the backend to serve the ML/AI models and **React.js** on the frontend for the user interface. The Blockchain component is integrated via **Web3.js** to interact with the smart contract (Solidity).

**Integration Example (Python - FastAPI Backend):**
To interact with the disease detection model, the backend exposes an endpoint. The Python code for the disease prediction model (`predict_plantdoc.py`) uses TensorFlow/Keras.

```python
# Example backend code (FastAPI)
from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = FastAPI()
model = load_model('path/to/efficientnetb4_model.h5') # Load the model

@app.post("/predict/disease")
async def predict_disease(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    
    # Map predicted_class to the disease name
    disease_name = "Disease X" # Mapping logic
    
    return {"prediction": disease_name, "confidence": float(predictions[0][predicted_class])}

# To run: uvicorn main:app --reload
```

**Integration Example (Blockchain - Web3.js on the Frontend):**
The React frontend uses Web3.js to interact with the crop sales smart contract.

```javascript
// Example frontend code (React/Web3.js)
import Web3 from 'web3';
import CropMarketplace from './contracts/CropMarketplace.json'; // Contract ABI

const web3 = new Web3(Web3.givenProvider || "http://localhost:7545");
const networkId = await web3.eth.net.getId();
const deployedNetwork = CropMarketplace.networks[networkId];
const contract = new web3.eth.Contract(
    CropMarketplace.abi,
    deployedNetwork && deployedNetwork.address,
);

// Function to list a crop for sale
async function listCropForSale(price, quantity) {
    const accounts = await web3.eth.getAccounts();
    await contract.methods.listCrop(price, quantity).send({ from: accounts[0] });
    console.log("Crop listed successfully!");
}
```

## URL

https://github.com/bunnysunny24/AgriSync