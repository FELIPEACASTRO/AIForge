# Data.gov - US Government Open Data

## Description

Data.gov is the United States Government's open data portal, launched in 2009 and managed by the U.S. General Services Administration (GSA). Its mission is to unleash the power of government open data to **inform decisions by the public and policymakers, drive innovation and economic activity, achieve agency missions, and strengthen the foundation of an open and transparent government**. It serves as the central catalog for federal, state, and local data, making it a fundamental resource for developing Artificial Intelligence and Machine Learning (AI/ML) models based on verified government information.

## Statistics

*   **Total Datasets:** Over **364,000** datasets (2025 data).
*   **Monthly Pageviews:** Approximately **1.95 million** pageviews (October 2025 data).
*   **Top Contributors:** U.S. Census Bureau (over 111k datasets), NOAA (over 102k datasets).
*   **Technology:** Powered by two open-source applications: CKAN and 11ty.

## Features

*   **Centralized Catalog:** Aggregates data from over 170 federal, state, and local agencies.
*   **CKAN Foundation:** Built on the open-source CKAN (Comprehensive Knowledge Archive Network) platform for data management and API functionality.
*   **Machine-Readable Data:** Focus on making data available in open, machine-readable formats, as required by the OPEN Government Data Act.
*   **Search and Filtering:** Robust tools to search and filter datasets by topic, format, agency, and tags (including "artificial-intelligence" and "machine-learning").
*   **Usage Metrics:** Metrics dashboard providing insights into the most viewed and downloaded datasets.

## Use Cases

*   **AI/ML Model Training:** Utilizing large volumes of government data (e.g., NOAA climate data, Census Bureau demographic data) to train Machine Learning models for forecasting and classification tasks.
*   **Public Policy Analysis:** AI models to analyze the impact of health (using HHS data) or environmental policies (using EPA data), enabling evidence-based decision-making.
*   **Anomaly and Fraud Detection:** Applying ML to government financial and contract data to identify unusual patterns and prevent fraud.
*   **Civic Application Development:** Creating mobile and web applications that use open data (e.g., traffic data, public safety, air quality) to improve citizens' lives.
*   **Scientific Research:** Using high-quality datasets (e.g., NASA and Department of Energy data) for academic research and new technology development.

## Integration

Integration is primarily done through the **CKAN API** (Comprehensive Knowledge Archive Network), which is the open-source platform powering Data.gov. The API allows for programmatic searching, metadata retrieval, and direct access to data resources.

**Code Example (Python for Dataset Search):**
```python
import requests
import json

# Base URL for the Data.gov CKAN API
CKAN_API_URL = "https://catalog.data.gov/api/3/action/"

def search_datasets(query):
    """
    Searches for datasets on Data.gov using the CKAN API.
    """
    action = "package_search"
    # Limits to 5 results for the example
    params = {"q": query, "rows": 5} 
    
    try:
        response = requests.get(CKAN_API_URL + action, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if data["success"]:
            print(f"Found {data['result']['count']} datasets for '{query}':\n")
            for pkg in data["result"]["results"]:
                print(f"  Title: {pkg['title']}")
                print(f"  Organization: {pkg['organization']['title'] if pkg.get('organization') else 'N/A'}")
                print(f"  URL: https://catalog.data.gov/dataset/{pkg['name']}")
                print("-" * 20)
        else:
            print("API search failed:", data.get("error"))
            
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to API: {e}")

if __name__ == "__main__":
    # Example search for datasets related to "artificial intelligence"
    search_datasets("artificial intelligence")
```

## URL

https://data.gov/
