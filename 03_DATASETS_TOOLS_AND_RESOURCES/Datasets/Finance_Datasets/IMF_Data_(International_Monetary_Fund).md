# IMF Data (International Monetary Fund)

## Description
**IMF Data** (International Monetary Fund) is a vast collection of global macroeconomic and financial data. The main dataset, **International Financial Statistics (IFS)**, has been restructured and its indicators reorganized into more than 20 smaller thematic datasets (such as CPI, BOP, EER, MFS, etc.) on the IMF Data Portal. This data is crucial for the analysis of the global economy, balance of payments, government finance, and monetary statistics. The restructuring aims to provide a more consistent and granular data model.

## Statistics
**Coverage:** Macroeconomic and financial data for more than 180 countries.
**Frequency:** Ranges from Annual, Quarterly, and Monthly to Daily, depending on the thematic dataset.
**Versions:** The data is continuously updated. The IFS data structure was restructured into more than 20 thematic datasets, with the most recent access notice dated March 2025. There is no single total file size, as it is a dynamic data portal.

## Features
IMF Data is composed of more than 20 thematic datasets that cover indicators from more than 180 countries. The main topics include: Labor Statistics (LS), Consumer Price Index (CPI), Balance of Payments (BOP), International Investment Position (IIP), Monetary and Financial Statistics (MFS), and National Economic Accounts (NEA). The data is accessible via web portal and API, with support for the SDMX 2.1 and 3.0 formats.

## Use Cases
Macroeconomic modeling and forecasting. Analysis of financial stability and balance of payments. Academic and policy research on global economic indicators (GDP, inflation, exchange rates, etc.). Development of AI/ML models to forecast economic and financial trends and analyze sovereign risk.

## Integration
Access to IMF data can be done in several ways:
1.  **Data Portal:** Interactive access and download via web interface at [https://data.imf.org/](https://data.imf.org/).
2.  **API (SDMX 2.1 and 3.0):** Programmatic access via the SDMX API. The IMF recommends the Python library `sdmx1` for queries. To access data that was part of the original IFS, you must use the filter `c[IFS_Flag]=True` in the API query.
3.  **Other Tools:** MATLAB, R, Stata, and Excel Add-In.
**Python Code Example (Public Access):**
```python
import sdmx
IMF_DATA = sdmx.Client('IMF_DATA')
# Example: Access the CPI dataset for the USA and Canada starting from 2018
data_msg = IMF_DATA.data('CPI', key='USA+CAN.CPI.CP01.IX.M', params={'startPeriod': 2018})
cpi_df = sdmx.to_pandas(data_msg)
```

## URL
[https://data.imf.org/](https://data.imf.org/)
