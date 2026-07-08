# Awesome Spectral Indices (ASI) and spyndex

## Description

"Awesome Spectral Indices" (ASI) is a standardized catalog and a Python library (`spyndex`) that compiles and manages a vast collection of spectral indices for Remote Sensing applications, including those that use the Red Edge (RE), Near Infrared (NIR), and Short-Wave Infrared (SWIR) bands. Published in 2023, ASI aims to standardize the use of spectral indices, providing formulas, required bands, and references for each index. It is an essential tool for feature engineering in Biomass and Agriculture projects, facilitating the application of indices such as NDVI, EVI, CIRE (Chlorophyll Index Red Edge), and NDWI (which frequently uses SWIR).

## Statistics

More than 200 cataloged spectral indices. Published in 2023. The `spyndex` library is compatible with Python and integrates with platforms such as Google Earth Engine (via `eemont`). The catalog is continuously updated by the community. The standardized spectral bands include 17 parameters, covering RE, NIR, and SWIR for Landsat, Sentinel-2, and MODIS.

## Features

Standardized catalog of more than 200 spectral indices. Includes indices that use the Red Edge (RE1, RE2, RE3), NIR (N, N2), and SWIR (S1, S2) bands from sensors such as Sentinel-2 and Landsat-8/9. Provides complete metadata for each index: short name, long name, formula, required bands, compatible platforms, and application domain (e.g., vegetation, water, soil). The Python library (`spyndex`) enables efficient computation of the indices over data arrays.

## Use Cases

**Biomass and Carbon Estimation:** Use of indices such as CIRE (Chlorophyll Index Red Edge) and SWIR-based indices to estimate chlorophyll content and water content, which are direct proxies for Aboveground Biomass (AGB) in forests and pastures. **Plant Health Monitoring:** Early detection of water stress (using SWIR indices) and nutrient deficiency (using Red Edge indices). **Crop and Pasture Mapping:** Classification and discrimination of different land cover types and pasture degradation stages, leveraging Red Edge sensitivity. **Productivity Modeling:** Integration of spectral indices as features in Machine Learning models to predict Gross Primary Productivity (GPP) and crop yield.

## Integration

Integration is done through the Python library `spyndex`.

**Installation:**
```bash
pip install spyndex
```

**Usage Example (Computing NDVI and CIRE with Sentinel-2 bands):**
```python
import spyndex
import numpy as np

# Example reflectance data (simulated values)
# N: NIR (B8), R: Red (B4), RE1: Red Edge 1 (B5)
reflectance_data = {
    "N": np.array([0.4, 0.3, 0.5]),
    "R": np.array([0.1, 0.05, 0.15]),
    "RE1": np.array([0.2, 0.15, 0.25])
}

# Compute multiple indices at once
indices = spyndex.compute(
    indices=["NDVI", "CIRE"],
    **reflectance_data
)

# Result (dictionary with the result arrays)
# print(indices["NDVI"])
# print(indices["CIRE"])
```

The catalog is also accessible in JSON and CSV formats for integration into other platforms (e.g., Google Earth Engine, R).

## URL

https://github.com/awesome-spectral-indices/awesome-spectral-indices