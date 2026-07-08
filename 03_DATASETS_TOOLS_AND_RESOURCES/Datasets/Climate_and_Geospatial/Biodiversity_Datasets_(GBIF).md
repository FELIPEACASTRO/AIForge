# Biodiversity Datasets (GBIF)

## Description
The Global Biodiversity Information Facility (GBIF) is an international network and data infrastructure funded by governments worldwide, aiming to provide free and open access to biodiversity occurrence data. It aggregates data from thousands of institutions around the world, making them available for research, policy, and decision-making. GBIF acts as a hub for primary biodiversity data, including occurrence records, checklists, and sampling event data, sourced from museum collections, citizen science, and other sources [1].

## Statistics
**Occurrence Records:** More than **3.1 billion** species occurrence records (as of September 2025) [3].
**Datasets:** More than **118,713** published datasets [1].
**Publishing Institutions:** More than **2,588** publishing institutions [1].
**Scientific Use (2023):** More than **1,700** scientific articles published in 2023 used GBIF-mediated data [4].
**Trend (2024):** The total number of occurrence records surpassed **3 billion** in 2024 [5].

## Features
**Primary Biodiversity Data:** Includes occurrence records (where and when a species was observed), taxonomic checklists, and sampling event data.
**Open Access:** All data are made available under open licenses, facilitating unrestricted use for research and policy.
**Standardization:** The data are standardized to the **Darwin Core Archive (DwC-A)** format, ensuring interoperability.
**Robust API:** Offers a complete API for programmatic access to the data, enabling integration with external tools and systems [2].
**Usage Monitoring:** Has a literature tracking system that identified more than **10,000 uses** in peer-reviewed articles through July 2024 [1].

## Use Cases
**Ecological and Biological Research:** Study of species distribution, ecological niche modeling, analysis of biodiversity patterns at taxonomic, temporal, and spatial scales [1].
**Conservation and Policy:** Monitoring the state of biodiversity and progress toward international targets, such as those of the Convention on Biological Diversity (CBD). GBIF is a key data source for indicators such as the Species Status Index [1].
**Climate Change:** Assessment of the impacts of climate change on the geographic distribution of species.
**Human Health and Agriculture:** Applications related to the distribution of disease vectors and species of agricultural interest [1].

## Integration
Access to GBIF data can be achieved in two main ways [2]:
1.  **GBIF.org (Direct Download):** Through the web interface, users can search, filter, and download data in three main formats: **Simple CSV** (selection of common terms), **Darwin Core Archive (DwC-A)** (includes original data and GBIF interpretation), and **Species list** (distinct list of names).
2.  **Programmatic Access (API):** GBIF offers a complete RESTful API for occurrence downloads and metadata access. Libraries such as **`rgbif`** (for R) and **`pygbif`** (for Python) facilitate integration and the download of large volumes of data directly into analysis environments. GBIF has also launched an experimental **SQL Downloads** API for more complex queries [2].
*   **Terms of Use:** Users must agree to the GBIF **Data User Agreement**, which requires proper citation of the data used [2].

## URL
[https://www.gbif.org/](https://www.gbif.org/)
