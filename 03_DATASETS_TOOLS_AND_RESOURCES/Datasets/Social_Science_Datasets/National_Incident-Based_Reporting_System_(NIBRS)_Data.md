# National Incident-Based Reporting System (NIBRS) Data

## Description
The National Incident-Based Reporting System (NIBRS) is the FBI's primary crime data collection system in the United States, replacing the older Summary Reporting System (SRS) of the Uniform Crime Reporting (UCR) Program. NIBRS collects detailed incident-level data on crimes, including information about victims, offenders, the relationship between them, properties involved, and weapons used. It is the most comprehensive and granular source of crime statistics in the U.S., making it essential for criminology research and artificial intelligence applications.

## Statistics
The dataset is massive, with millions of records per year. For example, the 2022 report contained more than 11 million criminal offenses. The annual master data files are of considerable size (hundreds of megabytes or gigabytes) and are made available in fixed-length ASCII text format, compressed in WinZip. The data is updated annually, with the most recent version available (as of 2025) being that of 2024.

## Features
Incident-level data (more detailed than summary data); Collects information on 81 types of crimes (compared to 10 in the previous system); Includes details about the context of the crime (time, location, weapons, property value); Detailed demographic information about victims and offenders (age, sex, race, ethnicity); Enables the analysis of multiple crimes within a single incident.

## Use Cases
Crime and criminal 'hotspot' prediction using Machine Learning; Analysis of crime trends and allocation of police resources; Academic research in criminology and sociology; Development of offender profiling models; Evaluation of public safety policies.

## Integration
The data can be accessed and downloaded in several ways: 1. **Crime Data API:** A read-only web service that returns data in JSON or CSV. 2. **Direct Downloads:** Annual Master Files in compressed ASCII text format (requires programming knowledge for extraction). 3. **Data Discovery Tool:** Allows the creation of custom queries and the download of data subsets in CSV. Complete technical documentation (NIBRS Data Dictionary) is available on the FBI portal.

## URL
[https://cde.ucr.cjis.gov/](https://cde.ucr.cjis.gov/)
