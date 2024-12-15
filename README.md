# TDS-IITM-Project2
## A LLM-based Data Analysis and Narrative Generation Script

### Overview

The `TDS-IITM-Project2` repository contains a script that performs an automated analysis of datasets, generates visualizations, and provides a narrative using an LLM model. This project aims to help users quickly understand the structure and key insights from their data with the aid of machine learning models, statistical methods, and data visualization techniques. 

The script allows users to load datasets, explore statistical summaries, handle missing values, perform clustering, create correlation heatmaps, generate distribution histograms, and more. Additionally, it integrates an LLM (Language Learning Model) for narrative generation that provides insights into the data analysis.

### Features

- **Data Loading & Encoding Detection:** Automatically detects and loads datasets with the correct encoding, ensuring accurate data processing.
  
- **Data Exploration:** Includes detailed statistical summaries, type descriptions, and missing value analysis for a comprehensive view of the data.

- **Visualizations:** 
  - Correlation heatmaps to understand relationships between numerical features.
  - Histograms to display the distribution of numerical columns.
  - Pie charts for categorical data distribution.
  - KMeans clustering for grouping data points based on similarities.

- **Narrative Generation:** Utilizes the GPT-4o-mini model via API to generate a well-written, insightful narrative based on the dataset analysis.

### Requirements

- Python version 3.11 or higher
- The following Python packages:
  - `httpx`
  - `pandas`
  - `seaborn`
  - `openai`
  - `matplotlib`
  - `numpy`
  - `scikit-learn`
  - `charset_normalizer`
  - `requests`

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/kunj-10/TDS-IITM-Project2.git
   cd TDS-IITM-Project2

2. Install uv

3. set your *AIPROXY_TOKEN*

4. run:
    `uv run autolysis.py <dataset.csv>`