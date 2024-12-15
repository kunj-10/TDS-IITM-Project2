# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "pandas",
#   "seaborn",
#   "openai",
#   "matplotlib",
#   "numpy",
#   "scikit-learn",
#   "charset_normalizer",
#   "requests"
# ]
# ///

import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from charset_normalizer import from_path
from scipy.stats import skew, kurtosis, shapiro
from os import path
import requests



def check_ai_proxy_token():
    token = os.getenv("AIPROXY_TOKEN")
    if not token:
        print("Error: AIPROXY_TOKEN environment variable not set.")
        sys.exit(1)
    print("AI Proxy token found.", token)
    return token

def load_csv():
    try:
        df = pd.read_csv(filename)
        print(f"Successfully loaded {filename} with {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        sys.exit(1)

def explore_data_generic(df):
    data_analysis = {
        "Columns and respective dtypes in dataset": df.dtypes.apply(str).to_dict(),
        "Mathematical Description done using pandas dataframe.describe(): ": df.describe().to_dict(),
        "Missing": df.isnull().sum().to_dict()
    }
    return data_analysis

def make_corr_heatmap(df):
    global count_images
    features = df.select_dtypes(include=[np.number]) 
    if features.empty:
        print("No Numerical columns found for Correlation Heatmap")
        return
    corr = features.corr()   
    sns.heatmap(corr, annot=True, annot_kws={"size": 5}, cmap="coolwarm", fmt='.2f')
    plt.xticks(fontsize=10, rotation=45, ha='right')  
    plt.yticks(fontsize=10, rotation=0)     
    plt.title(f"Heatmap for {"".join(filename.split('.')[:-1])}")
    count_images += 1
    plt.savefig(f"{"".join(filename.split('.')[:-1])}_corr_heatmap.png", dpi=300, bbox_inches="tight")
    print("Saved Correlation Heatmap", count_images)

    return corr.to_dict()

def perform_kmeans_clustering(df, n_clusters=3):
    global count_images
    features = df.select_dtypes(include=[np.number]) 
    if features.empty:
        print("No Numerical columns found for Kmeans Clustering")
        return
    imputer = SimpleImputer(strategy="mean")
    features = pd.DataFrame(imputer.fit_transform(features), columns=features.columns)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(features_scaled)

    cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=features.columns).to_dict()
        
    img_made = False
    if count_images < 3:
        plt.scatter(df[features.columns[0]], df[features.columns[1]], c=df['Cluster'], cmap='viridis')
        plt.xlabel(features.columns[0])
        plt.ylabel(features.columns[1])
        plt.title('Clusters')
        plt.savefig(f"{"".join(filename.split('.')[:-1])}_Clusters.png", dpi=300, bbox_inches="tight")
        img_made = True
        count_images += 1   
        print("Saved Kmeans Cluster", count_images)

    analysis_text = f"The KMeans clustering plot above shows the segmentation of the dataset into {n_clusters} clusters, based on the selected features: {', '.join(features.columns[:2])}. Each cluster, represented by a distinct color, groups similar data points together, highlighting underlying patterns in the dataset. The centroids of the clusters, located at the mean of the points, provide insight into the central tendencies of the data for each cluster."


    return df, cluster_centers, analysis_text, img_made

def histogram_generate(df):
    global count_images
    # Identify numerical columns
    numeric_columns = df.select_dtypes(include=['number'])
    if numeric_columns.empty:
        print("No Numerical columns found for Histogram Generation")
        return

    # Perform the Shapiro-Wilk test for normality
    p_values = numeric_columns.apply(lambda x: shapiro(x)[1])  # Apply Shapiro test and get p-value
    best_column = p_values.idxmax()  # Select the column with the highest p-value

    # If p-value > 0.05, the data is normally distributed
    if p_values[best_column] > 0.05:
        print(f"The best column for normal distribution is: {best_column}")
    else:
        print(f"No column fits a normal distribution well. Best column: {best_column}, p-value: {p_values[best_column]}")

    # Determine if the best column is discrete or continuous
    # A discrete column typically has fewer unique values
    unique_values = df[best_column].nunique()
    is_discrete = unique_values < 10  # You can adjust this threshold based on your dataset

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    if is_discrete:
        # Plot only histogram for discrete data (no KDE)
        sns.histplot(df[best_column], kde=False, color='blue', bins=unique_values)
        plt.title(f"Discrete Distribution of {best_column}")
    else:
        # Plot histogram and KDE for continuous data
        sns.histplot(df[best_column], kde=True, color='blue', bins=20)
        plt.title(f"Continuous Distribution of {best_column}")
    
    plt.xlabel(best_column)
    plt.ylabel("Frequency")
    plt.grid(True)
    count_images += 1
    plt.savefig(f"{"".join(filename.split('.')[:-1])}_{'_'.join(best_column.split())}_distribution.png", dpi=300, bbox_inches="tight")
    print("Saved Histogram", count_images)
    return best_column

def plot_pie_chart_without_labels_for_small_categories(df, column, threshold=5):
    global count_images
    # Calculate the value counts and their percentages
    value_counts = df[column].value_counts()
    percentages = value_counts / value_counts.sum() * 100

    # Filter categories that are smaller than the threshold (5%)
    small_categories = percentages[percentages < threshold].index
    # Combine the small categories into a single "Other" category
    value_counts['Other'] = value_counts[small_categories].sum()

    # Remove small categories from value_counts
    value_counts = value_counts.drop(small_categories)

    # Get a color palette from Seaborn (or define your own list of colors)
    colors = sns.color_palette('viridis', len(value_counts))

    # Plot the pie chart
    plt.figure(figsize=(10, 10))  # Set the figure size
    wedges, texts, autotexts = plt.pie(value_counts, labels=value_counts.index, 
                                       autopct='%1.1f%%', colors=colors, 
                                       textprops={'fontsize': 10}, 
                                       labeldistance=1.2, pctdistance=0.85, 
                                       wedgeprops={'width': 0.4})

    # Hide labels for the "Other" category by setting the label to an empty string
    for i, text in enumerate(texts):
        if texts[i].get_text() == "Other":
            text.set_text('Other')

    # Add title and remove the y-label for cleaner chart
    plt.title(f'Distribution of {column}')
    plt.ylabel('')  # Hide the y-label for a cleaner pie chart
    plt.savefig(f"{"".join(filename.split('.')[:-1])}_{'_'.join(column.split())}_pie_chart.png", dpi=300, bbox_inches="tight")
    count_images += 1
    print("Saved Pie Chart", count_images)
    return column

def plot_categorical_pie_chart(df, threshold=10):
    # Select categorical columns
    categorical_columns = df.select_dtypes(include=['object', 'category'])

    
    for column in categorical_columns:
        # Check if the number of unique values is below the threshold
        if df[column].nunique() < threshold:
            return plot_pie_chart_without_labels_for_small_categories(df, column)
            
    
    # if categorical_columns.empty:
    print("No Categorical Columns found for finding the Pie Chart")


def get_narration(data_analysis, cluster_centers, corr, shape):
    """Use GPT-4o-mini to generate narration for the dataset."""

    prompt = f"""
    You are a data scientist analyzing a dataset. Here is the context:

    1. **Columns and Data Types**:
       {data_analysis['Columns and respective dtypes in dataset']}
    2. **Statistical Summary**:
       Key statistics for numerical columns:
       {data_analysis['Mathematical Description done using pandas dataframe.describe(): ']}
    3. **Missing Values**:
       {data_analysis['Missing']}
    4. **KMeans Cluster Centers**:
       {cluster_centers}
    5. **Correlation Heatmap (top correlations)**:
       {corr}
    6. **Dataset Shape**:
       {shape}

    Write a concise, engaging narration that highlights:
    - Key patterns or anomalies in the data types and statistics.
    - The significance of missing data and its potential impact.
    - Insights from clustering and correlation analysis.
    - Observations about the dataset's size and structure.
    """

    headers = {
        'Authorization': f'Bearer {AIPROXY_TOKEN}',
        'Content-Type': 'application/json'
    }

    # Making the request to the AI Proxy endpoint
    try:
        response = requests.post(
            "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 600
            },
            headers=headers,
        )

        # Check for a successful response
        if response.status_code == 200:
            response_data = response.json()
            narration = response_data['choices'][0]['message']['content'].strip()

            # Get the monthly cost if available in the response
            monthly_cost = response_data.get('usage', {}).get('monthlyCost', 'Not Available')
            print(f"Monthly Credit Usage: {monthly_cost} USD")

            # Ensure the cost is below 5.0 USD as per your requirement
            if isinstance(monthly_cost, (int, float)) and monthly_cost >= 5.0:
                print("Warning: Monthly cost exceeds the $5.00 limit.")

            return narration
        else:
            print(f"Error: Received status code {response.status_code}")
            return "Error in generating narration."

    except Exception as e:
        print("Error generating narration:", e)
        return "Error in generating narration."

def writeReadme(df, filename, data_analysis, corr, histogram_column, piechart_column, kmeans_image, narration):
    with open ("README.md", 'w') as f:
        lines_to_write = []

        # Heading of Readme.md
        lines_to_write.append(f"# '{filename}' Dataset Analysis\n")
        
        
        #Overview of Analysis
        dataset_name = "".join(filename.split('.')[:-1])

        columns = df.columns.tolist()
        key_attributes = ", ".join(columns[:5])

        overview = f"""
## Overview

The dataset used in this analysis is the **{dataset_name}**, which contains data on various attributes related to {dataset_name.lower()}. The primary goal of this analysis is to explore the relationships between different features, identify patterns in the data, and provide visualizations that illustrate the distribution of key variables.

The dataset includes information such as **{key_attributes}**, which are crucial for understanding trends and making data-driven decisions. This report highlights key statistical metrics and visual representations of the dataset, including distributions, correlations, and clustering results.

This analysis will also provide insights into missing data, trends in the numerical and categorical features, and how different attributes relate to each other.
"""

        lines_to_write.append(overview)
        
        
        # Some basic statistics on dataset
        lines_to_write.append("## Summary Statistics")
        lines_to_write.append(f"- Number of Columns: {df.shape[1]}")
        lines_to_write.append(f"- Number of Rows: {df.shape[0]}")
        missing_col = []
        for col in data_analysis['Missing']:
            if data_analysis["Missing"][col] > 0:
                missing_col.append(f"   - {col}: {data_analysis['Missing'][col]}")
        if missing_col:
            lines_to_write.append(f"- Number of Missing values in different Columns: ")
            lines_to_write += missing_col

        # LLM Story or Narrative
        lines_to_write.append("## Narrative of dataset: ")
        lines_to_write.append(narration)
        
        #Visualisations and Their Breif description
        lines_to_write.append("\n## Visualisations:")

        # Correlation Heatmap
        if corr:
            lines_to_write.append("### Correlation Heatmap for the Numerical Data:")
            lines_to_write.append("A correlation heatmap was generated to visualize the relationships between numerical features in the dataset.\n")
            heatmap_path =f"{"".join(filename.split('.')[:-1])}_corr_heatmap.png"
            lines_to_write.append(f"![Correlation HeatMap]({heatmap_path})")
        
        # Histogram
        if histogram_column:
            lines_to_write.append(f"\n### Distribution for '{histogram_column}' Column of Dataset: \n")
            dist_path = f"{"".join(filename.split('.')[:-1])}_{'_'.join(histogram_column.split())}_distribution.png"
            lines_to_write.append(f"![{histogram_column} distribution]({dist_path})")
        
        # Piechart
        if piechart_column:
            lines_to_write.append(f"\n### Pie-Chart for '{piechart_column}' Column of Dataset: \n")
            dist_path = f"{"".join(filename.split('.')[:-1])}_{'_'.join(piechart_column.split())}_pie_chart.png"
            lines_to_write.append(f"![{piechart_column} Pie Chart]({dist_path})")

        # Kmeans Cluster
        if kmeans_image:
            lines_to_write.append(f"\n### Kmeans cluster for Dataset:")
            lines_to_write.append(analysis_text_kmeans+'\n')
            dist_path = f"{"".join(filename.split('.')[:-1])}_Clusters.png"
            lines_to_write.append(f"![Clusters png]({dist_path})")
        lines_to_write.append("\n\n")
        

        # Finally writting to Readme
        f.writelines([line + "\n" for line in lines_to_write])
        print("Readme Made!")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)

    AIPROXY_TOKEN = check_ai_proxy_token()
    count_images = 0

    filename = sys.argv[1]

    # Load Data
    try:
        result = from_path(filename).best()
        print(f"Detected encoding: {result.encoding}")
        df = pd.read_csv(filename, encoding=result.encoding)
    except Exception as e:
        print("Error reading the encoding trying with default utf-8")
        df = pd.read_csv(filename)

    print("\nInitial setup complete. Ready for analysis!")

    # Basic Data analysis
    data_analysis = explore_data_generic(df)
    
    # Visulaisations
    corr = make_corr_heatmap(df)

    histogram_column = histogram_generate(df)
    piechart_column = plot_categorical_pie_chart(df)

    df, cluster_centers, analysis_text_kmeans, kmeans_image = perform_kmeans_clustering(df)

    # Narrations
    narration = get_narration(data_analysis, cluster_centers, corr, df.shape)

    writeReadme(df, filename, data_analysis, corr, histogram_column, piechart_column, kmeans_image, narration)


