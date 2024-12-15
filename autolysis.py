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
#   "requests",
#   "ipykernel"
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



def check_ai_proxy_token()->str:
    """
    Checks for the presence of the AIPROXY_TOKEN environment variable.

    If the AIPROXY_TOKEN is not set, an error message is displayed, 
    and the program exits. If found, the token is printed and returned.

    Returns:
        str: The value of the AIPROXY_TOKEN environment variable.
    """
    token = os.getenv("AIPROXY_TOKEN")
    if not token:
        print("Error: AIPROXY_TOKEN environment variable not set.")
        sys.exit(1)
    print("AI Proxy token found.", token)
    return token

def load_csv()->pd.DataFrame:
    """
    Loads a CSV file into a pandas DataFrame.

    Attempts to read the CSV file specified by the global variable `filename`. 
    If successful, prints the number of rows and columns in the DataFrame and 
    returns it. If an error occurs during loading, the error message is printed, 
    and the program exits.

    Returns:
        pandas.DataFrame: The loaded DataFrame.

    Raises:
        SystemExit: If an error occurs while loading the CSV file.
    """
    try:
        df = pd.read_csv(filename)
        print(f"Successfully loaded {filename} with {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        sys.exit(1)

def explore_data(df: pd.date_range)->dict:
    """
    Performs a detailed indepth exploration of the given dataset.

    This function analyzes the DataFrame to provide:
    - Column names with their data types.
    - Descriptive statistics for numerical columns.
    - Counts of missing values in each column.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        dict: A dictionary summarizing data types, descriptive statistics, 
        and missing values for the dataset.
    """
    data_analysis = {
        "Columns and respective dtypes in dataset": df.dtypes.apply(str).to_dict(),
        "Mathematical Description done using pandas dataframe.describe(): ": df.describe().to_dict(),
        "Missing": df.isnull().sum().to_dict()
    }
    return data_analysis

def make_corr_heatmap(df:pd.DataFrame, count_images)->tuple:
    """
    Generates and saves a correlation heatmap for numerical features in the DataFrame.

    This function selects all numerical columns from the DataFrame, calculates their 
    correlation matrix, and creates a heatmap visualization. The heatmap is saved as 
    an image file in the current working directory. If no numerical columns are present, 
    the function exits without generating the heatmap.

    Args:
        df (pd.DataFrame): The input DataFrame containing data to analyze.
        count_images (int): A counter to track the number of saved images.

    Returns:
        dict: The correlation matrix represented as a dictionary. If no numerical columns 
        are found, returns None.
    """
    features = df.select_dtypes(include=[np.number]) 
    if features.empty:
        print("No Numerical columns found for Correlation Heatmap")
        return
    corr = features.corr()   
    sns.heatmap(corr, annot=True, annot_kws={"size": 5}, cmap="coolwarm", fmt='.2f')
    plt.xticks(fontsize=10, rotation=45, ha='right')  
    plt.yticks(fontsize=10, rotation=0)     
    plt.title(f"Correlation Heatmap")
    count_images += 1
    plt.savefig(f"corr_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved Correlation Heatmap", count_images)

    return corr.to_dict(), count_images

def perform_kmeans_clustering(df:pd.DataFrame, count_images:int, n_clusters:int = 3)->tuple:
    """
    Performs KMeans clustering on the numerical features of the provided DataFrame and generates a clustering plot.

    This function first selects numerical features from the DataFrame, handles missing values by imputing 
    them with the mean, scales the data, and applies KMeans clustering to segment the data into a specified 
    number of clusters. If the number of clusters is less than or equal to 3, it generates and saves a 
    scatter plot visualizing the clustering result.

    Args:
        df (pd.DataFrame): The input DataFrame containing data to be clustered.
        count_images (int): A counter to track the number of saved images.
        n_clusters (int, optional): The number of clusters to form. Defaults to 3.

    Returns:
        tuple: A tuple containing:
            - df (pd.DataFrame): The original DataFrame with an added 'Cluster' column, indicating the assigned cluster.
            - cluster_centers (dict): A dictionary representing the centroids of each cluster.
            - analysis_text (str): A descriptive analysis of the clustering result.
            - img_made (bool): A flag indicating if the clustering plot was generated and saved.

    Notes:
        - The function assumes that the DataFrame contains numerical columns for clustering.
        - The plot generated is saved as "Clusters.png" if the number of images saved is less than 3.
    """
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
        plt.savefig(f"Clusters.png", dpi=300, bbox_inches="tight")
        plt.close()
        img_made = True
        count_images += 1   
        print("Saved Kmeans Cluster", count_images)

    analysis_text = f"The KMeans clustering plot above shows the segmentation of the dataset into {n_clusters} clusters, based on the selected features: {', '.join(features.columns[:2])}. Each cluster, represented by a distinct color, groups similar data points together, highlighting underlying patterns in the dataset. The centroids of the clusters, located at the mean of the points, provide insight into the central tendencies of the data for each cluster."


    return df, cluster_centers, analysis_text, img_made, count_images

def histogram_generate(df:pd.DataFrame, count_images:int)->tuple:
    """
    Generates and saves a histogram (with optional KDE) for the best column in the DataFrame based on normality.

    This function identifies the numerical columns in the given DataFrame, performs the Shapiro-Wilk test for normality 
    on each numerical column, and selects the column with the highest p-value. If the p-value is greater than 0.05, 
    it is considered normally distributed. Based on this, it determines if the selected column is discrete or continuous 
    and generates a corresponding histogram. The plot is saved as an image with the column name as part of the filename.

    Args:
        df (pd.DataFrame): The input DataFrame containing numerical data to analyze.
        count_images (int): A counter to track the number of saved images.

    Returns:
        str: The name of the column that was selected as the best for normal distribution.

    Notes:
        - The function uses the Shapiro-Wilk test to assess normality.
        - It classifies columns with fewer than 10 unique values as discrete.
        - The plot is saved as an image with a filename based on the column name.
        - The function modifies the global variable `count_images` to track the number of saved images.
    """
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
    plt.savefig(f"{'_'.join(best_column.split())}_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved Histogram", count_images)
    return best_column, count_images

def plot_pie_chart_without_labels_for_small_categories(df:pd.DataFrame, count_images:int, column:str, threshold:int=5)->tuple:
    """
    Plots a pie chart of the distribution of a specified column, grouping small categories into an 'Other' category, 
    and saves it as an image.

    This function calculates the value counts and their percentages for the specified column in the DataFrame.
    It combines categories that account for less than a specified threshold percentage (default 5%) into a single 'Other' category. 
    The pie chart is generated and saved with the values displayed as percentages. Labels for the 'Other' category are hidden.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data to analyze.
        count_images (int): A counter to track the number of saved images.
        column (str): The column name in the DataFrame whose distribution will be visualized.
        threshold (float, optional): The percentage threshold for grouping small categories into the 'Other' category. Default is 5%.

    Returns:
        str: The name of the column used for the pie chart, to indicate which column was plotted.

    Notes:
        - The function filters categories based on their relative percentage and groups small categories (below the threshold) into an 'Other' category.
        - The resulting pie chart is saved as a PNG image, and the global counter `count_images` is incremented.
    """
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
    plt.savefig(f"{'_'.join(column.split())}_pie_chart.png", dpi=300, bbox_inches="tight")
    plt.close()
    count_images += 1
    print("Saved Pie Chart", count_images)
    return column, count_images

def plot_categorical_pie_chart(df:pd.DataFrame, count_images:int, threshold:int =10):
    """
    Generates pie charts for categorical columns in a DataFrame, where the number of unique values is below the specified threshold.
    If a column has fewer unique values than the threshold, the pie chart is plotted using the 'Other' category for small categories.

    The function iterates over all categorical columns in the DataFrame. For each column, if the number of unique values is less 
    than the threshold, it calls the `plot_pie_chart_without_labels_for_small_categories` function to create a pie chart for that column.

    Args:
        df (pd.DataFrame): The input DataFrame containing the categorical data.
        count_images (int): A counter to track the number of saved images.
        threshold (int, optional): The maximum number of unique values for which a pie chart will be generated. Default is 10.

    Returns:
        None: The function does not return any value but saves the pie chart image(s) for the selected column(s).

    Notes:
        - The function checks for categorical columns in the DataFrame and generates pie charts for columns with fewer unique values than the threshold.
        - If no categorical columns are found, a message will be printed.
        - The function uses the `plot_pie_chart_without_labels_for_small_categories` function to generate the pie charts.
    """

    # Select categorical columns
    categorical_columns = df.select_dtypes(include=['object', 'category'])

    
    for column in categorical_columns:
        # Check if the number of unique values is below the threshold
        if df[column].nunique() < threshold:
            return plot_pie_chart_without_labels_for_small_categories(df,count_images, column)
            
    
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

def write_heading(filename: str) -> str:
    """Generate the heading section for the README."""
    return f"# '{filename}' Dataset Analysis\n"

def write_overview(df: pd.DataFrame, filename: str, data_analysis: dict) -> str:
    """Generate the overview section for the README."""
    dataset_name = "".join(filename.split('.')[:-1])
    columns = df.columns.tolist()
    key_attributes = ", ".join(columns[:5])

    overview = f"""
## Overview

The dataset used in this analysis is the **{dataset_name}**, which contains data on various attributes related to {dataset_name.lower()}. The primary goal of this analysis is to explore the relationships between different features, identify patterns in the data, and provide visualizations that illustrate the distribution of key variables.

The dataset includes information such as **{key_attributes}**, which are crucial for understanding trends and making data-driven decisions. This report highlights key statistical metrics and visual representations of the dataset, including distributions, correlations, and clustering results.

This analysis will also provide insights into missing data, trends in the numerical and categorical features, and how different attributes relate to each other.
"""
    return overview

def write_summary_statistics(df: pd.DataFrame, data_analysis: dict) -> str:
    """Generate the summary statistics section for the README."""
    lines = ["## Summary Statistics"]
    lines.append(f"- Number of Columns: {df.shape[1]}")
    lines.append(f"- Number of Rows: {df.shape[0]}")
    
    missing_col = []
    for col in data_analysis['Missing']:
        if data_analysis["Missing"][col] > 0:
            missing_col.append(f"   - {col}: {data_analysis['Missing'][col]}")
    
    if missing_col:
        lines.append(f"- Number of Missing values in different Columns: ")
        lines += missing_col
    
    return "\n".join(lines)

def write_narrative(narration: str) -> str:
    """Generate the narrative section for the README."""
    return f"## Narrative of dataset: \n{narration}"

def write_visualizations(df: pd.DataFrame, filename: str, corr: dict, histogram_column: str, piechart_column: str, kmeans_image: bool) -> str:
    """Generate the visualizations section for the README."""
    lines = ["\n## Visualisations:"]

    # Correlation Heatmap
    if corr:
        lines.append("### Correlation Heatmap for the Numerical Data:")
        lines.append("A correlation heatmap was generated to visualize the relationships between numerical features in the dataset.\n")
        heatmap_path = "corr_heatmap.png"
        lines.append(f"![Correlation HeatMap]({heatmap_path})")
    
    # Histogram
    if histogram_column:
        lines.append(f"\n### Distribution for '{histogram_column}' Column of Dataset: \n")
        dist_path = f"{'_'.join(histogram_column.split())}_distribution.png"
        lines.append(f"![{histogram_column} distribution]({dist_path})")
    
    # Piechart
    if piechart_column:
        lines.append(f"\n### Pie-Chart for '{piechart_column}' Column of Dataset: \n")
        dist_path = f"{'_'.join(piechart_column.split())}_pie_chart.png"
        lines.append(f"![{piechart_column} Pie Chart]({dist_path})")

    # Kmeans Cluster
    if kmeans_image:
        lines.append(f"\n### Kmeans cluster for Dataset:")
        lines.append(analysis_text_kmeans + '\n')
        dist_path = "Clusters.png"
        lines.append(f"![Clusters png]({dist_path})")

    return "\n".join(lines)

def write_conclusion() -> str:
    """Generate the conclusion section for the README."""
    conclusion = """
## Conclusion

In this analysis, we explored the dataset to uncover patterns and relationships between its attributes. Here are some key takeaways:
1. The correlation heatmap revealed significant relationships between the numerical features, helping us identify potential areas for deeper analysis.
2. The histogram analysis showed the distribution of data for the selected column, providing insights into its nature (whether it's discrete or continuous).
3. The pie chart visualized the distribution of categorical values, making it easier to understand the prevalence of different categories.
4. K-means clustering helped group similar data points, uncovering potential segments within the dataset.

Overall, this analysis serves as a foundation for further exploration, predictive modeling, and decision-making.
"""
    return conclusion

def write_readme(df: pd.DataFrame, filename: str, data_analysis: dict, corr: dict, histogram_column: str, piechart_column: str, kmeans_image: bool, narration: str) -> None:
    """
        Generate and write the complete README file.
        It pieces together the narartion of the LLM with the indepth analysis made by script to write a indepth prooper analysis on the Dataset
    """

    with open("README.md", 'w') as f:
        # Write each section using the helper functions
        f.write(write_heading(filename))
        f.write(write_overview(df, filename, data_analysis))
        f.write(write_summary_statistics(df, data_analysis))
        f.write(write_narrative(narration))
        f.write(write_visualizations(df, filename, corr, histogram_column, piechart_column, kmeans_image))
        f.write(write_conclusion())  # Add conclusion section
    
    print("Readme Made!")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)

    AIPROXY_TOKEN = check_ai_proxy_token()

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
    data_analysis = explore_data(df)
    
    count_images = 0

    # Visulaisations
    corr = make_corr_heatmap(df, count_images)

    histogram_column = histogram_generate(df, count_images)
    piechart_column = plot_categorical_pie_chart(df, count_images)

    df, cluster_centers, analysis_text_kmeans, kmeans_image = perform_kmeans_clustering(df, count_images)

    # Narrations
    narration = get_narration(data_analysis, cluster_centers, corr, df.shape)

    write_readme(df, filename, data_analysis, corr, histogram_column, piechart_column, kmeans_image, narration)

