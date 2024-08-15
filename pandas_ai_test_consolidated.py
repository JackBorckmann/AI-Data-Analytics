import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandasai.llm.openai import OpenAI
from openai import OpenAI as OpenAIClient
from pandasai import Agent
from dotenv import load_dotenv
import streamlit as st
import time

# ... (keep existing imports and setup)
# Load environment variables from .env file
load_dotenv()

# Retrieve OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set the path to the northwinds directory relative to the script location
script_dir = os.path.dirname(os.path.abspath(__file__))
northwinds_dir = os.path.join(script_dir, 'northwinds')

def load_data():
    """
    Load CSV files from the northwinds directory into pandas DataFrames.
    
    Returns:
    dict: A dictionary containing DataFrames with their names as keys.
    """
    try:
        dataframes = {}
        for filename in os.listdir(northwinds_dir):
            if filename.endswith('.csv'):
                df_name = filename.split('.')[0]
                df = pd.read_csv(os.path.join(northwinds_dir, filename))
                dataframes[df_name] = df
        return dataframes
    except FileNotFoundError as e:
        print(f"Error loading data: {str(e)}")
        return None

def call_gpt_4o(prompt):
    """
    Send a prompt to the GPT-4o model and return the response.

    Args:
    prompt (str): The input prompt for the model.

    Returns:
    str: The model's response.
    """
    client = OpenAIClient(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
    model="gpt-4o", # Specify the GPT-4 Turbo model
    messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    print(response.choices[0].message.content)

    return response.choices[0].message.content

def preprocess_data(dataframes):
    """
    Preprocess the loaded dataframes, including date column conversion.

    Args:
    dataframes (list): List of pandas DataFrames to preprocess.

    Returns:
    list: The preprocessed DataFrames.
    """
    # Check if we have the expected number of dataframes
    if len(dataframes) != 7:
        print(f"Warning: Expected 7 dataframes, but got {len(dataframes)}.")
    
    # Attempt to convert date columns to datetime
    date_columns = ['orderdate', 'requireddate', 'shippeddate']
    
    for i, df in enumerate(dataframes):
        for col in date_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                    print(f"Converted {col} to datetime in dataframe {i}")
                except Exception as e:
                    print(f"Error converting {col} to datetime in dataframe {i}: {str(e)}")
    
    # Additional preprocessing steps can be added here
    
    return dataframes

def generate_metadata(dataframes):
    """
    Generate metadata for each dataframe using GPT-4 Turbo.

    Args:
    dataframes (dict): Dictionary of pandas DataFrames.

    Returns:
    dict: Metadata summaries for each DataFrame.
    """
    metadata = {}
    for df_name, df in dataframes.items():
        prompt = f"Analyze this dataframe named {df_name}:\n{df.head().to_string()}\n{df.dtypes.to_string()}\nProvide a brief summary of the data and its potential use in sales analysis."
        metadata[df_name] = call_gpt_4o(prompt)
    return metadata

def generate_queries(user_input, metadata, num_queries=5):
    metadata_str = "\n".join([f"{k}: {v}" for k, v in metadata.items()])
    prompt = f"""
    User input: {user_input}
    Metadata: {metadata_str}
    Generate {num_queries} diverse PandasAI queries to analyze sales data across multiple tables. 
    Aim for a mix of simple and complex queries that provide deep insights.
    Use lowercase column names in your queries.
    Each query should be a complete, single-line Python expression using Pandas operations.
    Do not include any explanatory text, SQL syntax, or code block markers.
    Ensure all column names used exist in the provided dataframes.
    Return only the Pandas queries, one per line.
    Example of a good query: 
    order_details.merge(orders, on='orderid').groupby('customerid')['quantity'].sum().sort_values(descending=True)
    """
    queries = call_gpt_4o(prompt)
    cleaned_queries = [line.strip().lstrip('0123456789. *') for line in queries.split('\n') if line.strip() and not line.startswith(('```', '#'))]
    return cleaned_queries[:num_queries]  # Ensure we return exactly num_queries

def perform_analysis(dataframes, queries, max_retries=3):
    results = []
    for i, query in enumerate(queries):
        for attempt in range(max_retries):
            try:
                llm = OpenAI(model_name="gpt-4o", api_key=OPENAI_API_KEY)
                agent = Agent(list(dataframes.values()), config={"llm": llm})
                response = agent.chat(f"Execute this Pandas query and explain the results in detail: {query}")
                results.append({"query": query, "result": response, "success": True})
                print(f"Query {i+1} executed successfully on attempt {attempt+1}")
                break
            except Exception as e:
                print(f"Error on attempt {attempt+1} for query '{query}': {str(e)}")
                if attempt == max_retries - 1:
                    results.append({"query": query, "result": str(e), "success": False})
                    print(f"Query {i+1} failed after {max_retries} attempts")
                else:
                    time.sleep(2)  # Wait before retrying
    return results

def interpret_results(results):
    successful_results = [r for r in results if r["success"]]
    failed_results = [r for r in results if not r["success"]]
    
    prompt = f"""
    Interpret these analysis results:
    Successful queries and results:
    {successful_results}
    
    Failed queries:
    {failed_results}
    
    Provide detailed insights on sales trends and performance. 
    Highlight any interesting patterns or anomalies in the data.
    Suggest potential areas for further investigation based on these results.
    If any queries failed, suggest alternative approaches or queries that might yield similar insights.
    """
    interpretation = call_gpt_4o(prompt)
    return interpretation

def visualize_insights(results):
    visualizations = []
    for result in results:
        if result["success"] and isinstance(result["result"], pd.DataFrame):
            df = result["result"]
            if len(df) > 1 and df.select_dtypes(include=[np.number]).shape[1] > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                df.plot(kind='bar', ax=ax)
                plt.title(f"Visualization for query: {result['query'][:50]}...")
                plt.tight_layout()
                visualizations.append(fig)
    return visualizations

def main():
    st.title("PandasAI Sales Analysis")

    dataframes = load_data()
    if not dataframes:
        st.error("Failed to load data. Please check your data files and try again.")
        return

    st.write("Loaded Dataframes:")
    for df_name, df in dataframes.items():
        st.write(f"{df_name}: {df.shape[0]} rows, {df.shape[1]} columns")

    user_input = st.text_input("Enter your analysis query:", "Analyze sales trends across different regions and identify top-performing products")
    num_queries = st.slider("Number of queries to generate", min_value=3, max_value=10, value=5)

    if st.button("Start Analysis"):
        with st.spinner("Generating metadata..."):
            metadata = generate_metadata(dataframes)

        with st.spinner(f"Generating {num_queries} queries..."):
            queries = generate_queries(user_input, metadata, num_queries)
        
        #st.subheader("Generated Queries")
        #for i, query in enumerate(queries, 1):
        #    st.code(query, language="python")

        with st.spinner("Performing analysis..."):
            results = perform_analysis(dataframes, queries)

        st.subheader("Analysis Results")
        for i, result in enumerate(results, 1):
            st.write(f"Query {i}:")
            st.code(result["query"], language="python")
            if result["success"]:
                st.write("Result:")
                st.write(result["result"])
            else:
                st.error(f"Query failed: {result['result']}")

        with st.spinner("Interpreting results..."):
            interpretation = interpret_results(results)
        
        st.subheader("Interpretation")
        st.write(interpretation)

        # with st.spinner("Generating visualizations..."):
        #     visualizations = visualize_insights(results)
        
        # if visualizations:
        #     st.subheader("Visualizations")
        #     for fig in visualizations:
        #         st.pyplot(fig)

if __name__ == "__main__":
    main()