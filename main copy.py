import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from openai import OpenAI
from dotenv import load_dotenv
import os
import shelve
import tempfile
import json
import plotly


# Load environment variables from .env file
load_dotenv()

class DataAnalysisChatbot:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'chatbot_data')
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            st.warning("OpenAI API key not found in .env file. Please set the OPENAI_API_KEY variable in your .env file.")
        else:
            self.client = OpenAI(api_key=self.openai_api_key)

    def _get_db(self):
        return shelve.open(self.db_path)

    @property
    def dataframes(self):
        with self._get_db() as db:
            return db.get('dataframes', {})

    @dataframes.setter
    def dataframes(self, value):
        with self._get_db() as db:
            db['dataframes'] = value

    @property
    def analyses_results(self):
        with self._get_db() as db:
            return db.get('analyses_results', {})

    @analyses_results.setter
    def analyses_results(self, value):
        with self._get_db() as db:
            db['analyses_results'] = value

    @property
    def chat_history(self):
        with self._get_db() as db:
            return db.get('chat_history', [])

    @chat_history.setter
    def chat_history(self, value):
        with self._get_db() as db:
            db['chat_history'] = value

    @property
    def graph_memory(self):
        with self._get_db() as db:
            return db.get('graph_memory', {})

    @graph_memory.setter
    def graph_memory(self, value):
        with self._get_db() as db:
            db['graph_memory'] = value

    def load_file(self, file):
        file_name = file.name
        file_extension = os.path.splitext(file_name)[1].lower()

        if file_extension == '.csv':
            return self.load_csv(file)
        elif file_extension in ['.xlsx', '.xls']:
            return self.load_excel(file)
        else:
            return f"Unsupported file format: {file_extension}"

    def load_csv(self, file):
        try:
            df = pd.read_csv(file)
            dataframes = self.dataframes
            dataframes[file.name] = df
            self.dataframes = dataframes
            return f"Successfully loaded {file.name}. Shape: {df.shape}"
        except Exception as e:
            return f"Error loading {file.name}: {str(e)}"

    def load_excel(self, file):
        try:
            df = pd.read_excel(file)
            dataframes = self.dataframes
            dataframes[file.name] = df
            self.dataframes = dataframes
            return f"Successfully loaded {file.name}. Shape: {df.shape}"
        except Exception as e:
            return f"Error loading {file.name}: {str(e)}"

    def get_basic_info(self, file_name):
        if file_name not in self.dataframes:
            return f"File {file_name} not found."
        df = self.dataframes[file_name]
        info = f"Basic information for {file_name}:\n"
        info += f"Shape: {df.shape}\n"
        info += f"Columns: {', '.join(df.columns)}\n"
        info += f"Data types:\n{df.dtypes}\n"
        info += f"Summary statistics:\n{df.describe().to_string()}\n"
        return info

    def create_histogram(self, file_name, column):
            if file_name not in self.dataframes:
                return f"File {file_name} not found."
            df = self.dataframes[file_name]
            if column not in df.columns:
                return f"Column {column} not found in {file_name}."
            fig = px.histogram(df, x=column, title=f"Histogram of {column} in {file_name}")
            graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            self.add_to_graph_memory("Histogram", file_name, f"Histogram of {column}", graph_json)
            return fig

    def create_scatter_plot(self, file_name, x_column, y_column):
        if file_name not in self.dataframes:
            return f"File {file_name} not found."
        df = self.dataframes[file_name]
        if x_column not in df.columns or y_column not in df.columns:
            return f"One or both columns not found in {file_name}."
        fig = px.scatter(df, x=x_column, y=y_column, title=f"Scatter plot: {x_column} vs {y_column} in {file_name}")
        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        self.add_to_graph_memory("Scatter Plot", file_name, f"{x_column} vs {y_column}", graph_json)
        return fig

    def perform_correlation_analysis(self, file_name):
        if file_name not in self.dataframes:
            return f"File {file_name} not found."
        df = self.dataframes[file_name]
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        fig = px.imshow(corr_matrix, title=f"Correlation Heatmap for {file_name}")
        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        self.add_to_graph_memory("Correlation Analysis", file_name, "Correlation heatmap", graph_json)
        result = corr_matrix.to_string()
        self.add_to_analysis_memory(f"Correlation Analysis for {file_name}", result)
        return fig, result

    def perform_pca(self, file_name):
        if file_name not in self.dataframes:
            return f"File {file_name} not found."
        df = self.dataframes[file_name]
        numeric_df = df.select_dtypes(include=[np.number])
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        pca = PCA()
        pca_result = pca.fit_transform(scaled_data)
        explained_variance_ratio = pca.explained_variance_ratio_
        fig = px.line(x=range(1, len(explained_variance_ratio) + 1), y=np.cumsum(explained_variance_ratio),
                      title=f"Cumulative Explained Variance Ratio for {file_name}")
        fig.update_layout(xaxis_title="Number of Components", yaxis_title="Cumulative Explained Variance Ratio")
        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        self.add_to_graph_memory("PCA", file_name, "Cumulative Explained Variance Ratio", graph_json)
        result = f"Explained variance ratio: {explained_variance_ratio.tolist()}"
        self.add_to_analysis_memory(f"PCA for {file_name}", result)
        return fig, result

    def perform_clustering(self, file_name, n_clusters=3):
        if file_name not in self.dataframes:
            return f"File {file_name} not found."
        df = self.dataframes[file_name]
        numeric_df = df.select_dtypes(include=[np.number])
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_data)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)
        result_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
        result_df['Cluster'] = cluster_labels
        fig = px.scatter(result_df, x='PC1', y='PC2', color='Cluster', title=f"K-means Clustering (k={n_clusters}) for {file_name}")
        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        self.add_to_graph_memory("Clustering", file_name, f"K-means Clustering (k={n_clusters})", graph_json)
        result = f"Performed K-means clustering with {n_clusters} clusters. Cluster centers: {kmeans.cluster_centers_.tolist()}"
        self.add_to_analysis_memory(f"Clustering for {file_name}", result)
        return fig, result

    def generate_insights(self, file_name):
        if file_name not in self.dataframes:
            return f"File {file_name} not found."
        df = self.dataframes[file_name]
        
        summary = f"Dataset summary for {file_name}:\n"
        summary += f"Number of rows: {df.shape[0]}\n"
        summary += f"Number of columns: {df.shape[1]}\n"
        summary += f"Columns: {', '.join(df.columns)}\n"
        summary += f"Numeric columns summary:\n{df.describe().to_string()}\n"
        summary += f"Full dataset:\n{df.to_string()}\n\n"
        
        prompt = f"""
        Given the following dataset summary, provide 3-5 key insights about the data:

        {summary}

        Focus on:
        1. Potential patterns or trends in the numeric data
        2. Unusual or extreme values
        3. Relationships between different variables
        4. Distribution of data in key columns
        5. Any notable characteristics of non-numeric data
        
        Present your insights in a clear, numbered list format.

        Also suggest 2-4 hypotheses that could be tested with further analysis:

        Your hypotheses should:
        1. Be specific and testable
        2. Relate to potential relationships between variables
        3. Address possible business or research questions that this data could answer
        4. Consider both numeric and categorical variables when applicable

        Present your hypotheses in a clear, numbered list format.
        
        And finally provide 2-4 recommendations for further analysis or data collection:

        Your recommendations should:
        1. Suggest specific analyses that could yield valuable insights
        2. Identify any potential gaps in the current data
        3. Propose visualizations that could help understand the data better
        4. Consider both descriptive and inferential statistical approaches

        Present your recommendations in a clear, numbered list format.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                #gpt-4o
                messages=[
                    {"role": "system", "content": "You are a helpful data analyst assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            insights = response.choices[0].message.content
            self.add_to_analysis_memory(f"Generated Insights for {file_name}", insights)
            return insights
        except Exception as e:
            return f"Error generating insights: {str(e)}"

    def generate_cross_file_insights(self, selected_files):
        if len(selected_files) < 2:
            return "At least two files are required for cross-file analysis."

        try:
            # Prepare a summary of selected dataframes
            summary = "Cross-file analysis summary:\n\n"
            for file_name in selected_files:
                if file_name not in self.dataframes:
                    return f"File {file_name} not found."
                df = self.dataframes[file_name]
                summary += f"File: {file_name}\n"
                summary += f"Number of rows: {df.shape[0]}\n"
                summary += f"Number of columns: {df.shape[1]}\n"
                summary += f"Columns: {', '.join(df.columns)}\n"
                summary += f"Numeric columns summary:\n{df.describe().to_string()}\n\n"
                summary += f"Full dataset:\n{df.to_string()}\n\n"

            # Find common columns across selected dataframes
            common_columns = set.intersection(*[set(self.dataframes[file].columns) for file in selected_files])
            summary += f"Common columns across selected files: {', '.join(common_columns)}\n\n"

            # Generate insights using OpenAI API
            prompt = f"""
            You are a data analysis expert with direct access to multiple datasets. Your task is to perform a comprehensive cross-file analysis and provide concrete, data-driven insights. Use the provided data to perform calculations and analyses directly. Your goal is to provide valuable insights and actionable recommendations based on the combined data from these files. Please follow these steps in your analysis:

            1. Theme Identification:
                Identify the 5 most significant themes that emerge across the files.
                Each theme should incorporate data from at least 3 files.
                List which files contribute to each identified theme.
                Consider common data types, business processes, or organizational aspects that appear in multiple files.

            2. Analytical Insights:
                For each theme, generate 1-2 analytical insights (aim for 5-10 total insights).
                These insights should be based on cross-file analysis, drawing conclusions from multiple data sources.
                Perform actual calculations and analyses on the data to support your insights.
                Look for patterns, trends, or anomalies that become apparent when considering multiple data sources together.
                Consider the business implications of the combined data.

            3. Action Plans:
                Create a short action plan (2-5 sentences) for each insight.
                Suggest practical, implementable actions to address problems or improve processes based on the insights.
                Consider both short-term actions and longer-term strategic implications.

            4. Additional Analytical Elements:
                Choose and implement five additional analytical elements from the following list:
                a. Correlation analysis between key variables across files (perform actual correlations)
                b. Time series analysis for any temporal data (calculate trends, seasonality, etc.)
                c. Anomaly detection across datasets (identify actual outliers)
                d. Predictive modeling using data from multiple files (create a simple predictive model)
                e. Cluster analysis to identify groups or segments (perform actual clustering)
                f. Network analysis if there are relational data across files
                g. Text analysis for any textual data fields (perform sentiment analysis, word frequency, etc.)
                h. Geospatial analysis if location data is present (analyze geographical patterns)
                i. Comparative analysis between different subsets of data (perform statistical tests)
                j. Risk assessment based on cross-file data patterns (quantify risks)
                
                For each chosen element, perform the actual analysis using the provided data and report the results.

            5. Summary and Key Takeaways:
                Provide a high-level summary of the cross-file analysis (3-5 sentences).
                List the top 3-5 key takeaways that decision-makers should focus on.

            Remember to adapt your analysis to the specific nature and content of the provided datasets. Your analysis should provide a comprehensive view of the business operations, relationships, and potential areas for improvement across various aspects of the organization.

            Use pandas-like operations to perform calculations directly on the dataframes provided in the context. For example:
            - To calculate correlations: df.corr()
            - To perform time series analysis: df.groupby(pd.Grouper(freq='M')).mean()
            - To detect anomalies: df[(df - df.mean()).abs() > 3*df.std()]

            Given the following summary of the datasets:
            {summary}

            Please perform the cross-file analysis as described above. Ensure that your response is well-structured, with clear headings for each section of the analysis. Provide specific numerical results and data-driven insights based on actual calculations and analyses.
            """

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful data analyst assistant with direct access to pandas dataframes."},
                    {"role": "user", "content": prompt}
                ]
            )
            insights = response.choices[0].message.content
            self.add_to_analysis_memory(f"Cross-file Insights for {', '.join(selected_files)}", insights)
            return insights
        except Exception as e:
            return f"Error generating cross-file insights: {str(e)}"

    def add_to_chat_history(self, role, content):
        history = self.chat_history
        history.append({"role": role, "content": content})
        self.chat_history = history

    def add_to_graph_memory(self, analysis_type, file_name, description, graph_json):
        memory = self.graph_memory
        if analysis_type not in memory:
            memory[analysis_type] = []
        memory[analysis_type].append({"file": file_name, "description": description, "graph_data": graph_json})
        self.graph_memory = memory

    def add_to_analysis_memory(self, analysis_type, result):
        analyses_results = self.analyses_results
        analyses_results[analysis_type] = result
        self.analyses_results = analyses_results

    def get_memory_context(self):
            context = "Previous analyses and insights:\n\n"
            
            for analysis, result in self.analyses_results.items():
                context += f"{analysis}:\n{result}\n\n"
            
            context += "Generated graphs:\n\n"
            for analysis_type, graphs in self.graph_memory.items():
                context += f"{analysis_type}:\n"
                for graph in graphs:
                    context += f"- {graph['file']}: {graph['description']}\n"
                context += "\n"
            
            context += "Chat history:\n\n"
            for message in self.chat_history[-5:]:  # Include only the last 5 messages
                context += f"{message['role']}: {message['content']}\n"
            
            return context

    def chat_with_data(self, file_names, user_question):
        if not file_names:
            return "No files selected for analysis."
        
        context = "Dataset summaries:\n\n"
        for file_name in file_names:
            if file_name not in self.dataframes:
                return f"File {file_name} not found."
            
            df = self.dataframes[file_name]
            
            context += f"Dataset: {file_name}\n"
            context += f"Shape: {df.shape}\n"
            context += f"Columns: {', '.join(df.columns)}\n"
            context += f"Data types:\n{df.dtypes.to_string()}\n"
            context += f"Summary statistics:\n{df.describe().to_string()}\n"
            context += f"Full dataset:\n{df.to_string()}\n\n"
        
        memory_context = self.get_memory_context()
        
        prompt = f"""
        You are a data analysis assistant with direct access to the datasets. Your task is to answer questions about the data accurately and provide specific numerical answers when appropriate.

        Given the following dataset information, previous analysis results, and chat history:

        {context}

        {memory_context}

        Please answer the following question about the data:
        {user_question}

        Important instructions:
        1. If the question requires calculations or data manipulation, perform them directly and provide the specific numerical result.
        2. Use the full dataset provided in the context to answer questions, not just the summary statistics.
        3. If referring to a specific file, mention which file you're using in your answer.
        4. You can use previous analyses, insights, and generated graphs if they are relevant to the question.
        5. Provide concise and informative answers based on the available information.
        6. If you need to perform any Python operations to answer the question, you can use pandas functions directly on the dataframes provided in the context.

        Remember, you have full access to the data and can perform any necessary calculations. Always strive to give precise, numerical answers when appropriate."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful data analyst assistant with direct access to pandas dataframes."},
                    {"role": "user", "content": prompt}
                ]
            )
            answer = response.choices[0].message.content
            self.add_to_chat_history("user", user_question)
            self.add_to_chat_history("assistant", answer)
            return answer
        except Exception as e:
            return f"Error chatting with data: {str(e)}"

def home():
    st.title("Welcome to Data Analytics Tool!")
    st.write("""
    This powerful tool is designed to help you analyze and gain insights from your data effortlessly. 
    With our Data Analytics Tool, you can:

    - Import and manage multiple datasets
    - Perform various types of graph analysis
    - Generate textual insights and cross-file analysis
    - Chat with an AI assistant about your data

    To get started, use the menu on the left to navigate through different functionalities.
    """)

def import_data(chatbot):
    st.title("Import Data")
    
    # File uploader
    uploaded_files = st.file_uploader("Choose CSV or Excel files", type=["csv", "xlsx", "xls"], accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            result = chatbot.load_file(file)
            st.write(result)
    
    # Display and manage imported files
    st.subheader("Imported Files")
    if chatbot.dataframes:
        for file_name in chatbot.dataframes.keys():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"• {file_name}")
            with col2:
                if st.button(f"Delete {file_name}"):
                    del chatbot.dataframes[file_name]
                    st.success(f"{file_name} has been deleted.")
                    st.experimental_rerun()
    else:
        st.write("No files imported yet.")

def graph_analysis(chatbot):
    st.title("Graph Analysis")
    if not chatbot.dataframes:
        st.warning("Please import some data first.")
        return

    file_name = st.selectbox("Select file for analysis", list(chatbot.dataframes.keys()))
    analysis_type = st.selectbox("Select Analysis Type", ["Histogram", "Scatter Plot", "Correlation Analysis", "Principal Component Analysis", "K-means Clustering"])

    if analysis_type == "Histogram":
        column = st.selectbox("Select column for histogram", chatbot.dataframes[file_name].columns)
        if st.button("Generate Histogram"):
            fig = chatbot.create_histogram(file_name, column)
            st.plotly_chart(fig)

    elif analysis_type == "Scatter Plot":
        x_column = st.selectbox("Select X-axis column", chatbot.dataframes[file_name].columns)
        y_column = st.selectbox("Select Y-axis column", chatbot.dataframes[file_name].columns)
        if st.button("Generate Scatter Plot"):
            fig = chatbot.create_scatter_plot(file_name, x_column, y_column)
            st.plotly_chart(fig)

    elif analysis_type == "Correlation Analysis":
        if st.button("Perform Correlation Analysis"):
            fig, corr_matrix = chatbot.perform_correlation_analysis(file_name)
            st.plotly_chart(fig)
            st.text(corr_matrix)

    elif analysis_type == "Principal Component Analysis":
        if st.button("Perform PCA"):
            fig, explanation = chatbot.perform_pca(file_name)
            st.plotly_chart(fig)
            st.write(explanation)

    elif analysis_type == "K-means Clustering":
        n_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=3)
        if st.button("Perform Clustering"):
            fig, explanation = chatbot.perform_clustering(file_name, n_clusters)
            st.plotly_chart(fig)
            st.write(explanation)

def textual_analysis(chatbot):
    st.title("Textual Analysis and Insights")
    if not chatbot.dataframes:
        st.warning("Please import some data first.")
        return

    analysis_type = st.selectbox("Select Analysis Type", ["Basic Information", "Generate Insights", "Generate Cross-file Insights"])

    if analysis_type == "Basic Information":
        file_name = st.selectbox("Select file", list(chatbot.dataframes.keys()))
        info = chatbot.get_basic_info(file_name)
        st.text(info)

    elif analysis_type == "Generate Insights":
        file_name = st.selectbox("Select file", list(chatbot.dataframes.keys()))
        if st.button("Generate Insights"):
            insights = chatbot.generate_insights(file_name)
            st.write(insights)

    elif analysis_type == "Generate Cross-file Insights":
        available_files = list(chatbot.dataframes.keys())
        file_options = ["All Files"] + available_files
        selected_files = st.multiselect("Select files for cross-file analysis", file_options, default=["All Files"])
        
        if "All Files" in selected_files:
            selected_files = available_files

        if st.button("Generate Cross-file Insights"):
            if len(selected_files) < 2:
                st.warning("Please select at least two files for cross-file analysis.")
            else:
                insights = chatbot.generate_cross_file_insights(selected_files)
                st.write(insights)

def chatbot_interface(chatbot):
    st.title("Chat with Data")
    if not chatbot.dataframes:
        st.warning("Please import some data first.")
        return

    file_options = list(chatbot.dataframes.keys()) + ["All Files"]
    selected_files = st.multiselect("Select files to analyze", file_options, default=["All Files"])
    
    if "All Files" in selected_files:
        selected_files = list(chatbot.dataframes.keys())
    
    # Display chat history
    st.subheader("Chat History")
    for message in chatbot.chat_history:
        if message['role'] == 'user':
            st.text_area("You:", value=message['content'], height=100, key=f"user_{chatbot.chat_history.index(message)}")
        else:
            st.info(message['content'])
    
    # New message input
    user_question = st.text_input("Ask a question about the selected data:")
    if st.button("Send"):
        # Display user message
        st.text_area("You:", value=user_question, height=100, key="user_new")
        
        # Get and display assistant response
        answer = chatbot.chat_with_data(selected_files, user_question)
        st.info(answer)
        
        # Update chat history in session state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        
        # Clear the input box and refresh
        st.experimental_rerun()

def main():
    st.set_page_config(layout="wide")

    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = DataAnalysisChatbot()
    chatbot = st.session_state.chatbot

    with st.sidebar:
        selected = option_menu(
            menu_title="Main Menu",
            options=["Home", "Import Data", "Graph Analysis", "Textual Analysis", "Chatbot"],
            icons=["house", "cloud-upload", "graph-up", "file-text", "chat-dots"],
            menu_icon="yin-yang",
            default_index=0,
        )

    if selected == "Home":
        home()
    elif selected == "Import Data":
        import_data(chatbot)
    elif selected == "Graph Analysis":
        graph_analysis(chatbot)
    elif selected == "Textual Analysis":
        textual_analysis(chatbot)
    elif selected == "Chatbot":
        chatbot_interface(chatbot)

if __name__ == "__main__":
    main()