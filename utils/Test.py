import os
import pandas as pd
from joblib import Parallel, delayed
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI

class FastCSVAgent:
    def __init__(self, api_key):
        self.api_key = api_key
        self.llm = ChatOpenAI(openai_api_key=self.api_key, temperature=0)
        self.dataframes = {}
        self.merged_dataframe = None
        self.agent = None

    def load_csv_files(self, folder_path):
        """Load CSV files in parallel."""
        def load_csv(file):
            return pd.read_csv(os.path.join(folder_path, file))
        
        files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
        self.dataframes = dict(zip(files, Parallel(n_jobs=-1)(delayed(load_csv)(file) for file in files)))
        print(f"Loaded {len(self.dataframes)} CSV files.")

    def decide_query_action(self, query):
        """Decide if merging is needed or which table to use."""
        schemas = {name: list(df.columns) for name, df in self.dataframes.items()}
        prompt = f"""
        I have these schemas: {schemas}.
        Query: "{query}"
        Decide:
        1. Should I merge tables? If yes, provide details (tables, keys, method).
        2. If no merge is needed, suggest the best table to use.
        """
        response = self.llm.predict(prompt)
        return eval(response)

    def merge_tables(self, merge_details):
        """Merge tables based on LLM instructions."""
        try:
            tables = merge_details['tables']
            keys = merge_details.get('keys', [])
            method = merge_details.get('method', 'outer')
            dfs_to_merge = [self.dataframes[table] for table in tables]
            
            merged_df = dfs_to_merge[0]
            for df in dfs_to_merge[1:]:
                merged_df = pd.merge(merged_df, df, how=method, on=keys)
            
            return merged_df
        except Exception as e:
            print(f"Error during merging: {e}")
            return None

    def answer_query(self, query):
        """Answer queries efficiently."""
        decision = self.decide_query_action(query)
        
        if decision.get('merge_required', False):
            self.merged_dataframe = self.merge_tables(decision['merge_details'])
            df_to_query = self.merged_dataframe
        else:
            df_to_query = self.dataframes.get(decision['chosen_table'])
        
        if df_to_query is None:
            print("Error: No valid table to query.")
            return None
        
        # Use LangChain Pandas Agent
        self.agent = create_pandas_dataframe_agent(self.llm, df_to_query, verbose=True)
        result = self.agent.run(query)
        return result
