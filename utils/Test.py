import os
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI

class IntelligentCSVAgent:
    def __init__(self, api_key):
        self.api_key = api_key
        self.llm = ChatOpenAI(openai_api_key=self.api_key, temperature=0)
        self.dataframes = {}
        self.merged_dataframe = None
        self.agent = None

    def load_csv_files(self, folder_path):
        """Load multiple CSV files into dataframes."""
        for file in os.listdir(folder_path):
            if file.endswith('.csv'):
                file_path = os.path.join(folder_path, file)
                df = pd.read_csv(file_path)
                self.dataframes[file] = df
        print(f"Loaded {len(self.dataframes)} CSV files.")

    def process_query(self, query):
        """Decide how to handle the query using the LLM."""
        # Prepare schema details for the LLM
        schemas = {name: list(df.columns) for name, df in self.dataframes.items()}
        
        prompt = f"""
        I have multiple datasets with the following schemas:
        {schemas}

        The user wants to ask: "{query}".
        
        Decide the following:
        1. Should I merge any of these tables to answer the query? If yes, specify which tables, the keys to merge on, and the method (e.g., inner, outer, etc.).
        2. If merging is not required, decide which single table to use to answer the query.
        3. Generate a plan for how to answer the query using the chosen table(s).
        Respond in a structured JSON format with keys: "merge_required", "merge_details", "chosen_table", "query_plan".
        """
        response = self.llm.predict(prompt)
        return response

    def merge_tables(self, merge_details):
        """Merge tables based on LLM instructions."""
        try:
            merge_plan = eval(merge_details)  # Parse LLM response
            tables_to_merge = merge_plan.get("tables", [])
            keys = merge_plan.get("keys", [])
            method = merge_plan.get("method", "outer")
            
            if len(tables_to_merge) < 2:
                print("Not enough tables to merge.")
                return None
            
            # Perform merging
            df_to_merge = [self.dataframes[table] for table in tables_to_merge]
            merged_df = df_to_merge[0]
            for df in df_to_merge[1:]:
                merged_df = pd.merge(merged_df, df, how=method, on=keys)
            
            print(f"Successfully merged tables: {tables_to_merge}")
            return merged_df
        except Exception as e:
            print(f"Error during merging: {e}")
            return None

    def answer_query(self, query):
        """Answer queries based on LLM's decision."""
        # Get LLM's decision
        llm_response = self.process_query(query)
        print("LLM Decision:")
        print(llm_response)
        
        # Parse the response
        decision = eval(llm_response)  # Ensure response is structured JSON
        merge_required = decision.get("merge_required", False)
        chosen_table = decision.get("chosen_table", None)
        query_plan = decision.get("query_plan", "")
        
        # Handle merging if required
        if merge_required:
            self.merged_dataframe = self.merge_tables(decision.get("merge_details", {}))
            df_to_query = self.merged_dataframe
        else:
            df_to_query = self.dataframes.get(chosen_table, None)
        
        # Check if the chosen table is available
        if df_to_query is None:
            print("Error: No valid table available for querying.")
            return None

        # Initialize the LangChain Pandas Agent
        self.agent = create_pandas_dataframe_agent(self.llm, df_to_query, verbose=True)

        # Answer the query using the agent
        result = self.agent.run(query_plan)
        return result


# Example Usage
if __name__ == "__main__":
    # Initialize the agent with your OpenAI API key
    api_key = "your_openai_api_key"
    agent = IntelligentCSVAgent(api_key)
    
    # Load CSV files from a folder
    agent.load_csv_files("path_to_your_csv_folder")
    
    # Query the agent
    result = agent.answer_query("What is the total revenue for 2024?")
    print("Query Result:", result)
