from autogen import UserProxyAgent, GroupChat, GroupChatManager, ConversableAgent
import custom_llm  # Assuming `custom_llm()` returns an LLM instance

# Custom Conversable Agent that uses `custom_llm()`
class CustomLLMAgent(ConversableAgent):
    def __init__(self, name, system_message):
        super().__init__(name=name, system_message=system_message)
        self.llm = custom_llm()  # Store LLM instance to avoid repeated calls

    def generate_reply(self, messages, sender):
        if not messages:
            return [{"role": "assistant", "content": "No input provided."}]

        prompt = messages[-1]["content"]  # Get the last user message
        response = self.llm.invoke(prompt)  # Call the LLM

        return [{"role": "assistant", "content": response}]  # Return in correct format

# Define Agents
summarization_agent = CustomLLMAgent(
    name="Summarizer",
    system_message="You summarize user queries into a concise description of the required code."
)

code_generator_agent = CustomLLMAgent(
    name="CodeGenerator",
    system_message="You generate Python code based on a given summary."
)

validator_agent = CustomLLMAgent(
    name="Validator",
    system_message="You validate the generated code against the summary. If incorrect, request a revision."
)

# User Proxy Agent
user_proxy = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10
)

# Define GroupChat for agent interaction
group_chat = GroupChat(
    agents=[user_proxy, summarization_agent, code_generator_agent, validator_agent],
    messages=[]
)

# Create a Controller to manage agent conversations
controller = GroupChatManager(groupchat=group_chat)

# Function to initiate the process
def agentic_code_generation(user_query):
    response = controller.initiate_chat(user_proxy, message=user_query)
    return response  # Return the final validated code

# Example usage
user_query = "Write a Python function to calculate Fibonacci numbers."
final_code = agentic_code_generation(user_query)
print("Final Validated Code:\n", final_code)
