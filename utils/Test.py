from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager, ConversableAgent
import custom_llm  # Assuming `custom_llm()` returns an instance of an LLM

# Custom Conversable Agent that uses custom_llm()
class CustomLLMAgent(ConversableAgent):
    def generate_reply(self, messages, sender):
        prompt = messages[-1]["content"]  # Get last message as input
        response = custom_llm().invoke(prompt)  # Use custom_llm() for response
        return response

# Define Agents
summarization_agent = CustomLLMAgent(
    name="Summarizer",
    system_message="Convert user queries into a concise description of the required code."
)

code_generator_agent = CustomLLMAgent(
    name="CodeGenerator",
    system_message="Generate Python code based on a given summary."
)

validator_agent = CustomLLMAgent(
    name="Validator",
    system_message="Validate the code against the summary. If incorrect, request a revision."
)

# User Proxy
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
    return response  # Final validated code

# Example usage
user_query = "Write a Python function to calculate Fibonacci numbers."
final_code = agentic_code_generation(user_query)
print("Final Validated Code:\n", final_code)
