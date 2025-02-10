from autogen import UserProxyAgent, ConversableAgent, GroupChat, GroupChatManager
import custom_llm  # Ensure custom_llm() is implemented

# ✅ Custom LLM Wrapper
class LLMWrapper:
    def __init__(self):
        self.llm = custom_llm()  # Your custom LLM instance

    def invoke(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return response.strip() if response.strip() else "No valid response generated."

# ✅ Custom Agent with `register_reply`
class CustomLLMAgent(ConversableAgent):
    def __init__(self, name, system_message):
        super().__init__(name=name, system_message=system_message)
        self.llm_wrapper = LLMWrapper()
        self.register_reply(self.generate_reply)  # ✅ Register reply function

    def generate_reply(self, messages, sender):
        """Automatically generates replies using the LLM"""
        if not messages:
            return [{"role": "assistant", "content": "No input provided."}]

        prompt = messages[-1]["content"]  
        response = self.llm_wrapper.invoke(prompt)

        # 🔹 Strict Rules to Prevent Incorrect Outputs
        if self.name == "Summarizer" and "```python" in response:
            response = "Error: You are not allowed to generate code. Summarize only."

        if self.name == "Coder" and not response.startswith("```python"):
            response = "Error: Only return valid Python code."

        return [{"role": "assistant", "content": response}]

# ✅ System Messages for Agents
summarizer_prompt = """You are a requirement analyst. Extract clear requirements from the user's request.
DO NOT generate any code. Format:
- Task: Describe what needs to be implemented.
- Input: List input types.
- Output: Expected result.
- Constraints: Performance considerations."""

coder_prompt = """You are an expert Python developer. Generate only Python code. DO NOT explain or modify requirements.
Return only Python code like this:
```python
# Your code here
```"""

validator_prompt = """You are a senior software engineer. Validate the code for correctness and efficiency.
- If correct, respond with: "✅ Code is correct."
- If incorrect, explain the issues and request improvements."""

final_judge_prompt = """You are the final reviewer.
- If Validator approves the code, finalize the response.
- If Validator requests changes, return feedback to Coder.
"""

# ✅ Create Agents
summarizer_agent = CustomLLMAgent(name="Summarizer", system_message=summarizer_prompt)
coder_agent = CustomLLMAgent(name="Coder", system_message=coder_prompt)
validator_agent = CustomLLMAgent(name="Validator", system_message=validator_prompt)
final_judge_agent = CustomLLMAgent(name="FinalJudge", system_message=final_judge_prompt)

# ✅ User Proxy Agent (Entry Point)
user_proxy = UserProxyAgent(
    name="User",
    human_input_mode="ALWAYS",
    max_consecutive_auto_reply=10
)

# ✅ Group Chat Setup
group_chat = GroupChat(
    agents=[user_proxy, summarizer_agent, coder_agent, validator_agent, final_judge_agent],
    messages=[]
)

# ✅ Group Chat Manager to Control the Conversation
controller = GroupChatManager(groupchat=group_chat)

# ✅ Function to Run the Full Workflow
def agentic_code_generation(user_query):
    if not user_query.strip():
        return "Error: Empty user query."

    response = controller.initiate_chat(user_proxy, message=user_query)
    return response  # ✅ Final validated output

# ✅ Example Call
if __name__ == "__main__":
    user_query = "Write a Python function to calculate Fibonacci numbers."
    final_code = agentic_code_generation(user_query)
    print("Final Validated Code:\n", final_code)
