from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool

# Define tools
def tool_a(inputs):
    return f"ToolA executed with inputs: {inputs}"

def tool_b(inputs):
    return f"ToolB executed with inputs: {inputs}"

def tool_c(inputs):
    return f"ToolC executed with inputs: {inputs}"

tools = [
    Tool(name="ToolA", func=tool_a, description="Requires name and city."),
    Tool(name="ToolB", func=tool_b, description="Requires age."),
    Tool(name="ToolC", func=tool_c, description="Requires name, age, and city."),
]

# Initialize LLM
llm = ChatOpenAI(temperature=0)

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define the prompt for dynamic questioning and tool execution
prompt_template = """
You are a helpful AI that needs to collect user information before calling tools.
Follow these rules:
1. Ask one question at a time.
2. Allow the user to modify their previous answers at any point.
3. Once all required information is collected, call the tools in the correct order.

Available tools:
- ToolA requires name and city.
- ToolB requires age.
- ToolC requires name, age, and city.

Ask relevant questions first. Ensure all required answers are collected before calling the tools.

Memory so far: {chat_history}

Now, ask the next relevant question or call the tools if all data is available.
"""

# Initialize agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    agent_kwargs={"system_message": prompt_template},
)

# Run chat loop
if __name__ == "__main__":
    print("AI Chatbot Started. Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = agent.run(user_input)
        print(f"Bot: {response}")
