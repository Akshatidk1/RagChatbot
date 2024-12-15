024-12-15 21:57:44.216 Uncaught app execution
Traceback (most recent call last):
  File "C:\Users\UZ731DM\OneDrive - EY\Documents\log_price_forecast\myevn\Lib\site-packages\streamlit\runtime\scriptrunner\exec_code.py", line 88, in exec_func_with_error_handling
    result = func()
             ^^^^^^
  File "C:\Users\UZ731DM\OneDrive - EY\Documents\log_price_forecast\myevn\Lib\site-packages\streamlit\runtime\scriptrunner\script_runner.py", line 579, in code_to_exec
    exec(code, module.__dict__)
  File "C:\Users\UZ731DM\OneDrive - EY\Documents\log_price_forecast\app.py", line 23, in <module>
    st.session_state.agents = initialize_agents(
                              ^^^^^^^^^^^^^^^^^^
  File "C:\Users\UZ731DM\OneDrive - EY\Documents\log_price_forecast\agents.py", line 19, in initialize_agents
    pandas_agent = create_pandas_dataframe_agent(model, data, verbose=True, memory=memory)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\UZ731DM\OneDrive - EY\Documents\log_price_forecast\myevn\Lib\site-packages\langchain_experimental\agents\agent_toolkits\pandas\base.py", line 249, in create_pandas_dataframe_agent
    raise ValueError(
ValueError: This agent relies on access to a python repl tool which can execute arbitrary code. This can be dangerous and requires a specially sandboxed environment to be safely used. Please read the security notice in the doc-string of this function. You must opt-in to use this functionality by setting allow_dangerous_code=True.For general security guidelines, please see: https://python.langchain.com/docs/security/ 
C:\Users\UZ731DM\OneDrive - EY\Documents\log_price_forecast\myevn\Lib\site-packages\langchain_experimental\agents\agent_toolkits\pandas\base.py:283: UserWarning: Received additional kwargs {'memory': ConversationBufferMemory(chat_memory=InMemoryChatMessageHistory(messages=[]), return_messages=True, memory_key='chat_history')} which are no longer supported.
  warnings.warn(
C:\Users\UZ731DM\OneDrive - EY\Documents\log_price_forecast\agents.py:33: LangChainDeprecationWarning: LangChain agents will continue to be supported, but it is recommended for new use cases to be built with LangGraph. LangGraph offers a more flexible and full-featured framework for building agents, including support for tool-calling, persistence of state, and human-in-the-loop workflows. See LangGraph documentation for more details: https://langchain-ai.github.io/langgraph/. Refer here for its pre-built ReAct agent: https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/
  return initialize_agent(tools, model, agent="zero-shot-react-description", verbose=True, prompt=chat_prompt_template)
2024-12-15 21:58:24.808 Uncaught app execution
Traceback (most recent call last):
  File "C:\Users\UZ731DM\OneDrive - EY\Documents\log_price_forecast\myevn\Lib\site-packages\streamlit\runtime\scriptrunner\exec_code.py", line 88, in exec_func_with_error_handling
    result = func()
             ^^^^^^
  File "C:\Users\UZ731DM\OneDrive - EY\Documents\log_price_forecast\myevn\Lib\site-packages\streamlit\runtime\scriptrunner\script_runner.py", line 579, in code_to_exec
    exec(code, module.__dict__)
  File "C:\Users\UZ731DM\OneDrive - EY\Documents\log_price_forecast\app.py", line 23, in <module>
    st.session_state.agents = initialize_agents(
                              ^^^^^^^^^^^^^^^^^^
  File "C:\Users\UZ731DM\OneDrive - EY\Documents\log_price_forecast\agents.py", line 33, in initialize_agents
    return initialize_agent(tools, model, agent="zero-shot-react-description", verbose=True, prompt=chat_prompt_template)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\UZ731DM\OneDrive - EY\Documents\log_price_forecast\myevn\Lib\site-packages\langchain_core\_api\deprecation.py", line 182, in warning_emitting_wrapper
    return wrapped(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\UZ731DM\OneDrive - EY\Documents\log_price_forecast\myevn\Lib\site-packages\langchain\agents\initialize.py", line 73, in initialize_agent
    agent_obj = agent_cls.from_llm_and_tools(
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\UZ731DM\OneDrive - EY\Documents\log_price_forecast\myevn\Lib\site-packages\langchain\agents\mrkl\base.py", line 138, in from_llm_and_tools
    cls._validate_tools(tools)
  File "C:\Users\UZ731DM\OneDrive - EY\Documents\log_price_forecast\myevn\Lib\site-packages\langchain\agents\mrkl\base.py", line 162, in _validate_tools
    validate_tools_single_input(cls.__name__, tools)
  File "C:\Users\UZ731DM\OneDrive - EY\Documents\log_price_forecast\myevn\Lib\site-packages\langchain\agents\utils.py", line 17, in validate_tools_single_input
    if not tool.is_single_input:
           ^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\UZ731DM\OneDrive - EY\Documents\log_price_forecast\myevn\Lib\site-packages\pydantic\main.py", line 856, in __getattr__     
    raise AttributeError(f'{type(self).__name__!r} object has no attribute {item!r}')
AttributeError: 'AgentExecutor' object has no attribute 'is_single_input'
