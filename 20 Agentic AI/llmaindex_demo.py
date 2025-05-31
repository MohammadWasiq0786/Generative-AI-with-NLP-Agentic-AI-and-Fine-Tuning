from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.callbacks import CallbackManager
from typing import Any, List
from euri_llm import euri_completion
import wikipedia
import sqlite3

# ✅ Metadata required by LlamaIndex
class LLMMetadata:
    def __init__(self):
        self.context_window = 4096
        self.num_output = 1000
        self.model_name = "gpt-4.1-nano"
        self.is_chat_model = True

class ChatMessageWrapper:
    def __init__(self, content):
        self.message = type("Msg", (), {"content": content})

class EuriaiLLM:
    def __init__(self):
        self._metadata = LLMMetadata()
        self._callback_manager = CallbackManager([])

    def complete(self, prompt: str, **kwargs) -> str:
        return euri_completion([{"role": "user", "content": prompt}])

    def chat(self, messages: List[Any], **kwargs) -> Any:
        system_prompt = {
            "role": "system",
            "content": (
                "You are an AI agent. Use this format to reason:\n"
                "Thought: ...\n"
                "Action: tool_name\n"
                "Action Input: {\"input\": \"...\"}\n"
                "Observation: ...\n"
                "Answer: ...\n"
                "Tools available: to_uppercase, calculator, wikipedia_search, sql_executor"
            )
        }
        msg_payload = [system_prompt] + [{"role": m.role, "content": m.content} for m in messages]
        response_text = euri_completion(msg_payload)
        return ChatMessageWrapper(response_text)


    @property
    def metadata(self) -> Any:
        return self._metadata

    @property
    def callback_manager(self):
        return self._callback_manager

    @callback_manager.setter
    def callback_manager(self, value):
        self._callback_manager = value
        
def to_uppercase(input: str) -> str:
    return input.upper()
        
def wikipedia_search(input: str) -> str:
    try:
        summary = wikipedia.summary(input, sentences=2)
        return f"Here's a brief summary of {input}:\n{summary}"
    except Exception as e:
        return f"Error from Wikipedia: {str(e)}"
    
def calculator(input: str) -> str:
    try:
        result = eval(input, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error in calculation: {str(e)}"
    
tools = [FunctionTool.from_defaults(fn = to_uppercase),
 FunctionTool.from_defaults(fn = wikipedia_search),
 FunctionTool.from_defaults(fn = calculator)]


agent = ReActAgent.from_tools(
    tools,llm = EuriaiLLM(),verbose = True ,tool_choice_mode = "always" 
)
agent.chat("convert my name is sudhanshu to uppercase")
agent.chat("give me a 3*4")