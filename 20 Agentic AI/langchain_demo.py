from langchain.agents import create_react_agent, AgentExecutor
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain.llms.base import LLM
from euri_llm import euri_completion

@tool
def expert_wrtier(input):
    """thi sis my expert writer """
    message = [{"role":"user","content" : f"write a sort poen on {input}"}]
    return euri_completion(messages=message)

@tool
def expert_math(input):
    """this is my expert math tools """
    result = eval(input,{"__builtins__":{}},{})
    return result

tools = [expert_wrtier,expert_math]

class EuriaiLLM(LLM):
    def _call(self, prompt, stop=None):
        return euri_completion([{"role": "user", "content": prompt}])

    @property
    def _llm_type(self):
        return "euri-llm"


prompt = PromptTemplate.from_template(
    """You are an intelligent agent with access to tools: {tool_names}

{tools}

Use this format strictly:
Thought: describe what you want to do
Action: the tool to use, one of [{tool_names}]
Action Input: the input in JSON format, e.g., {{"input": "2+2"}}
Observation: result of the action
... (repeat Thought/Action/Observation if needed)
Thought: I now know the final answer
Final Answer: the final response to the user

Begin!

Question: {input}
{agent_scratchpad}"""
)

agent  = create_react_agent(
    llm = EuriaiLLM(),
    tools = tools,
    prompt = prompt,
    output_parser = ReActSingleInputOutputParser()
)

executor = AgentExecutor(
    agent = agent,
    tools = tools,
    verbose = True,
    handle_poarsing_error= True
)

response = executor.invoke({"input" : "give me a poem based on earth and try to execure 4*6"})

print(response['output'])