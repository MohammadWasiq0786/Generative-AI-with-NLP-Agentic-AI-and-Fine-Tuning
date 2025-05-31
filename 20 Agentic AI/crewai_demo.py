from crewai import Agent, Task, Crew
from langchain_core.language_models.llms import LLM
from langchain_core.runnables import RunnableLambda
from typing import Optional, List, Union
from euri_llm import euri_completion

# ✅ Version-safe LLM wrapper
class EuriLLMWrapper(LLM):
    def _call(self, prompt: Union[str, object], stop: Optional[List[str]] = None) -> str:
        # Safely convert any object with `.to_string()` or `.text` to string
        if hasattr(prompt, "to_string"):
            prompt = prompt.to_string()
        elif hasattr(prompt, "text"):
            prompt = prompt.text
        elif not isinstance(prompt, str):
            prompt = str(prompt)
        return euri_completion([{"role": "user", "content": prompt}])

    @property
    def _llm_type(self) -> str:
        return "euri-llm"

    def bind(self, **kwargs):
        return RunnableLambda(lambda x: self._call(x["input"] if isinstance(x, dict) else x))

# ✅ Instantiate your LLM
llm = EuriLLMWrapper()


researcher = Agent(
    role="AI Researcher",
    goal="Gather detailed facts about AI trends",
    backstory="Expert in tech research",
    verbose=True,
    llm=llm
)

writer = Agent(
    role="Technical Writer",
    goal="Write professional articles based on research",
    backstory="Experienced in writing AI blog posts",
    verbose=True,
    llm=llm
)

task1 = Task(
    description="Research on future of generative AI",
    expected_output="A detailed summary of the current and future trends in generative AI.",
    agent=researcher
)

task2 = Task(
    description="Write an article based on the research",
    expected_output="A well-written technical article ready for publication.",
    agent=writer
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    verbose=2
)


retult = crew.kickoff()
print(retult)
