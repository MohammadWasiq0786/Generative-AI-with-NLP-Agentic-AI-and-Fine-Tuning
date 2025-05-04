from langgraph.graph import StateGraph
from typing import TypedDict

from agents.researcher import run_researcher
from agents.synthesizer import run_synthesizer
from agents.mapper import run_mapper

class GraphState(TypedDict):
    input: str
    raw_info: dict
    summary: str
    result: dict

def build_langgraph():
    graph = StateGraph(GraphState)

    def wrapped_researcher(state: GraphState):
        return {"raw_info": run_researcher(state["input"])}

    def wrapped_synthesizer(state: GraphState):
        return {"summary": run_synthesizer(state["raw_info"])}

    def wrapped_mapper(state: GraphState):
        return {"result": run_mapper(state["summary"])}

    graph.add_node("research", wrapped_researcher)
    graph.add_node("synthesize", wrapped_synthesizer)
    graph.add_node("map", wrapped_mapper)

    graph.set_entry_point("research")
    graph.add_edge("research", "synthesize")
    graph.add_edge("synthesize", "map")

    return graph.compile()

def autonomous_pipeline(topic):
    langgraph_flow = build_langgraph()
    result = langgraph_flow.invoke({"input": topic})
    return result["result"]
