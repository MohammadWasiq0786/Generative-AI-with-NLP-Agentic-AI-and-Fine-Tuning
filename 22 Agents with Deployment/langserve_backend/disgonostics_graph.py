from typing import TypedDict
from langgraph.graph import StateGraph,END
from langchain_core.runnables import RunnableLambda
from tools.diagnosis_tool import ai_diagnos
from tools.symptom_checker import check_symptom


class DiagnosticState(TypedDict):
    input:str
    symptom_area : str
    diagnosis: str
    
def build_graph():
    graph = StateGraph(DiagnosticState)
    
    def symptom_step(state):
        return {
                "input":state["input"],
                "symptom_area" : check_symptom.invoke(state["input"]),
                "diagnosis": state.get("diagnosis","")
        }
    
    graph.add_node("symptomcheck",RunnableLambda(symptom_step))
    
    
    def diagnosis_step(state):
        return {
                "input":state["input"],
                "symptom_area" : state['symptom_area'],
                "diagnosis":ai_diagnos.invoke(state['input'])
        }
        
        
    graph.add_node("Aidiagnosis" , RunnableLambda(diagnosis_step))
    
    
    graph.set_entry_point("symptomcheck")
    graph.add_edge("symptomcheck","Aidiagnosis")
    graph.add_edge("Aidiagnosis",END)
    
    return graph.compile()

