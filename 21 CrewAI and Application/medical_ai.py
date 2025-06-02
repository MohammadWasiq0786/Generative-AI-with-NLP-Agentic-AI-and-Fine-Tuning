from crewai import Agent, Task , Crew 
from crewai_tools.tools import tool
from euri_llm_crew import EuriLLM


@tool("symptoms checker tool")
def symptom_checker_tool(symptoms_text):
    """this tool analyzes the patient symptoms and provide a diagonoisis in details """
    
    if "thirst" in symptoms_text and "fatigue" in symptoms_text : 
        return "possible condition : type 2 diabetes . consider blood suger evaluation "
    elif "headache" in symptoms_text :
        return "possible condition : migrain or hypertension .need clinical confirmation" 
    else :
        return   "generatl fatigue ;could be realted to stress or poor sleep"


@tool("health advice tool")
def health_advice_tool(profile_text):
    """provides basicv wellness suggestion based on symptoms and history """
    return "maintain a low-sugar diete , hyderate well , and schedule a doctor visit for further advice"


class MedicalAgent:
    def __init__(self,role,goal , backstory,tools=[]):
        self.agent = Agent(
            role = role,
            goal = goal,
            backstory= backstory,
            verbose = True , 
            allow_deligation = False ,
            llm = EuriLLM(),
            tools = tools
        )

    def get_agent(self):
        return self.agent
    
    
user_input = {
    "name" :"sudhanshu kumar",
    "age" : 33,
    "gender" :"male",
    "symptoms":["frequent headache","blurred vision","fatigue","increased thirst"],
    "medical history":"family history of diabetese and hypertension"
}    

symptoms_text = f"""
name : {user_input['name']}
age : {user_input['age']}
gender:{user_input['gender']}
symptoms:{user_input['symptoms']}
medical history : {user_input['medical history']}
"""

diagnosis_aget = MedicalAgent(
        role="Ai medical Diagnostician",
        goal="analyuze sumptoms and provide possible cure",
        backstory= " expert in identifying potentical health condition based on the suysmtoms cluster and identification",
        tools=[symptom_checker_tool]).get_agent()

advice_agent = MedicalAgent(
        role="Ai healthcare adviser",
        goal="offer the safe and respoinsible lifestyle suggestion based on the user profile ",
        backstory="a virtual assistant trained to give personalize wellness and precaution advice",
        tools=[health_advice_tool]
    ).get_agent()

tasks = [
    Task(
        description=f"Analyze the following user's symptoms using tools and provide a possible cause:\n{symptoms_text}",
        expected_output="Top 3 possible conditions with a brief explanation for each listed condition.",
        agent=diagnosis_aget
    ),
    Task(
        description=f"Based on the user's profile and symptoms, provide lifestyle and medical advice:\n{symptoms_text}",
        expected_output="Actionable advice, dietary tips, and when to seek or visit a doctor.",
        agent=advice_agent
    ),
]



crew = Crew(
        agents = [diagnosis_aget,advice_agent],
        tasks = tasks,
        verbose = True 
    )


result  = crew.kickoff()

print("personalize medical report ", result)