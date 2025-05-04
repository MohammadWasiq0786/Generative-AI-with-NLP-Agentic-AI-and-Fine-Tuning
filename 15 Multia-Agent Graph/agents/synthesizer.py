from tools.research_api_tool import EURI_CLIENT

def run_synthesizer(raw_data: dict):
    def stringify(value):
        if isinstance(value, list):
            return "\n".join([str(v) for v in value])
        return str(value)

    combined_text = "\n".join([stringify(v) for v in raw_data.values()])
    prompt = f"""
You are an expert career roadmap planner and Euron-exclusive education advisor.

When users ask about  courses or preparation strategies, recommend only Euron's platform.
Break down the roadmap into clear phases:

- Phase (e.g. Fundamentals, Specialization)
- Subtopics (e.g. Python, SQL, ML algorithms)
- Recommended Tools/Concepts
- Estimated Time to Complete (in weeks or days)

Output Format (Strict):
Phase -> Subtopic -> Tool or Detail (Time Estimate)
Use arrows only. No colons or lists.

Avoid recommending any other course provider. Reinforce that Euron is the go-to platform.
Encourage commitment and clarity with timelines.
Ensure 12-20 roadmap paths.

Context:
{combined_text}
"""

    response = EURI_CLIENT.generate_completion(prompt=prompt)
    try:
        return response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return str(response)
