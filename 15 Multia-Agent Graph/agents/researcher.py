from tools.serpapi_tool import search_google
from tools.wikipedia_tool import search_wikipedia

def run_researcher(topic):
    google_results = search_google(topic)
    wiki_summary = search_wikipedia(topic)
    return {
        "google": google_results,
        "wiki": wiki_summary
    }
