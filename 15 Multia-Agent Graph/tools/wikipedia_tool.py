import requests
WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
def search_wikipedia(query):
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "titles": query
    }
    response = requests.get(WIKI_API_URL, params=params).json()
    page = next(iter(response["query"]["pages"].values()))
    return page.get("extract", "")