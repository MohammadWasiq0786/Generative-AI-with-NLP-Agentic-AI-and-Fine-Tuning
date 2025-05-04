import os

# Define the folder structure
folders = [
    "knowledge_graph_builder",
    "knowledge_graph_builder/agents",
    "knowledge_graph_builder/tools",
    "knowledge_graph_builder/workflows",
    "knowledge_graph_builder/utils",
    "knowledge_graph_builder/data",
    "knowledge_graph_builder/data/outputs"
]

# Define the files to be created in each folder
files = {
    "knowledge_graph_builder/app.py": "",
    "knowledge_graph_builder/agents/researcher.py": "",
    "knowledge_graph_builder/agents/synthesizer.py": "",
    "knowledge_graph_builder/agents/mapper.py": "",
    "knowledge_graph_builder/tools/serpapi_tool.py": "",
    "knowledge_graph_builder/tools/wikipedia_tool.py": "",
    "knowledge_graph_builder/tools/research_api_tool.py": "",
    "knowledge_graph_builder/workflows/langgraph_router.py": "",
    "knowledge_graph_builder/utils/graphviz_exporter.py": "",
    "knowledge_graph_builder/utils/config.py": "",
    "knowledge_graph_builder/requirements.txt": ""
}

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create files
for filepath, content in files.items():
    with open(filepath, "w") as f:
        f.write(content)

print("Folder structure created successfully.")
