import json

def run_mapper(synthesized_text):
    lines = synthesized_text.split("\n")
    nodes = []
    edges = []
    added_nodes = set()
    for line in lines:
        if "->" in line:
            parts = [part.strip() for part in line.split("->")]
            for part in parts:
                if part and part not in added_nodes:
                    nodes.append({"name": part})
                    added_nodes.add(part)
            for i in range(len(parts) - 1):
                edges.append({"source": parts[i], "target": parts[i + 1]})
    return {"nodes": nodes, "edges": edges}
