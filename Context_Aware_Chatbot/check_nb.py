import json
import sys

def summarize_nb(nb_path, out_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    with open(out_path, 'w', encoding='utf-8') as out:
        for i, c in enumerate(nb.get('cells', [])):
            if c['cell_type'] == 'markdown':
                out.write(f"--- Cell {i} (Markdown) ---\n")
                out.write("".join(c['source']).replace("\n", " ")[:200] + "\n\n")
            elif c['cell_type'] == 'code':
                out.write(f"--- Cell {i} (Code) ---\n")
                out.write("".join(c['source']).replace("\n", " ")[:200] + "\n")
                for o in c.get('outputs', []):
                    if 'text' in o:
                        out.write("OUTPUT: " + "".join(o['text']).replace("\n", " ")[:200] + "\n")
                out.write("\n")

if __name__ == "__main__":
    summarize_nb('Context_Aware_RAG_Chatbot.ipynb', 'nb_summary.txt')
