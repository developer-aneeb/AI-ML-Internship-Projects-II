import json
import sys

def export_code(nb_path, out_path):
    try:
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
            
        with open(out_path, 'w', encoding='utf-8') as out:
            for i, c in enumerate(nb.get('cells', [])):
                if c['cell_type'] == 'code':
                    out.write(f"# --- Cell {i} ---\n")
                    out.write("".join(c['source']))
                    out.write("\n\n")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    export_code('Context_Aware_RAG_Chatbot.ipynb', 'exported_code.py')
