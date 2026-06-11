import json
import traceback

try:
    with open('Context_Aware_RAG_Chatbot.ipynb', 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    with open('parsed_output.txt', 'w', encoding='utf-8') as out:
        try:
            nb = json.loads(content)
            for i, c in enumerate(nb.get('cells', [])):
                if c['cell_type'] == 'code':
                    out.write(f"--- Code Cell {i} ---\n")
                    out.write("".join(c['source']))
                    out.write("\n\n")
        except Exception as e:
            out.write("JSON Decode Error:\n")
            out.write(traceback.format_exc())
            
except Exception as e:
    with open('error.log', 'w', encoding='utf-8') as err:
        err.write(traceback.format_exc())
