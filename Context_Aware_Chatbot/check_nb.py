import json

with open('Context_Aware_RAG_Chatbot.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for c in nb.get('cells', []):
    if c['cell_type'] == 'markdown':
        print("=== MARKDOWN ===")
        print("".join(c['source']))
    elif c['cell_type'] == 'code':
        print("=== CODE ===")
        source = "".join(c['source'])
        print(source[:50])
        for o in c.get('outputs', []):
            if 'text' in o:
                print("--- OUTPUT TEXT ---")
                print("".join(o['text']).strip())
