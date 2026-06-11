import json
import io

try:
    with open('Context_Aware_RAG_Chatbot.ipynb', 'rb') as f:
        raw = f.read()

    # Try decoding
    decoded = raw.decode('utf-16' if b'\x00' in raw else 'utf-8')
    
    with open('Context_Aware_RAG_Chatbot_utf8.ipynb', 'w', encoding='utf-8') as out:
        out.write(decoded)
        
    print("Successfully converted to UTF-8")
except Exception as e:
    with open('error.log', 'w', encoding='utf-8') as err:
        err.write(str(e))
