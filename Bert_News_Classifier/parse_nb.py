import json
import sys

def parse_notebook(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    with open('d:\\projects\\Internship-II\\Bert_News_Classifier\\notebook_summary.txt', 'w', encoding='utf-8') as out:
        for cell in nb.get('cells', []):
            if cell['cell_type'] == 'markdown':
                source = cell.get('source', [])
                if isinstance(source, list):
                    source = "".join(source)
                out.write("MARKDOWN:\n" + source + "\n\n")
            elif cell['cell_type'] == 'code':
                outputs = cell.get('outputs', [])
                for output in outputs:
                    if output.get('output_type') == 'stream':
                        text = output.get('text', '')
                        if isinstance(text, list):
                            text = "".join(text)
                        out.write("OUTPUT:\n" + text + "\n")
                    elif 'text/plain' in output.get('data', {}):
                        text = output['data']['text/plain']
                        if isinstance(text, list):
                            text = "".join(text)
                        out.write("OUTPUT (text/plain):\n" + text + "\n")
                        
if __name__ == '__main__':
    parse_notebook('d:\\projects\\Internship-II\\Bert_News_Classifier\\AG_News_BERT_Classifier.ipynb')
