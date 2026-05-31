import json
import sys

def extract(notebook_path, output_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for cell in nb.get('cells', []):
            if cell.get('cell_type') == 'code':
                source = "".join(cell.get('source', []))
                f.write(source + "\n\n")

if __name__ == "__main__":
    extract(sys.argv[1], sys.argv[2])
