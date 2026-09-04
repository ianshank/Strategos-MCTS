import ast

with open('src/training/unified_orchestrator.py', 'r', encoding='utf-8') as f:
    source = f.read()

tree = ast.parse(source)
for node in tree.body:
    if isinstance(node, ast.ClassDef) and node.name == 'UnifiedTrainingOrchestrator':
        print(f"Class {node.name} has {len(node.body)} members")
        for item in node.body:
            if isinstance(item, ast.FunctionDef) or isinstance(item, ast.AsyncFunctionDef):
                print(f"  - {item.name}")
