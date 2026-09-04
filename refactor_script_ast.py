import ast
import os

def main():
    with open('src/training/unified_orchestrator.py', 'r', encoding='utf-8') as f:
        source = f.read()

    tree = ast.parse(source)

    mixins = {
        'MetricsMixin': ['_log_metrics', '_get_memory_utilization'],
        'CheckpointMixin': ['_save_checkpoint', 'load_checkpoint'],
        'SelfPlayMixin': ['_generate_self_play_data'],
        'TrainingMixin': ['_train_policy_value_network', '_compute_gradient_norm', '_train_hrm_agent', '_train_trm_agent', '_evaluate']
    }

    os.makedirs('src/training/orchestrator_components', exist_ok=True)
    with open('src/training/orchestrator_components/__init__.py', 'w', encoding='utf-8') as f:
        pass

    # Find the class and extract methods
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == 'UnifiedTrainingOrchestrator':
            main_class = node
            break
    else:
        print("UnifiedTrainingOrchestrator not found")
        return

    extracted_nodes = {k: [] for k in mixins.keys()}
    remaining_body = []

    for item in main_class.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found_mixin = False
            for mixin_name, method_names in mixins.items():
                if item.name in method_names:
                    extracted_nodes[mixin_name].append(item)
                    found_mixin = True
                    break
            if not found_mixin:
                remaining_body.append(item)
        else:
            remaining_body.append(item)

    main_class.body = remaining_body

    # We also need to add bases to the class
    for mixin_name in mixins.keys():
        main_class.bases.append(ast.Name(id=mixin_name, ctx=ast.Load()))

    # Create the mixin files
    for mixin_name, nodes in extracted_nodes.items():
        mixin_class = ast.ClassDef(
            name=mixin_name,
            bases=[],
            keywords=[],
            body=nodes if nodes else [ast.Pass()],
            decorator_list=[]
        )
        
        # We need to add some imports to the mixin files. For simplicity, just add some common ones.
        imports_code = """from typing import Any
import time
from pathlib import Path
import torch
import torch.nn as nn
from src.observability.logging import get_structured_logger
from src.training.replay_buffer import Experience

logger = get_structured_logger(__name__)
"""
        mixin_ast = ast.parse(imports_code)
        mixin_ast.body.append(mixin_class)

        filename = f"src/training/orchestrator_components/{mixin_name.lower().replace('mixin', '_mixin')}.py"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(ast.unparse(mixin_ast))

    # Add imports to the original file
    imports_ast = ast.parse("""from .orchestrator_components.metrics_mixin import MetricsMixin
from .orchestrator_components.checkpoint_mixin import CheckpointMixin
from .orchestrator_components.self_play_mixin import SelfPlayMixin
from .orchestrator_components.training_mixin import TrainingMixin
""")
    # Insert after docstring or at top
    insert_idx = 0
    if isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
        insert_idx = 1
    tree.body[insert_idx:insert_idx] = imports_ast.body

    with open('src/training/unified_orchestrator.py', 'w', encoding='utf-8') as f:
        f.write(ast.unparse(tree))

    print("Refactoring complete using ast.unparse")

if __name__ == '__main__':
    main()
