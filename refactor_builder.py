import ast
import os

def main():
    with open('src/framework/graph/builder.py', 'r', encoding='utf-8') as f:
        source = f.read()

    tree = ast.parse(source)

    mixins = {
        'MetaControllerNodesMixin': [
            '_init_meta_controller',
            '_init_neuro_symbolic',
            '_extract_meta_controller_features'
        ],
        'RoutingNodesMixin': [
            '_route_decision_node',
            '_neural_route_decision',
            '_rule_based_route_decision',
            '_route_to_agents'
        ],
        'ConsensusNodesMixin': [
            '_aggregate_results_node',
            '_evaluate_consensus_node',
            '_check_consensus'
        ],
        'CoreNodesMixin': [
            '_entry_node',
            '_retrieve_context_node',
            '_wrap_node',
            '_node_retry',
            '_create_adk_node_handler'
        ]
    }

    os.makedirs('src/framework/graph/builder_components', exist_ok=True)
    with open('src/framework/graph/builder_components/__init__.py', 'w', encoding='utf-8') as f:
        pass

    # Find the class and extract methods
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == 'GraphBuilder':
            main_class = node
            break
    else:
        print("GraphBuilder not found")
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
        
        # Add basic imports that might be needed by the methods
        imports_code = """from typing import Any, Dict, List, Optional, cast, Union
import time
import asyncio
from typing import Literal
from src.observability.logging import get_structured_logger
from langgraph.graph import StateGraph, END
from pydantic import BaseModel
from src.framework.graph.state import GraphState

logger = get_structured_logger(__name__)
"""
        mixin_ast = ast.parse(imports_code)
        mixin_ast.body.append(mixin_class)

        filename = f"src/framework/graph/builder_components/{mixin_name.lower().replace('mixin', '_mixin')}.py"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(ast.unparse(mixin_ast))

    # Add imports to the original file
    imports_ast = ast.parse("""from .builder_components.metacontrollernodes_mixin import MetaControllerNodesMixin
from .builder_components.routingnodes_mixin import RoutingNodesMixin
from .builder_components.consensusnodes_mixin import ConsensusNodesMixin
from .builder_components.corenodes_mixin import CoreNodesMixin
""")
    insert_idx = 0
    if isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
        insert_idx = 1
    tree.body[insert_idx:insert_idx] = imports_ast.body

    with open('src/framework/graph/builder.py', 'w', encoding='utf-8') as f:
        f.write(ast.unparse(tree))

    print("Builder refactoring complete.")

if __name__ == '__main__':
    main()
