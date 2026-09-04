import ast
import os

def main():
    with open('src/games/chess/ui.py', 'r', encoding='utf-8') as f:
        source = f.read()

    tree = ast.parse(source)

    modules = {
        'rendering': [
            'get_piece_unicode',
            'fen_to_board',
            'render_board_html',
            'render_scorecard_html',
            'format_move_history',
            'render_learning_status',
            'render_learning_board_html',
            'format_analysis'
        ],
        'game_logic': [
            'get_game_status',
            'initialize_game',
            'validate_move',
            'apply_player_move',
            'make_ai_move_sync',
            'get_ai_move',
            'undo_move',
            'export_game_pgn'
        ],
        'continuous_learning': [
            'start_continuous_learning',
            'stop_continuous_learning',
            'pause_continuous_learning',
            'get_learning_status'
        ]
    }

    os.makedirs('src/games/chess/ui_components', exist_ok=True)
    with open('src/games/chess/ui_components/__init__.py', 'w', encoding='utf-8') as f:
        pass

    extracted_nodes = {k: [] for k in modules.keys()}
    remaining_body = []

    for item in tree.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found_module = False
            for module_name, method_names in modules.items():
                if item.name in method_names:
                    extracted_nodes[module_name].append(item)
                    found_module = True
                    break
            if not found_module:
                remaining_body.append(item)
        else:
            remaining_body.append(item)

    tree.body = remaining_body

    # Create the module files
    for module_name, nodes in extracted_nodes.items():
        module_ast = ast.Module(body=[], type_ignores=[])
        
        # Add basic imports
        imports_code = """import asyncio
import io
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
import traceback

import chess
import chess.pgn
from langgraph.graph import StateGraph

from src.framework.mcts.neural_mcts import GameState, NeuralMCTS
from src.games.chess.state import ChessState, get_legal_actions, is_terminal

logger = logging.getLogger(__name__)
"""
        imports_ast = ast.parse(imports_code)
        module_ast.body.extend(imports_ast.body)
        module_ast.body.extend(nodes)

        filename = f"src/games/chess/ui_components/{module_name}.py"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(ast.unparse(module_ast))

    # Add imports to the original file
    imports_ast = ast.parse("""from .ui_components.rendering import (
    get_piece_unicode, fen_to_board, render_board_html, render_scorecard_html,
    format_move_history, render_learning_status, render_learning_board_html, format_analysis
)
from .ui_components.game_logic import (
    get_game_status, initialize_game, validate_move, apply_player_move,
    make_ai_move_sync, get_ai_move, undo_move, export_game_pgn
)
from .ui_components.continuous_learning import (
    start_continuous_learning, stop_continuous_learning, pause_continuous_learning, get_learning_status
)
""")
    insert_idx = 0
    if isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
        insert_idx = 1
    tree.body[insert_idx:insert_idx] = imports_ast.body

    with open('src/games/chess/ui.py', 'w', encoding='utf-8') as f:
        f.write(ast.unparse(tree))

    print("UI refactoring complete.")

if __name__ == '__main__':
    main()
