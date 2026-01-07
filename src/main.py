"""Entry point for the checkers game
Run: python -m src.main [--ai-white] [--ai-black] [--ai-both] [--model <path>]
"""

import sys
from typing import Set, Optional, Tuple

from src.engine.board import Board
from src.ui import PygameUI


def parse_args(argv) -> Tuple[Set[str], Optional[str]]:
    ai: Set[str] = set()
    model_path: Optional[str] = None
    if '--ai-both' in argv:
        ai.add('white')
        ai.add('black')
    else:
        if '--ai-white' in argv:
            ai.add('white')
        if '--ai-black' in argv:
            ai.add('black')
    if '--model' in argv:
        idx = argv.index('--model')
        if idx + 1 < len(argv):
            model_path = argv[idx + 1]
        else:
            print('Warning: --model provided but no path given; ignoring')
    return ai, model_path


def main(argv=None):
    argv = argv or sys.argv[1:]
    ai_players, model_path = parse_args(argv)

    model = None
    if model_path is not None:
        try:
            import torch
            from src.nn import CheckersNet, load_model

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = CheckersNet()
            load_model(model, model_path, device=device)
            print(f"Loaded model from {model_path} on device {device}")
        except Exception as e:
            print("Failed to load model:", e)
            print("Proceeding without model. To use a model, install PyTorch and provide a valid path to --model")
            model = None

    board = Board.setup_start()
    ui = PygameUI(board, ai_players=ai_players, model=model)
    try:
        ui.run()
    except Exception as e:
        print("Error running UI:", e)
        print("If this is an ImportError for pygame or torch, install them with: python -m pip install pygame torch")
        raise


if __name__ == "__main__":
    main()
