"""Entry point for the checkers game
Run: python -m src.main [--ai-white] [--ai-black] [--ai-both] [--model <path>] [--train <out_path>] [--train-epochs <n>] [--nn-controls <none|white|black|both>]
Options:
--ai-white       : make the white player AI-controlled
--ai-black       : make the black player AI-controlled
--ai-both        : make both players AI-controlled
"""

import sys
from typing import Set, Optional, Tuple

from src.engine.board import Board
from src.ui import PygameUI


def parse_args(argv) -> Tuple[Set[str], Optional[str], Optional[str], int, Set[str]]:
    ai: Set[str] = set()
    model_path: Optional[str] = None
    train_path: Optional[str] = None
    train_epochs = 5
    model_control: Set[str] = set()

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

    if '--train' in argv:
        idx = argv.index('--train')
        if idx + 1 < len(argv):
            train_path = argv[idx + 1]
        else:
            print('Warning: --train provided but no path given; ignoring')

    if '--train-epochs' in argv:
        try:
            idx = argv.index('--train-epochs')
            if idx + 1 < len(argv):
                train_epochs = int(argv[idx + 1])
        except Exception:
            pass

    if '--nn-controls' in argv:
        try:
            idx = argv.index('--nn-controls')
            if idx + 1 < len(argv):
                val = argv[idx + 1].lower()
                if val == 'none':
                    model_control = set()
                elif val == 'white':
                    model_control = {'white'}
                elif val == 'black':
                    model_control = {'black'}
                elif val == 'both':
                    model_control = {'white', 'black'}
        except Exception:
            pass

    return ai, model_path, train_path, train_epochs, model_control


def main(argv=None):
    argv = argv or sys.argv[1:]
    ai_players, model_path, train_path, train_epochs, model_control = parse_args(argv)

    # Optional imports for model/training
    model = None
    CheckersNet = None
    train_random_positions = None
    load_model = None
    try:
        import torch
        from src.nn import CheckersNet as _CN, train_random_positions as _train_random_positions, load_model as _load_model
        CheckersNet = _CN
        train_random_positions = _train_random_positions
        load_model = _load_model
    except Exception:
        # PyTorch or nn utilities not available; that's fine for interactive use without model
        pass

    # If --train was passed, run non-interactive training and save model then exit
    if train_path is not None:
        if CheckersNet is None or train_random_positions is None:
            print("Training requested but PyTorch or training utilities are not available. Install PyTorch and try again.")
            return
        # create model, train, save
        print(f"Training model for {train_epochs} epochs; will save to {train_path} when done")
        model = CheckersNet()
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            train_random_positions(model, epochs=train_epochs, n_positions=128, device=device)
            # save
            from src.nn import save_model
            save_model(model, train_path)
            print(f"Training completed and model saved to {train_path}")
        except Exception as e:
            print("Training failed:", e)
        return

    # Otherwise, attempt to load model path if provided
    if model_path is not None and CheckersNet is not None and load_model is not None:
        try:
            import torch
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
    # set initial model control if provided
    ui.model_control = model_control

    try:
        ui.run()
    except Exception as e:
        print("Error running UI:", e)
        print("If this is an ImportError for pygame or torch, install them with: python -m pip install pygame torch")
        raise


if __name__ == "__main__":
    main()
