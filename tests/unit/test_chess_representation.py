"""Unit tests for src/games/chess/representation.py."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# We need to mock torch and chess before importing the module under test
# since they may not be available in the test environment.


def _make_mock_piece(piece_type, color):
    """Create a mock chess piece."""
    piece = MagicMock()
    piece.piece_type = piece_type
    piece.color = color
    return piece


@pytest.fixture
def mock_torch():
    """Provide a mock torch module if torch is not available."""
    try:
        import torch
        yield torch
    except ImportError:
        mock_torch_mod = MagicMock()
        mock_torch_mod.float32 = "float32"

        def _zeros(*shape, dtype=None):
            """Create a numpy-backed mock tensor."""
            arr = np.zeros(shape)
            tensor = MagicMock()
            tensor.shape = arr.shape

            def getitem(idx):
                return arr[idx]

            def setitem(idx, val):
                arr[idx] = val

            tensor.__getitem__ = getitem
            tensor.__setitem__ = setitem

            def fill_(val):
                arr[...] = val

            # Each plane should support fill_
            class PlaneMock:
                def __init__(self, plane_idx):
                    self._idx = plane_idx

                def fill_(self, val):
                    arr[self._idx, :, :] = val

                def __getitem__(self, key):
                    return arr[self._idx][key]

                def __setitem__(self, key, val):
                    arr[self._idx][key] = val

            # Override __getitem__ to return plane mocks for integer indices

            def smart_getitem(idx):
                if isinstance(idx, int):
                    return PlaneMock(idx)
                return arr[idx]

            tensor.__getitem__ = smart_getitem
            tensor.__setitem__ = lambda idx, val: arr.__setitem__(idx, val)
            tensor._arr = arr
            return tensor

        mock_torch_mod.zeros = _zeros

        def _stack(tensors):
            arrs = [t._arr for t in tensors]
            stacked = np.stack(arrs)
            result = MagicMock()
            result.shape = stacked.shape
            result._arr = stacked
            return result

        mock_torch_mod.stack = _stack

        with patch.dict(sys.modules, {"torch": mock_torch_mod}):
            yield mock_torch_mod


@pytest.fixture
def mock_chess():
    """Provide the chess module or a mock."""
    try:
        import chess
        yield chess
    except ImportError:
        mock_chess_mod = MagicMock()
        mock_chess_mod.WHITE = True
        mock_chess_mod.BLACK = False
        mock_chess_mod.SQUARES = list(range(64))
        mock_chess_mod.Board = MagicMock
        yield mock_chess_mod


@pytest.mark.unit
class TestChessBoardRepresentationImport:
    """Test that the module can be imported and basic properties work."""

    def test_piece_to_plane_mapping(self):
        from src.games.chess.representation import PIECE_TO_PLANE
        assert PIECE_TO_PLANE[1] == 0  # Pawn
        assert PIECE_TO_PLANE[2] == 1  # Knight
        assert PIECE_TO_PLANE[3] == 2  # Bishop
        assert PIECE_TO_PLANE[4] == 3  # Rook
        assert PIECE_TO_PLANE[5] == 4  # Queen
        assert PIECE_TO_PLANE[6] == 5  # King

    def test_default_config(self):
        from src.games.chess.representation import ChessBoardRepresentation
        rep = ChessBoardRepresentation()
        assert rep.num_planes == 22  # Default: 12 + 1 + 4 + 1 + 1 + 1 + 2
        assert rep.input_shape == (22, 8, 8)

    def test_custom_config(self):
        from src.games.chess.config import ChessBoardConfig
        from src.games.chess.representation import ChessBoardRepresentation
        config = ChessBoardConfig(include_history=True, history_length=4)
        rep = ChessBoardRepresentation(config)
        # 22 base + 4 * 12 history = 70
        assert rep.num_planes == 22 + 4 * 12

    def test_repr(self):
        from src.games.chess.representation import ChessBoardRepresentation
        rep = ChessBoardRepresentation()
        s = repr(rep)
        assert "ChessBoardRepresentation" in s
        assert "num_planes=22" in s


@pytest.mark.unit
class TestChessBoardRepresentationEncode:
    """Test encoding with the real chess and torch libraries if available."""

    @pytest.fixture(autouse=True)
    def skip_if_no_torch(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch not available")

    @pytest.fixture(autouse=True)
    def skip_if_no_chess(self):
        try:
            import chess  # noqa: F401
        except ImportError:
            pytest.skip("chess library not available")

    def test_encode_initial_position(self):
        import chess
        import torch

        from src.games.chess.representation import ChessBoardRepresentation

        rep = ChessBoardRepresentation()
        board = chess.Board()
        tensor = rep.encode(board)
        assert tensor.shape == (22, 8, 8)
        assert tensor.dtype == torch.float32

    def test_encode_white_perspective(self):
        import chess

        from src.games.chess.representation import ChessBoardRepresentation

        rep = ChessBoardRepresentation()
        board = chess.Board()
        tensor = rep.encode(board, from_perspective=1)

        # Plane 0 = white pawns: rank 1 should be all 1s
        for file in range(8):
            assert tensor[0, 1, file].item() == 1.0  # White pawns on rank 2 (index 1)

        # Plane 6 = black pawns: rank 6 should be all 1s
        for file in range(8):
            assert tensor[6, 6, file].item() == 1.0  # Black pawns on rank 7 (index 6)

    def test_encode_black_perspective(self):
        import chess

        from src.games.chess.representation import ChessBoardRepresentation

        rep = ChessBoardRepresentation()
        board = chess.Board()
        tensor = rep.encode(board, from_perspective=-1)

        # When flipped, black pieces use planes 0-5 and white pieces use planes 6-11
        # Black pawns (originally on rank 6, files 0-7) become rank 1 (7-6=1), files flipped
        for file in range(8):
            assert tensor[0, 1, file].item() == 1.0  # Black pawns appear in plane 0

    def test_encode_side_to_move_white(self):
        import chess

        from src.games.chess.representation import ChessBoardRepresentation

        rep = ChessBoardRepresentation()
        board = chess.Board()  # White to move
        tensor = rep.encode(board)
        # Plane 12 should be all 1s (white to move)
        assert tensor[12, 0, 0].item() == 1.0

    def test_encode_side_to_move_black(self):
        import chess

        from src.games.chess.representation import ChessBoardRepresentation

        rep = ChessBoardRepresentation()
        board = chess.Board()
        board.push_san("e4")  # Black to move
        tensor = rep.encode(board)
        # Plane 12 should be all 0s (black to move)
        assert tensor[12, 0, 0].item() == 0.0

    def test_encode_castling_rights_initial(self):
        import chess

        from src.games.chess.representation import ChessBoardRepresentation

        rep = ChessBoardRepresentation()
        board = chess.Board()
        tensor = rep.encode(board)
        # Planes 13-16 should all be 1 (all castling rights available)
        for plane in range(13, 17):
            assert tensor[plane, 0, 0].item() == 1.0

    def test_encode_no_castling_rights(self):
        import chess

        from src.games.chess.representation import ChessBoardRepresentation

        rep = ChessBoardRepresentation()
        board = chess.Board()
        board.set_castling_fen("-")
        tensor = rep.encode(board)
        for plane in range(13, 17):
            assert tensor[plane, 0, 0].item() == 0.0

    def test_encode_en_passant(self):
        import chess

        from src.games.chess.representation import ChessBoardRepresentation

        rep = ChessBoardRepresentation()
        board = chess.Board()
        board.push_san("e4")  # Creates en passant square at e3
        tensor = rep.encode(board)
        # Plane 17 = en passant. e3 = file 4, rank 2
        assert tensor[17, 2, 4].item() == 1.0

    def test_encode_halfmove_clock(self):
        import chess

        from src.games.chess.representation import ChessBoardRepresentation

        rep = ChessBoardRepresentation()
        board = chess.Board()
        board.halfmove_clock = 50
        tensor = rep.encode(board)
        # Plane 18 = halfmove clock normalized to [0,1]: 50/100 = 0.5
        assert tensor[18, 0, 0].item() == pytest.approx(0.5)

    def test_encode_fullmove_number(self):
        import chess

        from src.games.chess.representation import ChessBoardRepresentation

        rep = ChessBoardRepresentation()
        board = chess.Board()
        board.fullmove_number = 100
        tensor = rep.encode(board)
        # Plane 19 = fullmove normalized: 100/200 = 0.5
        assert tensor[19, 0, 0].item() == pytest.approx(0.5)

    def test_encode_castling_flipped(self):
        import chess

        from src.games.chess.representation import ChessBoardRepresentation

        rep = ChessBoardRepresentation()
        board = chess.Board()
        # Remove black castling rights
        board.set_castling_fen("KQ")
        tensor = rep.encode(board, from_perspective=-1)
        # When flipped, black castling (planes 13-14) should have white's original rights
        # and white castling (planes 15-16) should have black's (none)
        assert tensor[13, 0, 0].item() == 0.0  # was black kingside -> none
        assert tensor[14, 0, 0].item() == 0.0  # was black queenside -> none
        assert tensor[15, 0, 0].item() == 1.0  # was white kingside -> yes
        assert tensor[16, 0, 0].item() == 1.0  # was white queenside -> yes

    def test_encode_batch(self):
        import chess

        from src.games.chess.representation import ChessBoardRepresentation

        rep = ChessBoardRepresentation()
        boards = [chess.Board(), chess.Board()]
        tensor = rep.encode_batch(boards)
        assert tensor.shape == (2, 22, 8, 8)

    def test_encode_batch_different_perspectives(self):
        import chess

        from src.games.chess.representation import ChessBoardRepresentation

        rep = ChessBoardRepresentation()
        boards = [chess.Board(), chess.Board()]
        tensor = rep.encode_batch(boards, from_perspective=[1, -1])
        assert tensor.shape == (2, 22, 8, 8)

    def test_decode_piece_planes(self):
        import chess

        from src.games.chess.representation import ChessBoardRepresentation

        rep = ChessBoardRepresentation()
        board = chess.Board()
        tensor = rep.encode(board)
        decoded = rep.decode_piece_planes(tensor)
        # White pawns should have 8 positions
        assert "P" in decoded
        assert len(decoded["P"]) == 8
        # Black pawns too
        assert "p" in decoded
        assert len(decoded["p"]) == 8
        # Kings
        assert "K" in decoded
        assert len(decoded["K"]) == 1
        assert "k" in decoded
        assert len(decoded["k"]) == 1

    def test_encode_with_history(self):
        import chess

        from src.games.chess.config import ChessBoardConfig
        from src.games.chess.representation import ChessBoardRepresentation

        config = ChessBoardConfig(include_history=True, history_length=2)
        rep = ChessBoardRepresentation(config)
        board = chess.Board()
        board.push_san("e4")
        history_board = chess.Board()  # Initial position as history
        tensor = rep.encode(board, history=[history_board])
        # Total planes = 22 + 2*12 = 46
        assert tensor.shape[0] == 46


@pytest.mark.unit
class TestBoardToTensor:
    @pytest.fixture(autouse=True)
    def skip_if_no_deps(self):
        try:
            import chess  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch or chess not available")

    def test_board_to_tensor(self):
        import chess

        from src.games.chess.representation import board_to_tensor

        board = chess.Board()
        tensor = board_to_tensor(board)
        assert tensor.shape == (22, 8, 8)

    def test_board_to_tensor_black_perspective(self):
        import chess

        from src.games.chess.representation import board_to_tensor

        board = chess.Board()
        tensor = board_to_tensor(board, from_perspective=-1)
        assert tensor.shape == (22, 8, 8)


@pytest.mark.unit
class TestTensorToNumpy:
    @pytest.fixture(autouse=True)
    def skip_if_no_torch(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch not available")

    def test_tensor_to_numpy(self):
        import torch

        from src.games.chess.representation import tensor_to_numpy

        tensor = torch.zeros(22, 8, 8)
        arr = tensor_to_numpy(tensor)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (22, 8, 8)

    def test_tensor_to_numpy_values(self):
        import torch

        from src.games.chess.representation import tensor_to_numpy

        tensor = torch.ones(2, 3, 4)
        arr = tensor_to_numpy(tensor)
        assert np.all(arr == 1.0)


@pytest.mark.unit
class TestEnPassantFlipped:
    """Test en passant encoding with flipped perspective."""

    @pytest.fixture(autouse=True)
    def skip_if_no_deps(self):
        try:
            import chess  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch or chess not available")

    def test_en_passant_flipped(self):
        import chess

        from src.games.chess.representation import ChessBoardRepresentation

        rep = ChessBoardRepresentation()
        board = chess.Board()
        board.push_san("e4")
        tensor = rep.encode(board, from_perspective=-1)
        # e3 en passant square = square 20 (file=4, rank=2)
        # Flipped: file=7-4=3, rank=7-2=5
        assert tensor[17, 5, 3].item() == 1.0


@pytest.mark.unit
class TestRepetitionPlanes:
    """Test repetition count encoding."""

    @pytest.fixture(autouse=True)
    def skip_if_no_deps(self):
        try:
            import chess  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch or chess not available")

    def test_no_two_fold_repetition(self):
        import chess

        from src.games.chess.representation import ChessBoardRepresentation

        rep = ChessBoardRepresentation()
        board = chess.Board()
        board.push_san("e4")  # After one move, no repetition of this position
        tensor = rep.encode(board)
        # Plane 21 should be 0 (no 2-fold repetition)
        assert tensor[21, 0, 0].item() == 0.0

    def test_repetition_detection(self):
        import chess

        from src.games.chess.representation import ChessBoardRepresentation

        rep = ChessBoardRepresentation()
        board = chess.Board()
        # Create a repetition: Nf3 Nf6 Ng1 Ng8 (back to start)
        moves = ["Nf3", "Nf6", "Ng1", "Ng8"]
        for m in moves:
            board.push_san(m)
        tensor = rep.encode(board)
        # After returning to start position, is_repetition(1) should be True
        assert tensor[20, 0, 0].item() == 1.0
