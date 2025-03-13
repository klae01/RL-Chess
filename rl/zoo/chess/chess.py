import collections
from typing import Any, Dict, List, Tuple, Optional

import chess
import chess.pgn
import gymnasium as gym
import numpy as np
from pettingzoo.utils.agent_selector import agent_selector as AgentSelector
from pettingzoo.utils.env import AECEnv


def build_legal_moves() -> List[chess.Move]:
    piece_types = [
        chess.PAWN,
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
        chess.KING,
    ]
    colors = [chess.WHITE, chess.BLACK]
    legal_moves_set = set()
    for square in chess.SQUARES:
        for piece_type in piece_types:
            for color in colors:
                for fill in range(2):
                    board = chess.Board()
                    board.clear()
                    if fill:
                        for s in chess.SQUARES:
                            board.set_piece_at(s, chess.Piece(chess.PAWN, not color))
                    board.set_piece_at(square, chess.Piece(piece_type, color))
                    board.turn = color
                    legal_moves_set.update(board.legal_moves)
    return sorted(legal_moves_set, key=lambda move: move.uci())


SPECIAL_DRAW_CLAIM = chess.Move.from_uci("0000")
LEGAL_MOVES = [SPECIAL_DRAW_CLAIM] + build_legal_moves()
MAX_ACTIONS = len(LEGAL_MOVES)
MOVE_TO_INDEX = {move.uci(): idx for idx, move in enumerate(LEGAL_MOVES)}


class ChessEnv(AECEnv):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()

        self.possible_agents = [chess.WHITE, chess.BLACK]
        self.agents = self.possible_agents[:]
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.board = chess.Board()
        self.outcome = None

        # Define observation and action spaces
        self.observation_spaces = {
            color: gym.spaces.Dict(
                {
                    "fen": gym.spaces.Text(max_length=100, charset="utf-8"),
                    "action_mask": gym.spaces.Box(
                        low=0, high=1, shape=(MAX_ACTIONS,), dtype=np.bool_
                    ),
                }
            )
            for color in self.possible_agents
        }

        self.action_spaces = {
            color: gym.spaces.Discrete(MAX_ACTIONS) for color in self.possible_agents
        }

        # Initialize state variables
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

    def reset(self, seed=None, options=None):
        """Resets the chessboard and sets the first player."""
        self.board.reset()
        self.outcome = None
        self.agents = self.possible_agents[:]
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        for agent in self.agents:
            self.rewards[agent] = 0.0
            self._cumulative_rewards[agent] = 0.0
            self.terminations[agent] = False
            self.truncations[agent] = False
            self.infos[agent] = {}

    def step(self, action):
        """Processes an action for the current agent."""
        turn = self.agent_selection
        if self.terminations[turn] or self.truncations[turn]:
            self._was_dead_step(action)
            return

        claim_draw = False
        move = self.interpret_action(action)

        if move == SPECIAL_DRAW_CLAIM:
            claim_draw = True
            possible, draw_move = self.check_draw_claim_possibility()
            if not possible:
                raise ValueError("Draw claim is not possible in the current position.")
            if draw_move is not None:
                if self.board.is_legal(draw_move):
                    move = draw_move
                    self.board.push(move)
                else:
                    raise ValueError(f"Draw claim move {draw_move.uci()} is not legal.")
        else:
            if not self.board.is_legal(move):
                raise ValueError(
                    f"Invalid move attempted by agent {turn}: {move.uci()}"
                )
            self.board.push(move)

        self.outcome = self.board.outcome(claim_draw=claim_draw)
        if claim_draw and self.outcome is None:
            raise ValueError(
                "Invalid draw claim: draw conditions are not met after the move."
            )

        # Determine game outcome
        if self.outcome is not None:
            winner = self.outcome.winner
            if winner is not None:
                self.rewards[chess.WHITE] = 1.0 if winner == chess.WHITE else -1.0
                self.rewards[chess.BLACK] = -1.0 if winner == chess.WHITE else 1.0
            else:
                self.rewards[chess.WHITE] = 0.0
                self.rewards[chess.BLACK] = 0.0
            for agent in self.agents:
                self.terminations[agent] = True
                self.infos[agent] = dict(
                    termination=self.outcome.termination.name,
                    result=self.outcome.result(),
                )
        else:
            self.rewards[chess.WHITE] = 0.0
            self.rewards[chess.BLACK] = 0.0

        # Switch to the next player
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

    def observe(self, agent):
        """Returns the board state."""
        if agent not in self.agents:
            return None
        return {"fen": self.board.fen(), "action_mask": self.get_action_mask()}

    def render(self):
        """Displays the chessboard in Unicode format."""
        print(self.board.unicode())

    def state(self) -> np.ndarray:
        """Returns the full environment state as an array."""
        return np.array(self.board.fen(), dtype=str)

    def close(self):
        """Releases resources (not required in this implementation)."""
        pass

    def interpret_action(self, action):
        """Converts an action into a chess.Move object."""
        return LEGAL_MOVES[action]

    def can_claim_draw(self):
        """
        Checks whether the current position allows a claim for a draw
        due to the fifty-move rule or threefold repetition.
        """
        return self.board.can_claim_fifty_moves() or self.board.is_repetition(3)

    def check_draw_claim_possibility(self) -> Tuple[bool, Optional[chess.Move]]:
        """
        Checks if a draw claim is possible immediately or after a legal move.

        Returns:
            (True, None) if a draw claim is possible immediately.
            (True, move) if playing a legal move enables a draw claim.
            (False, None) if no move can lead to a draw claim.
        """
        if self.can_claim_draw():
            return True, None

        for move in self.board.legal_moves:
            self.board.push(move)
            if self.can_claim_draw():
                self.board.pop()
                return True, move
            self.board.pop()

        return False, None

    def get_action_mask(self) -> np.ndarray:
        mask = np.zeros(MAX_ACTIONS, dtype=np.bool_)
        claim_index = MOVE_TO_INDEX[SPECIAL_DRAW_CLAIM.uci()]
        mask[claim_index] = self.check_draw_claim_possibility()[0]
        for move in self.board.legal_moves:
            code = move.uci()
            if code not in MOVE_TO_INDEX:
                raise ValueError(
                    f"Unexpected move {code} not found in MOVE_TO_INDEX mapping. Please update the global legal moves set."
                )
            mask[MOVE_TO_INDEX[code]] = True
        return mask

    def save(self, fp=None):
        """Saves the current game in PGN format."""
        game = chess.pgn.Game()
        switchyard = collections.deque()
        while self.board.move_stack:
            switchyard.append(self.board.pop())
        game.setup(self.board)
        node = game
        while switchyard:
            move = switchyard.pop()
            node = node.add_variation(move)
            self.board.push(move)
        game.headers["Result"] = self.board.result()
        return print(game, file=fp, end="\n\n") if fp else game
