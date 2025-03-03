import collections
from typing import Any, Dict, List, Tuple

import chess
import chess.pgn
import gymnasium as gym
import numpy as np
from pettingzoo.utils.agent_selector import agent_selector as AgentSelector
from pettingzoo.utils.env import AECEnv

# --- Special action indices ---
SPECIAL_NONE = 0
SPECIAL_PROMOTE_QUEEN = 1
SPECIAL_PROMOTE_ROOK = 2
SPECIAL_PROMOTE_BISHOP = 3
SPECIAL_PROMOTE_KNIGHT = 4
SPECIAL_CASTLE_KINGSIDE = 5
SPECIAL_CASTLE_QUEENSIDE = 6
SPECIAL_DRAW_CLAIM = 7


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
                    "allowed_actions": gym.spaces.Sequence(
                        gym.spaces.Box(low=0, high=7, shape=(3,), dtype=np.int8)
                    ),
                }
            )
            for color in self.possible_agents
        }

        self.action_spaces = {
            color: gym.spaces.MultiDiscrete([64, 64, 8])
            for color in self.possible_agents
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

        claim_draw = action[2] == SPECIAL_DRAW_CLAIM
        move = self.interpret_action(action)

        if move != chess.Move.null():
            if not self.board.is_legal(move):
                raise ValueError(f"Invalid move attempted by {turn}: {move.uci()}")
            self.board.push(move)

        self.outcome = self.board.outcome(claim_draw=claim_draw)

        if claim_draw and self.outcome is None:
            raise ValueError("Invalid claim draw: Draw conditions not met")

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

        return {
            "fen": self.board.fen(),
            "allowed_actions": self.get_allowed_actions(agent),
        }

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
        from_sq, to_sq, special = action
        promotion = {
            SPECIAL_PROMOTE_QUEEN: chess.QUEEN,
            SPECIAL_PROMOTE_ROOK: chess.ROOK,
            SPECIAL_PROMOTE_BISHOP: chess.BISHOP,
            SPECIAL_PROMOTE_KNIGHT: chess.KNIGHT,
        }.get(special, None)

        if special == SPECIAL_DRAW_CLAIM:
            if from_sq == to_sq:
                return chess.Move.null()
            return chess.Move(from_sq, to_sq, promotion=promotion)

        return chess.Move(from_sq, to_sq, promotion=promotion)

    def can_claim_draw(self):
        """
        Checks whether the current position allows a claim for a draw
        due to the fifty-move rule or threefold repetition.
        """
        return self.board.can_claim_fifty_moves() or self.board.is_repetition(3)

    def can_claim_draw_after_move(self, move):
        """Checks whether a draw can be claimed after executing the given move."""
        self.board.push(move)
        claim_possible = self.can_claim_draw()
        self.board.pop()
        return claim_possible

    def get_allowed_actions(self, color):
        """Returns all currently legal actions."""
        allowed = []
        for move in self.board.legal_moves:
            special = SPECIAL_NONE
            if self.board.is_castling(move):
                if self.board.is_kingside_castling(move):
                    special = SPECIAL_CASTLE_KINGSIDE
                elif self.board.is_queenside_castling(move):
                    special = SPECIAL_CASTLE_QUEENSIDE
            elif move.promotion is not None:
                special = {
                    chess.QUEEN: SPECIAL_PROMOTE_QUEEN,
                    chess.ROOK: SPECIAL_PROMOTE_ROOK,
                    chess.BISHOP: SPECIAL_PROMOTE_BISHOP,
                    chess.KNIGHT: SPECIAL_PROMOTE_KNIGHT,
                }.get(move.promotion, SPECIAL_NONE)
            allowed.append((move.from_square, move.to_square, special))

            if self.can_claim_draw_after_move(move):
                allowed.append((move.from_square, move.to_square, SPECIAL_DRAW_CLAIM))

        if self.can_claim_draw():
            king_sq = self.board.king(color)
            if king_sq is not None:
                allowed.append((king_sq, king_sq, SPECIAL_DRAW_CLAIM))
        return allowed

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
