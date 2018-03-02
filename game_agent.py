"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
from math import sqrt


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    This should be the best heuristic function for your project submission.
    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    
		This heuristic is developed based on the idea that the proximity of the
		player to the center provides the player more options to move around in 
		the board. This is captured as a center_score based on distance of the 
		player from the center of the board. This score is weighted by the ratio
		of player move and opponent's legal move as this factor provides an 
		exact estimate of available moves for each player.
	"""
	
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    oppn_player = game.get_opponent(player)
    
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(oppn_player))

    if opp_moves == 0:
        return float("inf")

	# center of the board	
    w, h = game.width / 2., game.height / 2.
    p1x, p1y = game.get_player_location(player);
    p2x, p2y = game.get_player_location(oppn_player);
	#center_score for each player. Eucledian distance
    center_score_player = sqrt(((p1y-h)**2)+((p1x-w)**2))
    center_score_oppn_player = sqrt(((p1y-h)**2)+((p1x-w)**2))

    return float((center_score_player/center_score_oppn_player) * (own_moves/opp_moves))

def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    This should be the best heuristic function for your project submission.
    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    
	This heuristic is developed on the idea that ratio of available legal moves
	to the blank spaces for a player on the board determines the territory 
	that the player has, to play. Comparing the same value for the opponent 
	gives a picture of how aggresively either of the player has retricted the other
	player's movement. Here the difference of own_ratio_open_to_blankspaces to
	opp_ratio_open_to_blankspaces determines the score
	"""
	
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    blank_spaces = len(game.get_blank_spaces())
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    own_ratio_open_to_blankspaces = float(own_moves/blank_spaces)
    opp_ratio_open_to_blankspaces = float(opp_moves/blank_spaces)
    return float(own_ratio_open_to_blankspaces - opp_ratio_open_to_blankspaces)

def custom_score_3(game, player):
	"""Calculate the heuristic value of a game state from the point of view
	of the given player.
	Note: this function should be called from within a Player instance as
	`self.score()` -- you should not need to call this function directly.
	Parameters
	----------
	game : `isolation.Board`
		An instance of `isolation.Board` encoding the current state of the
		game (e.g., player locations and blocked cells).
	player : object
		A player instance in the current game (i.e., an object corresponding to
		one of the player objects `game.__player_1__` or `game.__player_2__`.)
	Returns
	-------
	float
		The heuristic value of the current game state to the specified player.
		This heuristic is developed on the idea that farther apart the players
		are, lesser their chance of restricting the other player movement. The
		availablity of open space to move reduces as the game of isolation 
		progresses by occupying positions. This will lead to more restrictions
		on the legal moves. The promixity of the players combined with the 
		ratio of own legal moves to	opponents legal moves gives a good 
		estimate of player winning chance.
	"""

	if game.is_loser(player):
		return float("-inf")
	
	if game.is_winner(player):
		return float("inf")
	
	oppn_player = game.get_opponent(player)
	own_moves = len(game.get_legal_moves(player))
	opp_moves = len(game.get_legal_moves(oppn_player))
	
	if opp_moves == 0:
		return float("inf")

	p1x, p1y = game.get_player_location(player);
	p2x, p2y = game.get_player_location(oppn_player);

	distance_btw_players = sqrt(((p2y-p1y)**2)+((p2x-p1x)**2))

	return float(distance_btw_players * (own_moves/opp_moves))
	
class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.
    ********************  DO NOT MODIFY THIS CLASS  ********************
    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)
    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.
    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.
        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************
        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.
        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).
        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.
        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.
        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md
        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves
        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.
            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Compute scores to determine minimax decision
        best_score = float("-inf")
        actions = game.get_legal_moves()

        if not actions:
            best_move = None
        else:
            best_move = actions[0]

        for move in actions:
            min_value = self._min_value(game.forecast_move(move), depth - 1)
            if min_value > best_score:
                best_score = min_value
                best_move = move

        return best_move

    def _min_value(self, game, depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if self._terminal_state(game, depth):
            return self.score(game, self)

        score = float("inf")
        for move in game.get_legal_moves():
            max_value = self._max_value(game.forecast_move(move), depth - 1)
            score = min(score, max_value)

        return score

    def _max_value(self, game, depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if self._terminal_state(game, depth):
            return self.score(game, self)

        score = float("-inf")
        for move in game.get_legal_moves():
            min_value = self._min_value(game.forecast_move(move), depth - 1)
            score = max(score, min_value)

        return score

    def _terminal_state(self, game, depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Limiting the search depth for tree search
        if depth <= 0:
            return True

        # Player has not more legal moves left
        if not bool(game.get_legal_moves()):
            return True

        return False

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.
        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.
        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************
        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).
        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.
        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left
        depth = self.search_depth

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            depth = 0
            alpha = float("-inf")
            beta = float("inf")
            while self.time_left() >= self.TIMER_THRESHOLD:
                depth += 1
                best_move = self.alphabeta(game, depth, alpha, beta)

        except SearchTimeout:
            pass

        # Return the best move from the last completed search iteration
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.
        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md
        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        alpha : float
            Alpha limits the lower bound of search on minimizing layers
        beta : float
            Beta limits the upper bound of search on maximizing layers
        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves
        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.
            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Compute scores to determine minimax decision
        best_score = float("-inf")
        actions = game.get_legal_moves()

        if not actions:
            best_move = (-1, -1)
        else:
            best_move = actions[0]

        for move in actions:
            min_value = self._min_value(game.forecast_move(move), depth - 1, alpha, beta)
            if min_value > best_score:
                best_score = min_value
                best_move = move

            # Found the best possible solution at this node?
            if best_score >= beta:
                break
            alpha = max(alpha, best_score)

        return best_move

    def _terminal_state(self, game, depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Restrict depth of tree search
        if depth <= 0:
            return True

        # Player is out of moves
        if not bool(game.get_legal_moves()):
            return True

        return False

    def _min_value(self, game, depth, alpha, beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if self._terminal_state(game, depth):
            return self.score(game, self)

        score = float("inf")
        for move in game.get_legal_moves():
            max_value = self._max_value(game.forecast_move(move), depth - 1, alpha, beta)
            score = min(score, max_value)

            # Found the best possible solution at this node?
            if score <= alpha:
                return score

            beta = min(beta, score)

        return score

    def _max_value(self, game, depth, alpha, beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if self._terminal_state(game, depth):
            return self.score(game, self)

        score = float("-inf")
        for move in game.get_legal_moves():
            min_value = self._min_value(game.forecast_move(move), depth - 1, \
                alpha, beta)
            score = max(score, min_value)

            # Found the best possible solution at this node?
            if score >= beta:
                return score

            alpha = max(alpha, score)

        return score
