# Student agent: Add your own agent here
from gettext import find
from tracemalloc import start
from typing import no_type_check
from agents.agent import Agent
from store import register_agent
import random as rnd
from copy import deepcopy
import sys


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.cur_pos = None
        self.adv_pos = None
        self.chess_board = None
        self.max_step = None
        self.board_size = None
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        # Opposite Directions
        self.opposites = {0: 2, 1: 3, 2: 0, 3: 1}

    def check_endgame(self):
        # Union-Find
        father = dict()
        for r in range(self.board_size):
            for c in range(self.board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(self.board_size):
            for c in range(self.board_size):
                for dir, move in enumerate(
                    self.moves[1:3]
                ):  # Only check down and right
                    if self.chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(self.board_size):
            for c in range(self.board_size):
                find((r, c))
        p0_r = find(tuple(self.p0_pos))
        p1_r = find(tuple(self.p1_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            return False, p0_score, p1_score
        return True, p0_score, p1_score
    
    def check_valid_step(self, start_pos, end_pos, barrier_dir):
        #### This code is copied from 'world.py'
        # Endpoint already has barrier or is boarder
        is_reached = False

        if not (0 <= end_pos(0) < self.board_size and 0 <= end_pos(1) < self.board_size):
            return is_reached
        if not 0 <= barrier_dir <= 3:
            return is_reached
            
        r, c = end_pos
        if self.chess_board[r, c, barrier_dir]:
            return False
        if start_pos[0] == end_pos[0] and start_pos[1] == end_pos[1]:
            return True

        # Get position of the adversary
        adv_pos = self.cur_pos if self.turn else self.adv_pos

        # BFS
        state_queue = [(start_pos, 0)]
        visited = {tuple(start_pos)}
        while state_queue and not is_reached:
            cur_pos, cur_step = state_queue.pop(0)
            r, c = cur_pos
            if cur_step == self.max_step:
                break
            for dir, move in enumerate(self.moves):
                if self.chess_board[r, c, dir]:
                    continue

                next_pos = cur_pos + move
                if next_pos[0] == adv_pos[0] and next_pos[1] == adv_pos[1] or tuple(next_pos) in visited:
                    continue
                if next_pos[0] == end_pos[0] and next_pos[1] == end_pos[1]:
                    is_reached = True
                    break

                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))

        return is_reached

    def random_step(self, chess_board, my_pos, adv_pos, max_step):
        #### This code comes from 'random_agent.py'
    
        # Moves (Up, Right, Down, Left)
        ori_pos = deepcopy(my_pos)
        steps = rnd.randint(0, max_step + 1)

        # Random Walk
        for _ in range(steps):
            r, c = my_pos
            dir = rnd.randint(0, 4)
            m_r, m_c = self.moves[dir]
            my_pos = (r + m_r, c + m_c)

            # Special Case enclosed by Adversary
            k = 0
            while chess_board[r, c, dir] or my_pos == adv_pos:
                k += 1
                if k > 300:
                    break
                dir = rnd.randint(0, 4)
                m_r, m_c = self.moves[dir]
                my_pos = (r + m_r, c + m_c)

            if k > 300:
                my_pos = ori_pos
                break

        # Put Barrier
        dir = rnd.randint(0, 4)
        r, c = my_pos
        while chess_board[r, c, dir]:
            dir = rnd.randint(0, 4)

        return my_pos, dir

    
    def world_step(self, turn, step = None):
        ## This code comes from step in world.py 

        if not turn:
            cur_pos = self.cur_pos
            adv_pos = self.adv_pos
        else:
            cur_pos = self.adv_pos
            adv_pos = self.cur_pos

        seek_valid_step = True
        while seek_valid_step:
            # Run a random step for the player
            if step is None:
                next_pos, dir = self.random_step(
                    deepcopy(self.chess_board),
                    tuple(cur_pos),
                    tuple(adv_pos),
                    self.max_step)
            else:
                next_pos = tuple(step[0:2])
                dir = step[2]
            
            seek_valid_step = False 

            if not self.check_valid_step(cur_pos, next_pos, dir):
                seek_valid_step = True
        
        
        if not turn:
            self.cur_pos = next_pos
        else:
            self.adv_pos = next_pos
        # Set the barrier to True
        r, c = next_pos

        # Set barrier:
        self.chess_board[r, c, dir] = True
        # Set the opposite barrier to True
        move = self.moves[dir]
        self.chess_board[r + move[0], c + move[1], self.opposites[dir]] = True

        # Change turn
        next_turn = 1 - turn

        results = self.check_endgame()
        self.results_cache = results
        return results, next_turn
    
    def run(self, first_step):
        is_end, p0_score, p1_score, next_turn = self.world_step(0, first_step)
        while not is_end:
            is_end, p0_score, p1_score, next_turn = self.world_step(next_turn)
        return p0_score, p1_score

    def autoplay(self, first_step):
        runcount = 10 # To be implemented in a more clever way        
        p1_win_count = 0 # THis has to be array for each possible move
        for _ in range(runcount):
            p0_score, p1_score = self.run(first_step)
            if p0_score > p1_score:  # THis has to be put in array for each possible move
                p1_win_count += 2 
            elif p0_score == p1_score:  # Tie
                p1_win_count += 1
        return p1_win_count

    def generate_steps(self,my_pos,max_step):
        # Generate list of possible steps in diamond shape around our current position
        # with maximum distance of max_step
        steps = []
        numSteps = 0
        for x in range(-max_step, max_step + 1):
            for y in range(-max_step, max_step + 1):
                if abs(x)+abs(y) <= max_step:
                    for dir in range(0,4):
                        steps.append([x,y,dir])
        
        return steps
        

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        possible_steps = self.generate_steps(my_pos,max_step)

        scores = [0] * len(possible_steps)
        for i, step in enumerate(possible_steps):

            # Filter invalid steps
            if self.check_valid_step(my_pos, tuple(step[0:2]), step[2]):
                continue

            self.cur_pos = deepcopy(my_pos)
            self.adv_pos = deepcopy(adv_pos)
            self.max_step = max_step
            self.chess_board = deepcopy(chess_board)
            self.board_size = len(chess_board) ## TODO: check it works

            scores[i] = self.autoplay(step)

        # dummy returnis_reached = False
        max_score = max(scores)
        indices = [index for index, item in enumerate(scores) if item == max_score]
        best_index = rnd.randint(0, len(indices) - 1)
        best_step = possible_steps[indices[best_index]]

        return tuple(best_step[0:2]), best_step[2]
