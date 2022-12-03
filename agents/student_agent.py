# Student agent: Add your own agent here
#from gettext import find
#from tracemalloc import start
#from typing import no_type_check
from agents.agent import Agent
from store import register_agent
import random as rnd
from copy import deepcopy
import sys
from time import time

## ----------------------- INFORMATION ABOUT IMPLEMENTATION ----------------------- ##

# We have copied some functions from world.py and altered them to fit in our code. 
# The purpose of doing that is to be able to make simulations of the game forward 
# in time but without affecting the original game.
# The functions we have copied are:
#       - check_endgame()
#       - check_valid_step()
#       - random_step() (comes from random_agent.py)
#       - step(), here it's named student_world_step() to avoid confusion
# 
# The function that is called 


@register_agent("student_agent")
class StudentAgent(Agent):
    #rnd.seed(16)
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
        self.turn = None                                # Whose turn it is. 0: our player. 1: opponent's turn
        self.cur_pos = None                             # Current position of our player
        self.adv_pos = None                             # Current position of opponent
        self.chess_board = None                         # Array containing the chessbord state in terms of barriers
        self.max_step = None                            # Maximum number of steps to move in game
        self.board_size = None                          # Board size
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1)) # Possible moves
        self.opposites = {0: 2, 1: 3, 2: 0, 3: 1}       # Opposite Directions

        # Created for use in our implementation of the agent:
        self.iter = 0                                   # Number of moves made so far by our agent
        self.MCTS_time = 0                              # Time spent on MCTS
        self.num_precalculated_steps = 0                # Number of precalculated states
        self.num_sim_MCTS = 10                          # Number of simulations to run per possible move
        self.num_sim_initial = 2                        # Number of simulations to run per possible move in pre-calculations
        

    def check_endgame(self, our_pos, adv_pos):
        #### This code is copied from 'world.py' and altered to fit in our code
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
        p0_r = find(tuple(our_pos))
        p1_r = find(tuple(adv_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            return False, p0_score, p1_score
        return True, p0_score, p1_score
    
    def check_valid_step(self, start_pos, end_pos, barrier_dir):
        #### This code is copied from 'world.py' and altered to fit in our code
        
        is_reached = False

        if not (0 <= end_pos[0] < self.board_size and 0 <= end_pos[1] < self.board_size):
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

                next_pos = (cur_pos[0] + move[0], cur_pos[1] + move[1])
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
        steps = rnd.randint(0, max_step)

        # Random Walk
        for _ in range(steps):
            r, c = my_pos
            dir = rnd.randint(0, 3)
            m_r, m_c = self.moves[dir]
            my_pos = (r + m_r, c + m_c)

            # Special Case enclosed by Adversary
            k = 0
            while chess_board[r, c, dir] or my_pos == adv_pos:
                k += 1
                if k > 300:
                    break
                dir = rnd.randint(0, 3)
                m_r, m_c = self.moves[dir]
                my_pos = (r + m_r, c + m_c)

            if k > 300:
                my_pos = ori_pos
                break

        # Put Barrier
        dir = rnd.randint(0, 3)
        r, c = my_pos
        while chess_board[r, c, dir]:
            dir = rnd.randint(0, 3)

        return my_pos, dir

    
    def student_world_step(self, turn, step = None):
        ## This code comes from 'step' function in world.py 
        # The function modifies the self.chess_board value to include the new step.
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

        results = self.check_endgame(self.cur_pos, self.adv_pos)
        #self.results_cache = results
        return results[0], results[1:3], next_turn
    
    def run(self, first_step):
        # This function runs the steps in the simulations of the MCTS
        is_end, scores, next_turn = self.student_world_step(0, first_step)
        while not is_end:
            is_end, scores, next_turn = self.student_world_step(next_turn)
 
        return scores[0], scores[1]


    def MCTS_simulation(self, runcount, possible_steps, chess_board, my_pos, adv_pos):
        # This function runs the MCTS simulations.
        
        # Overwrite self with inputted values to reset
        self.chess_board = deepcopy(chess_board)
        self.adv_pos =  deepcopy(adv_pos)
        self.cur_pos =  deepcopy(my_pos)

        # Check which steps in 'possible_steps'-input are valid and save them. 
        # Only perform MCTS with them 
        valid_steps = []
        for i, s in enumerate(possible_steps):
            if self.check_valid_step(my_pos, tuple(s[0:2]), s[2]):
                valid_steps.append(possible_steps[i])

        # Number of valid steps
        numValidSteps = len(valid_steps)
        # Array to hold the scores of each step
        score_eval = [0]*numValidSteps

        # Take out steps
        steps = deepcopy(valid_steps)
        for j in range(1, runcount + 1): # Loop over instances/simulations of the game
            for i,s in enumerate(steps): # Loop over possible steps.
                # Reset values
                self.turn = 0
                self.chess_board = deepcopy(chess_board)
                self.cur_pos = deepcopy(my_pos)
                self.adv_pos = deepcopy(adv_pos)
                
                # Run one instance of the game 
                p0_score, _ = self.run(s)
                score_eval[i] = p0_score+score_eval[i] # Save absolute scores
            
            # Begin to eliminate bad steps
            if j>=1 and len(score_eval) > 0: 
                # find median and remove bad steps
                scores_sorted = deepcopy(score_eval)
                scores_sorted.sort()
                score_median = scores_sorted[len(scores_sorted)//4] # remove 3/4 of the steps

                # Find the bad steps (worse evaluation than median score)
                bad_score_idx = [i if score < score_median else -1 for i, score in enumerate(score_eval)]
                # Remove from list of potential steps, so that we loop over a smaller array
                steps = [step for step, bad_score_index in zip(steps, bad_score_idx) if bad_score_index < 0]
                # Also remove from the score evaluation list
                score_eval = [score for score, bad_score_index in zip(score_eval, bad_score_idx) if bad_score_index < 0]
            
            # If we have too few steps left, break out of the loop
            if len(steps) <= 2:
                break

        # Maximum score
        max_score = max(score_eval)
        # Take out indices of maximum score
        indices = [index for index, item in enumerate(score_eval) if item == max_score]
        # Choose a random index among the maximum scores and return the step. 
        random_best_index = rnd.randint(0, len(indices) - 1)
        best_step = steps[indices[random_best_index]]
        
        return best_step 


    def initialStep(self, possible_steps, chess_board, my_pos, adv_pos, numStep,num_autoplay):
        # This function performs the pre-calculations

        # Overwrite self with the chess_board
        self.chess_board = deepcopy(chess_board)

        best_steps = [] # Array for saving the steps to take.
        for i in range(numStep): # Loop over number of pre-calculated steps
            print("Pre-calculate step number {}".format(i+1))
            # Generate possible steps
            possible_steps = self.generate_steps(my_pos,self.max_step)
            # Find the best initial step using MCTS
            best_steps.append(self.MCTS_simulation(num_autoplay,possible_steps, chess_board, my_pos, adv_pos))
            # Reset values
            self.chess_board = deepcopy(chess_board)
            self.cur_pos = deepcopy(my_pos)
            self.adv_pos = deepcopy(adv_pos)
            # Take step with our player
            is_end, scores, next_turn= self.student_world_step(0, best_steps[i]) # alters self.cur_pos (our position)
            # See if we reached the end of the game. If so, break
            results = self.check_endgame(self.cur_pos, self.adv_pos)
            if results[0]:
                break
            # Take random step with the other player
            is_end, scores, next_turn = self.student_world_step(next_turn)  # alters self.adv (their position)
            # set new chess_board and positions!!
            chess_board = deepcopy(self.chess_board) # Update chess_board value
            my_pos = deepcopy(self.cur_pos)
            adv_pos = deepcopy(self.adv_pos)

        return best_steps

     

    def generate_steps(self,my_pos,max_step):
        # Generate list of possible end positions in diamond shape around our current position
        # with maximum distance of max_step
        steps = []
        numSteps = 0
        for x in range(-max_step, max_step + 1):
            for y in range(-max_step, max_step + 1):
                if abs(x)+abs(y) <= max_step:
                    for dir in range(0,4):
                        new_x = my_pos[0] + x
                        new_y = my_pos[1] + y
                        steps.append([new_x, new_y, dir])
        
        return steps

    def choose_num_steps(self,num_steps):
        # This function initializes the number of simulations in the MCTS as well as the number of pre-calculated steps.
        if num_steps == 0:
            if self.board_size < 8:
                self.num_precalculated_steps = 2
                self.num_sim_MCTS = 20
                self.num_sim_initial = 10
            elif self.board_size == 8:
                self.num_precalculated_steps = 2
                self.num_sim_MCTS = 20
                self.num_sim_initial = 10
            elif self.board_size == 9:
                self.num_precalculated_steps = 3
                self.num_sim_MCTS = 10
                self.num_sim_initial = 20
            elif self.board_size == 10:
                self.num_precalculated_steps = 5
                self.num_sim_MCTS = 10
                self.num_sim_initial = 10
            elif self.board_size == 11:
                self.num_precalculated_steps = 5
                self.num_sim_MCTS = 2
                self.num_sim_initial = 4
            elif self.board_size == 12:
                self.num_precalculated_steps = 3
                self.num_sim_MCTS = 1
                self.num_sim_initial = 2
        else:
            # Here, we make the parameters dependent on the time spent.
            if self.MCTS_time > 1.9 and self.num_sim_MCTS>1:
                self.num_sim_MCTS = round(self.num_sim_MCTS/2)
            elif self.MCTS_time < 0.3 and self.num_sim_MCTS <50:
                self.num_sim_MCTS = self.num_sim_MCTS*2
        

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        This function returns the step of the agent.
        """

        # Save input values in self
        self.max_step = max_step
        self.board_size = len(chess_board)

        # Generate all possible steps from current position
        # Inside the functions, it is evaluated whether they are valid
        possible_steps = self.generate_steps(my_pos,max_step)

        # Get parameters for initial steps and MCTS
        self.choose_num_steps(self.iter)

        # If we are in the very first iteration, we perform the pre-calculations.
        if self.iter == 0:
            # Save statistics (hidden)
            #with open('data/threads_player1_boardsize_{}.txt'.format(self.board_size), 'w') as filehandle:
            #        filehandle.write(("Threads\n"))

            # Perform pre-calculations
            self.first_steps_list = self.initialStep(possible_steps, chess_board, my_pos, adv_pos,self.num_precalculated_steps,self.num_sim_initial)
        # Number of first steps
        len_list = len(self.first_steps_list)
        # If we still have pre-calculated steps left, take them.
        if self.iter < len_list:
            best_step = self.first_steps_list[self.iter]    # Take step (this value is returned)
            self.iter = self.iter +1                        # Increment iteration counter
        
        # If the pre-calculated step is invalid, return it and let the base code make os take a random step
        if self.iter < len_list and self.check_valid_step(my_pos, tuple(best_step[0:2]), best_step[2]):
            print("*** Precalculated move was invalid ***")
            return tuple(best_step[0:2]), best_step[2]
        # If we don't have anymore pre-calculated steps left, perform MCTS
        else:
            start_time = time()
            best_step = self.MCTS_simulation(self.num_sim_MCTS,possible_steps, chess_board, my_pos, adv_pos)
            self.MCTS_time = time() - start_time
        # Save statistics (hidden)
        # with open('data/threads_player1_boardsize_{}.txt'.format(self.board_size), 'a') as filehandle:
        #    filehandle.write(("{}\n".format(self.num_sim_MCTS)))

        return tuple(best_step[0:2]), best_step[2]
