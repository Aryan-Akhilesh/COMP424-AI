# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
#import sys
import numpy as np
from copy import deepcopy
import time
from queue import PriorityQueue


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

        # Some simple code to help you with timing. Consider checking 
        # time_taken during your search and breaking with the best answer
        # so far when it nears 2 seconds.
        start_time = time.time()
        time_taken = time.time() - start_time

        steps_taken = 0
        dist_bt_adversary = self.distance_to_adversary(my_pos, adv_pos)

        possible_moves = self.possible_moves(chess_board, my_pos, adv_pos, steps_taken, max_step)

        print("My AI's turn took ", time_taken, "seconds.")

        # dummy return
        return my_pos, self.dir_map["u"]

    # Heuristic function 1
    # We are going to try to trap the opponent by moving closer to them
    # We calculate the Manhattan distance for it since our agent cannot move diagonally
    # Also because it is less computationally expensive than Euclidean
    def distance_to_adversary(self, my_pos, adv_pos):
        x1, y1 = my_pos
        x2, y2 = adv_pos
        shortest_distance = np.abs(x2-x1) + np.abs(y2-y1)
        return shortest_distance

    # Heuristic function 2
    # We don't want to enter a box that has 3 walls
    def num_walls(self, chess_board, my_pos):
        num_walls = 0
        x1, y1 = my_pos
        for direction in self.dir_map.values():
            if chess_board[x1, y1, direction]:
                num_walls += 1
        return num_walls

    # Evaluation function
    def eval(self, chess_board, my_pos, adv_pos):
        h1 = self.distance_to_adversary(my_pos, adv_pos)
        h2 = self.num_walls(chess_board, my_pos)
        result = h2 + 0.7*h1
        return result

    def bfs(self, chess_board, my_pos, adv_pos, steps_taken, max_step):
        frontier = self.find_frontier(chess_board, my_pos, adv_pos)
        steps_taken += 1
        visited = []
        while steps_taken <= max_step:
            new_pos = frontier.get()[1]
            new_frontier = self.find_frontier(chess_board, new_pos, adv_pos)
            # Condition to check if I have reached to a desired box, ie the opponent is in an adjacent box
            steps_taken += 1

    def find_frontier(self, chess_board, my_pos, adv_pos):
        frontier = PriorityQueue()
        x1, y1 = my_pos
        for direction in self.dir_map.values():
            if not chess_board[x1, y1, direction]:
                if direction == 0 and (x1-1, y1) != adv_pos:
                    value = self.eval(chess_board, (x1-1, y1), adv_pos)
                    frontier.put((value, (x1-1, y1)))
                elif direction == 1 and (x1, y1+1) != adv_pos:
                    value = self.eval(chess_board, (x1, y1+1), adv_pos)
                    frontier.put((value, (x1, y1+1)))
                elif direction == 2 and (x1+1, y1) != adv_pos:
                    value = self.eval(chess_board, (x1+1, y1), adv_pos)
                    frontier.put((value, (x1+1, y1)))
                elif direction == 3 and (x1,y1-1) != adv_pos:
                    value = self.eval(chess_board, (x1, y1-1), adv_pos)
                    frontier.put((value, (x1, y1-1)))

        return frontier






