import torch
import random
import numpy as np
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import Deep_Net, DNNTrainer
from helper import plot

LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 80  # initial exploration factor
        self.model = Deep_Net(11, 256, 3)
        self.trainer = DNNTrainer(self.model, lr=LR)
        # This list can be used to store demonstration or training data.
        self.demo_data = []  

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),

            # Current direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location relative to head
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y   # food down
        ]
        return np.array(state, dtype=int)

    def heuristic_action(self, game):
        """
        Computes a heuristic move by simulating the three possible actions (straight, right, left)
        and choosing the one that minimizes the Manhattan distance to the food. Only safe moves (that
        do not result in a collision) are considered.
        """
        head = game.snake[0]
        food = game.food

        # Order of directions in clockwise order for snake movement:
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        current_idx = clock_wise.index(game.direction)
        candidate_moves = []  # list of tuples: (move, new_direction, new_head, distance)

        # Define the three possible moves
        possible_actions = {
            0: 0,   # Straight: no change
            1: 1,   # Right turn: increment index by 1
            2: -1   # Left turn: decrement index by 1
        }

        for move, delta in possible_actions.items():
            new_idx = (current_idx + delta) % 4
            new_direction = clock_wise[new_idx]

            # Simulate new head position based on new_direction
            x, y = head.x, head.y
            if new_direction == Direction.RIGHT:
                x += BLOCK_SIZE
            elif new_direction == Direction.LEFT:
                x -= BLOCK_SIZE
            elif new_direction == Direction.DOWN:
                y += BLOCK_SIZE
            elif new_direction == Direction.UP:
                y -= BLOCK_SIZE
            new_head = Point(x, y)

            # Check if the move is safe
            if not game.is_collision(new_head):
                # Compute Manhattan distance from new head to food
                distance = abs(food.x - new_head.x) + abs(food.y - new_head.y)
                candidate_moves.append((move, new_direction, new_head, distance))

        if candidate_moves:
            # Choose the move that minimizes the distance to food
            best_move = min(candidate_moves, key=lambda item: item[3])[0]
            # Convert the move index (0, 1, or 2) into one-hot encoding.
            action = [0, 0, 0]
            action[best_move] = 1
            return action
        else:
            # If no safe candidate, return a random valid move.
            action = [0, 0, 0]
            action[random.randint(0, 2)] = 1
            return action

    def get_action(self, state, game):
        """
        Returns a one-hot action:
          [1, 0, 0] -> straight,
          [0, 1, 0] -> right,
          [0, 0, 1] -> left.
        Uses a combination of ε–greedy exploration, DNN prediction, and a heuristic steering towards food.
        """
        # Update epsilon (exploration decreases with more games)
        self.epsilon = max(10, 80 - self.n_games)

        # First choose an action via ε–greedy randomness.
        if random.randint(0, 200) < self.epsilon:
            dnn_action = [0, 0, 0]
            dnn_action[random.randint(0, 2)] = 1
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            if len(state_tensor.shape) == 1:
                state_tensor = state_tensor.unsqueeze(0)
            prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item()
            dnn_action = [0, 0, 0]
            dnn_action[move] = 1

        # Compute the heuristic action (which directs the snake toward food).
        heuristic_action = self.heuristic_action(game)

        # For each move, simulate the resulting head position.
        def simulate_move(action):
            clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
            idx = clock_wise.index(game.direction)
            if action == [1, 0, 0]:
                new_idx = idx
            elif action == [0, 1, 0]:
                new_idx = (idx + 1) % 4
            else:  # action == [0, 0, 1]:
                new_idx = (idx - 1) % 4
            new_direction = clock_wise[new_idx]
            x, y = game.snake[0].x, game.snake[0].y
            if new_direction == Direction.RIGHT:
                x += BLOCK_SIZE
            elif new_direction == Direction.LEFT:
                x -= BLOCK_SIZE
            elif new_direction == Direction.DOWN:
                y += BLOCK_SIZE
            elif new_direction == Direction.UP:
                y -= BLOCK_SIZE
            return Point(x, y)

        # Simulate head positions for both candidate actions.
        head_dnn = simulate_move(dnn_action)
        head_heuristic = simulate_move(heuristic_action)

        # Calculate Manhattan distances from the candidate new positions to the food.
        food = game.food
        dnn_distance = abs(food.x - head_dnn.x) + abs(food.y - head_dnn.y)
        heuristic_distance = abs(food.x - head_heuristic.x) + abs(food.y - head_heuristic.y)

        # Choose the action that yields the lower distance to food.
        # Only choose an action if it is safe.
        if not game.is_collision(head_dnn) and not game.is_collision(head_heuristic):
            final_move = heuristic_action if heuristic_distance < dnn_distance else dnn_action
        elif not game.is_collision(head_heuristic):
            final_move = heuristic_action
        elif not game.is_collision(head_dnn):
            final_move = dnn_action
        else:
            # fallback to a random safe move if both candidate moves are unsafe.
            final_move = [0, 0, 0]
            final_move[random.randint(0, 2)] = 1

        return final_move

    def train_supervised(self, states, actions):
        """
        Trains on a batch of demonstration data.
        Actions are integer class indices:
          0 for [1,0,0], 1 for [0,1,0], and 2 for [0,0,1].
        """
        loss = self.trainer.train_step(states, actions)
        return loss

def run():
    """
    Runs the game session using the hybrid agent that favors moves bringing the snake
    closer to the food. The game is updated at high speed.
    """
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        state_old = agent.get_state(game)
        # The get_action method now also requires the game object for the heuristic component.
        final_move = agent.get_action(state_old, game)
        reward, game_over, score = game.play_step(final_move)

        if game_over:
            game.reset()
            agent.n_games += 1

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    run()
