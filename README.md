# Snake-Game-Ai-Using-Deep-Neural-Networks

# 🐍 Intelligent Snake Game using Deep Neural Network (DNN)

A reinforcement learning-based Snake Game where the agent is trained using a Deep Neural Network (DNN) and guided with heuristic knowledge to efficiently reach food and survive longer.

---

## 📌 Project Overview

This project implements a smarter Snake Game agent using a custom Deep Neural Network (DNN). The goal of the snake is to move through a grid environment, avoid collisions, and eat food to grow in length. Unlike traditional Q-learning or DDQN methods, this project uses DNN combined with intelligent exploration logic to improve gameplay efficiency and learning behavior.

---

## 📚 Existing Work

- Based on a Deep Q-Network (DQN) and Double DQN (DDQN) approach.
- Used ε-greedy exploration and two networks (main and target).
- Snake behavior was often unstable, moved in circles, and failed to reach food consistently.
- Learning was slower, and the training setup was complex.

---

## 🔧 Modifications Done

- Replaced DDQN with a simplified **Deep Neural Network (DNN)**.
- Added **heuristic guidance** using Manhattan distance to help the snake move toward food.
- Introduced **safe move checks** to prevent collisions with walls or the body.
- Improved snake **speed and responsiveness**.
- Simplified the overall architecture to focus on meaningful actions and learning.

---

## 🚀 Project Files & Their Roles

| File             | Description |
|------------------|-------------|
| `agent.py`       | Contains the DNN model, prediction logic, and action selection with food-guided strategy. |
| `model.py`       | Defines the neural network architecture using PyTorch. |
| `game.py`        | Implements the Snake game environment (food, movement, collisions). |
| `train.py`       | Runs the game loop and connects the agent to the environment. |
| `utils.py`       | Helper functions (if needed). |

---

## 🧠 How It Works

1. **State Representation:**  
   The snake observes its environment – its direction, position of food, and potential dangers.

2. **Action Selection:**  
   - DNN predicts action values.
   - A heuristic checks if the predicted action moves toward food and avoids danger.
   - Final action is chosen based on prediction + guidance.

3. **Reward Mechanism:**  
   - +10 for eating food  
   - -10 for collision  
   - -0.1 for each step (to avoid looping)

4. **Learning:**  
   - Stores experiences in memory.
   - Trains the model using batches of past experiences with `train_short_memory` and `train_long_memory`.

---

## 📊 Results Comparison

| Metric            | Existing (DDQN) | Modified (DNN + Heuristic) |
|-------------------|----------------|-----------------------------|
| Avg. Score        | Low            | Higher & Consistent         |
| Food Reached      | Rarely         | Frequently                  |
| Deaths per Game   | High           | Reduced                     |
| Training Time     | Longer         | Faster                      |
| Complexity        | High           | Simpler & Efficient         |

---

## 🔁 Future Work

- Add support for real-time training visualization.
- Implement full DNN-RL training with curriculum learning.
- Deploy the model as a playable web app.

---

## 📌 References

- [Deep Reinforcement Learning - PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Snake Game with AI - Python Projects on GitHub](https://github.com/yenchenlin/DeepLearningFlappyBird)
- Sutton, R. & Barto, A. - *Reinforcement Learning: An Introduction*

---

# Snake Game AI using Deep Neural Networks (DNN)

This repository contains an implementation of a Snake game where the snake is controlled by an artificial intelligence (AI) agent. The agent uses a Deep Neural Network (DNN) along with heuristic guidance to make decisions about its movements. Unlike traditional reinforcement learning approaches that require extensive exploration through reward-based training (such as DDQN), this project employs a simplified DNN structure that is well-suited for supervised or imitation learning tasks.

---

## Project Overview

The goal of this project is to develop an intelligent Snake game agent that can:
- Efficiently navigate the game environment.
- Avoid collisions (with boundaries or itself).
- Consistently reach food items by guiding its movement with both learned behavior and heuristic adjustments.

### Key Features
- **Deep Neural Network (DNN):**  
  A feed-forward network trained to predict the next move (go straight, turn right, or turn left) based on an 11-dimensional state vector representing the snake's current situation.

- **Heuristic Guidance:**  
  In addition to the DNN’s prediction, a heuristic mechanism is used to simulate potential moves and select the action that minimizes the Manhattan distance to the food while ensuring safe moves.

- **Increased Game Speed:**  
  The game speed has been increased to test real-time performance and responsiveness of the AI.

- **Modular Design:**  
  The project is organized into several modules to separate the AI logic (agent and model) from the game environment (game) and visualization (helper).

---

## Project Structure

- **model.py:**  
  Defines the DNN architecture using PyTorch and includes a trainer class for supervised training via CrossEntropyLoss.  
  - `Deep_Net`: Implements a multi-layer perceptron with ReLU activations.
  - `DNNTrainer`: Provides a routine for updating model weights based on state-action pairs.

- **agent.py:**  
  Contains the AI agent logic which:
  - Extracts the current game state (an 11-dimensional vector).
  - Uses the DNN to predict an action.
  - Merges the DNN prediction with a heuristic-based approach to steer the snake toward food while avoiding collisions.
  - Provides an interface (`get_action`) that is called each game step.

- **game.py:**  
  Implements the Snake game environment using Pygame:
  - Handles drawing, snake movement, collision detection, food placement, and game over conditions.
  - Provides the `play_step` method used in each iteration of the game loop.

- **helper.py:**  
  Contains utility functions for plotting game progress (score and mean score) during training or testing.

- **snake_game_human.py:**  
  A variant of the game module that allows for manual control of the snake. Useful for collecting demonstration data or testing the game without AI.

---

## How It Works

1. **State Representation:**
   - The state consists of 11 elements including:
     - Danger indicators (for straight, right, and left directions).
     - Current snake direction (one-hot encoded).
     - Relative position of the food (left, right, up, down).

2. **Action Selection:**
   - **DNN Prediction:**  
     The current state is fed to the DNN to obtain logits corresponding to the three possible actions.
   - **Heuristic Guidance:**  
     The agent simulates all three moves and computes the Manhattan distance to the food for each potential move. This helps bias the decision toward a safe and food-approaching action.
   - **Hybrid Strategy:**  
     A combination of the DNN and the heuristic is used. The final move is the one that minimizes the distance to the food while ensuring safety (i.e., no collisions).

3. **Game Loop:**
   - In every game iteration, the agent's `get_action` function is called.
   - The game environment then updates based on the action, checks for collisions, moves the snake, and redraws the frame.
   - If food is eaten, the snake grows and new food is placed.

4. **Training:**
   - While the current implementation is designed for supervision (imitation learning), it can be extended further.
   - The DNN can be trained on a dataset of state-action pairs collected from human demonstrations or using other forms of supervision.

---

## Requirements

- Python 3.x
- [PyTorch](https://pytorch.org/)
- [Pygame](https://www.pygame.org/)
- [Matplotlib](https://matplotlib.org/)

---

## How to Run

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/LalithSiramdasu/Snake-Game-Ai-Using-Deep-Neural-Networks.git
   cd snake-dnn-ai

Run the Game:

For AI-controlled mode, run:
python agent.py


For human-controlled mode, run:
python snake_game_human.py


