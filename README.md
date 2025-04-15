# Snake-Game-Ai-Using-Deep-Neural-Networks

# Snake Game AI using Deep Neural Networks (DNN)

A reinforcement learning-based Snake Game where the agent is trained using a Deep Neural Network (DNN) combined with heuristic guidance to efficiently reach food and survive longer. Unlike traditional reinforcement learning approaches (such as DDQN) that require extensive exploration and complex architectures, this project employs a simplified DNN structure that is well-suited for supervised or imitation learning tasks.

---

## 📌 Project Overview

This project implements a smarter Snake Game agent using a custom Deep Neural Network (DNN). The goal of the snake is to move through a grid environment, avoid collisions, and eat food to grow in length. The agent leverages both learned behavior and heuristic adjustments to:
- Efficiently navigate the game environment.
- Avoid collisions (with boundaries or itself).
- Consistently reach food items.

The DNN predicts the next move based on an 11-dimensional state vector, while a heuristic component ensures the action taken moves the snake toward the food with safety in mind. Additionally, the game speed has been increased to test real-time performance.

---

## 📚 Existing Work

- The initial approach was based on a Deep Q-Network (DQN) and Double DQN (DDQN).
- It used ε-greedy exploration with two networks (main and target) to learn from reward signals.
- The agent often exhibited unstable behavior, moving in circles or in one direction without efficiently reaching food.
- The learning process was slow and the training setup was complex.

---

## 🔧 Modifications Done

- **Replaced DDQN with a simplified Deep Neural Network (DNN):**  
  The project now uses a single DNN, simplifying the architecture and training process.

- **Integrated Heuristic Guidance:**  
  A heuristic component simulates potential moves and evaluates them based on the Manhattan distance to the food. This helps steer the snake toward food while avoiding unsafe moves.

- **Introduced Safe Move Checks:**  
  Before executing an action, the agent verifies that the move does not result in a collision with walls or the snake's body.

- **Enhanced Game Speed:**  
  The game's speed has been increased to better stress-test the real-time performance of the AI agent.

- **Simplified Overall Architecture:**  
  The modifications remove the need for multiple networks and complex reward-based learning, focusing instead on a combination of supervised learning and rule-based guidance.

---

## 🚀 Project Files & Their Roles

| File                | Description |
|---------------------|-------------|
| **`model.py`**      | Defines the DNN architecture using PyTorch and includes a trainer (`DNNTrainer`) for supervised training via CrossEntropyLoss. |
| **`agent.py`**      | Contains the AI agent logic that extracts the current game state, predicts an action using the DNN, and merges the prediction with heuristic guidance to select the final move. |
| **`game.py`**       | Implements the Snake game environment using Pygame, handling snake movement, collision detection, food placement, and game over conditions. |
| **`helper.py`**     | Provides utility functions for plotting game progress (score and mean score) during training or testing. |
| **`snake_game_human.py`** | A variant of the game that allows manual control of the snake for demonstration or data collection purposes. |

---

## 🧠 How It Works

1. **State Representation:**
   - The game state is represented as an 11-dimensional vector which includes:
     - Danger indicators for the snake's head in three directions (straight, right, left).
     - The current direction of the snake (one-hot encoded).
     - The relative position of the food (left, right, up, down).

2. **Action Selection:**
   - **DNN Prediction:**  
     The current state is passed through the DNN to yield logits corresponding to the three possible actions: go straight, turn right, or turn left.
   - **Heuristic Guidance:**  
     The agent simulates each potential move and computes the Manhattan distance from the snake's future head position to the food. This guides the agent toward actions that are both safe and goal-driven.
   - **Hybrid Strategy:**  
     The final action is selected by comparing the DNN prediction with the heuristic outcome, choosing the move that minimizes the distance to the food while avoiding collisions.

3. **Game Loop:**
   - At each iteration:
     - The agent extracts the current state.
     - An action is determined via the hybrid strategy.
     - The game environment (`game.py`) processes the action, updating the snake's position, checking for collisions, placing new food if needed, and rendering the frame.
   - The game awards positive rewards for eating food and penalizes for collisions, although the current implementation is optimized for imitation learning rather than reward-based RL training.

4. **Training:**
   - While the setup is designed for supervised or imitation learning (training on state-action pairs), it is extendable to further reinforcement learning techniques if desired.

---

## 📊 Results Comparison

| Metric            | Existing (DDQN) | Modified (DNN + Heuristic) |
|-------------------|-----------------|----------------------------|
| **Avg. Score**        | Low             | Higher & Consistent         |
| **Food Reached**      | Rarely          | Frequently                  |
| **Deaths per Game**   | High            | Reduced                     |
| **Training Time**     | Longer          | Faster                      |
| **Complexity**        | High            | Simpler & More Efficient    |

---

## 🔁 Future Work

- Integrate real-time training visualization.
- Extend the system with full DNN-RL training using curriculum learning.
- Deploy the model as a web app or interactive demonstration.

---

## 📌 References

- [Deep Reinforcement Learning - PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Snake Game with AI - Python Projects on GitHub](https://github.com/yenchenlin/DeepLearningFlappyBird)
- Sutton, R. & Barto, A. — *Reinforcement Learning: An Introduction*
- Pygame Documentation: [https://www.pygame.org/docs/](https://www.pygame.org/docs/)

---

## Requirements

- Python 3.x
- [PyTorch](https://pytorch.org/)
- [Pygame](https://www.pygame.org/)
- [Matplotlib](https://matplotlib.org/)

---

Below is the extended "How to Run" section for your README that details the entire process, from cloning the repository to launching both AI-controlled and human-controlled versions of the game.

---

## How to Run

Follow these steps to run the project on your local machine:

### 1. Clone the Repository

Open your terminal or command prompt and run:

```shell
git clone https://github.com/LalithSiramdasu/Snake-Game-Ai-Using-Deep-Neural-Networks.git
cd Snake-Game-Ai-Using-Deep-Neural-Networks
```

### 2. Set Up a Virtual Environment (Optional but Recommended)

Create and activate a virtual environment to keep project dependencies isolated:

- **On macOS/Linux:**

  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

- **On Windows:**

  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```

### 3. Install Dependencies

Install the required libraries using the provided `requirements.txt` file. Make sure you have pip updated:

```bash
pip install -r requirements.txt
```

*If you do not have a `requirements.txt` file yet, create one with the following lines:*

```
pygame
torch
matplotlib
```

### 4. Run the Game

#### For AI-Controlled Mode

This mode runs the game with the DNN agent controlling the snake. In your terminal, execute:

```bash
python agent.py
```

The agent will start playing the Snake game automatically. You should see the game window open, and the snake will begin to move, make decisions using the DNN and the heuristic strategy, and update the score accordingly.

#### For Human-Controlled Mode

If you want to manually play and test the game (or collect demonstration data):

```bash
python snake_game_human.py
```

In this mode, use the arrow keys to control the snake’s movement. The game window will respond to your keystrokes, and you can observe the snake’s behavior in real time.

### 5. Observing Game Progress

While the game is running in AI mode, a plotting window (or an IPython display) will show the score and average score progression over time. This gives you visual feedback on the agent’s performance.

### 6. Stopping the Game

- **To exit the game window:**  
  Click the close button on the window.

- **To exit the terminal session:**  
  Press `Ctrl + C` in the terminal.
