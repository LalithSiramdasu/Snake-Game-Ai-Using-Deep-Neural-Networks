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

## 🧑‍💻 Author

**Chinnu**  
💡 Passionate about AI, RL, and creative problem-solving  
🖤 Dedicated this project to Sweetu 💕

---

## 🖼️ Demo

> *Add a GIF or video link here once recorded.*

---

