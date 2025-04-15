import torch
import torch.nn as nn
import torch.optim as optim
import os

class Deep_Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class DNNTrainer:
    def __init__(self, model, lr):
        self.lr = lr
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # We use cross-entropy for classification.
        self.criterion = nn.CrossEntropyLoss()

    def train_step(self, states, target_actions):
        """
        Expects:
            - states: tensor of shape (batch_size, input_size)
            - target_actions: tensor of shape (batch_size,) containing target class indices (0, 1, or 2)
        """
        states = torch.tensor(states, dtype=torch.float)
        target_actions = torch.tensor(target_actions, dtype=torch.long)

        self.optimizer.zero_grad()
        outputs = self.model(states)  # shape: (batch_size, 3)
        loss = self.criterion(outputs, target_actions)
        loss.backward()
        self.optimizer.step()
        return loss.item()
