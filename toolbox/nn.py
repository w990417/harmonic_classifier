"""Module for models, loss functions, optimisers etc."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class ProtoCNN(nn.Module):
    def __init__(self):
        super(ProtoCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 21 * 42, 128)
        self.fc2 = nn.Linear(128, 52)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 21 * 42)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = x.view(-1, 4, 13) # reshape to match chord_frame's shape
        return x
    


class SoftmaxCNN(nn.Module):
    def __init__(self):
        super(SoftmaxCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 21 * 42, 128)
        self.batchnorm3 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 52)
        self.softmax = nn.Softmax(dim=2) # (batch_size, 4, 13)


    def forward(self, x):
        x = self.batchnorm1(self.conv1(x))
        x = self.pool(torch.relu(x))
        x = self.batchnorm2(self.conv2(x))
        x = self.pool(torch.relu(x))
        x = x.view(-1, 64 * 21 * 42)
        
        x = self.batchnorm3(self.fc1(x))
        x = torch.relu(x)
        x = self.fc2(x)
        x = x.view(-1, 4, 13) # reshape to match chord_frame's shape
        x = self.softmax(x)
        return x

# define loss function
class ChordLoss(nn.Module):
    """Weighted Binary Cross Entropy Loss
    Each row vector of the chord frame will be weighted by loss_weights and summed up to get the total loss.
    """
    def __init__(self, loss_weights):
        super(ChordLoss, self).__init__()
        self.loss_weights = loss_weights

    def forward(self, pred, targ):
        assert pred.shape == targ.shape, 'Ensure "pred" and "targ" have the same shape'

        loss = 0.0
        # calculate BCE loss for each row of chord_frame tensors
        for i, weight in enumerate(self.loss_weights):
            row_loss = F.binary_cross_entropy(pred[i], targ[i], reduction='mean')

            # scale row losses according to chord frame weights
            scaled_loss = row_loss * weight

            loss += scaled_loss
            
        return loss


def trainer(model, sampler, loss_fn, optim, num_epochs, save_every:int, save_dir:str, device):
    try:
        _trainer(model, sampler, loss_fn, optim, num_epochs, save_every, save_dir, device)
    except KeyboardInterrupt:
        print('Training interrupted. Model parameters saved to disk.')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(model.state_dict(), os.path.join(save_dir, "interrupted.pt"))


def _trainer(model, sampler, loss_fn, optim, num_epochs, save_every:int, save_path:str, device):
    """Trains CNN model and saves model parameters to disk every save_every epochs."""
    print('Training model on', device)
    model.to(device)
    list_loss = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batch_count = 0
        model.train()

        for inputs, targets in sampler:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # zero the parameter gradients
            optim.zero_grad()

            # reshape inputs to (batch_size, 1, 84, 168)
            inputs = inputs.unsqueeze(1)
            preds = model(inputs)
            loss = loss_fn(preds, targets)
            epoch_loss += loss.item()
            loss.backward()
            optim.step()
            batch_count += 1
            
        # update loss every epoch
        print(f"Epoch {epoch+1} loss: {epoch_loss:.4f}")
        list_loss.append(epoch_loss)
          
        # save model parameters to disk
        if (epoch + 1) % save_every == 0:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(model.state_dict(), os.path.join(save_path, f"epoch_{epoch+1}.pt"))

    # plot loss
    fig, ax = plt.subplots()
    ax.plot(range(1, num_epochs+1), list_loss)
    ax.set_title('Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    plt.show()

    # save plot as png
    fig.savefig(os.path.join(save_path, 'loss.png'))

    return list_loss
