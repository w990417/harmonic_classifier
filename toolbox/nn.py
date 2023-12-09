"""Module for models, loss functions, optimisers etc."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class CRNN(nn.Module):
    """CNN + LSTM model.
    CNN portion is for feature extraction.
    LSTM portion is for temporal modelling.

    Input samples are expected to be from the same track.
    """
    def __init__(self, cnn_dropout_rate=0.3, lstm_input_size=128, lstm_hidden_size=128):
        super(CRNN, self).__init__()
        
        self.cnn = nn.Sequential(
            # input shape: (batch_size, 1, 84, 168)
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(cnn_dropout_rate),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(cnn_dropout_rate),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.cnn_linear = nn.Linear(32 * 21 * 42, lstm_input_size)
        self.lstm_linear = nn.Linear(lstm_hidden_size, 52)
        self.softmax = nn.Softmax(dim=2)
        self.hidden_state = None
        self.lstm_input_size = lstm_input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_size)
 
    def forward(self, x):
        # forward pass CNN
        cnn_output = self.cnn(x)
        cnn_output = cnn_output.view(-1, 32 * 21 * 42)
        cnn_linear_output = self.cnn_linear(cnn_output)
        lstm_input = cnn_linear_output.view(cnn_output.shape[0], self.lstm_input_size)   
        # forward pass LSTM
        lstm_output, hidden_state = self.lstm(lstm_input)
        self.hidden_state = hidden_state
        lstm_linear_output = self.lstm_linear(lstm_output)
        lstm_linear_output = lstm_linear_output.view(-1, 4, 13) # reshape to match chord_frame's shape
        output = self.softmax(lstm_linear_output)
        return output
    
    def forward_validation(self, x):
        """Forward pass for validation set. Track_id is not provided."""
        # forward pass CNN
        cnn_output = self.cnn(x)
        cnn_output = cnn_output.view(-1, 32 * 21 * 42)
        cnn_linear_output = self.cnn_linear(cnn_output)
        lstm_input = cnn_linear_output.view(cnn_output.shape[0], self.lstm_input_size)
        # forward pass LSTM without hidden states
        lstm_output, _ = self.lstm(lstm_input)
        lstm_linear_output = self.lstm_linear(lstm_output)
        lstm_linear_output = lstm_linear_output.view(-1, 4, 13)
        output = self.softmax(lstm_linear_output)
        return output


class SoftmaxCNN(nn.Module):
    def __init__(self):
        super(SoftmaxCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 21 * 42, 128)
        self.fc2 = nn.Linear(128, 52)
        self.softmax = nn.Softmax(dim=2) # (batch_size, 4, 13)


    def forward(self, x):
        x = torch.relu(self.batchnorm1(self.conv1(x)))
        x = self.pool(x)
        x = torch.relu(self.batchnorm2(self.conv2(x)))
        x = self.pool(x)
        
        x = x.view(-1, 64 * 21 * 42)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(-1, 4, 13) # reshape to match chord_frame's shape
        x = self.softmax(x)
        return x
    
class DropoutCNN(nn.Module):
    def __init__(self, dropout_rate):
        super(DropoutCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 21 * 42, 256)
        self.fc2 = nn.Linear(256, 52)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid() # (batch_size, 4, 13)

    def forward(self, x):
        x = torch.relu(self.batchnorm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(x)
        x = torch.relu(self.batchnorm2(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(x)
        
        x = x.view(-1, 64 * 21 * 42)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.view(-1, 4, 13) # reshape to match chord_frame's shape
        x = self.sigmoid(x)
        return x
    


class OneHotLoss(nn.Module):
    """Weighted Binary Cross Entropy Loss
    Each row vector of the chord frame will be weighted by loss_weights and summed up to get the total loss.
    The loss for the correct note's index will be multiplied by wrong_penalty to encourage the model to predict 
    the correct note to minimise the loss.
    """
    def __init__(self, loss_weights, wrong_penalty):
        super(OneHotLoss, self).__init__()
        self.loss_weights = loss_weights
        self.wrong_penalty = wrong_penalty
        self.loss_fn = nn.BCELoss(reduction='none')

    def forward(self, pred, targ):
        # shape=(batch_size, 4, 13)
        assert pred.shape == targ.shape, f'Ensure "pred{pred.shape}" and "targ{targ.shape}" have the same shape'

        total_loss = 0.0
        loss = self.loss_fn(pred, targ)

        for i, weight in enumerate(self.loss_weights):
            row_loss = loss[:, i]
            
            row_loss = row_loss.sum(dim=1)
            row_loss = torch.where(pred[:,i].argmax() == targ[:,i].argmax(),
                                   row_loss, row_loss * self.wrong_penalty)
            row_loss = row_loss.sum()
            total_loss += weight * row_loss

        return total_loss


class ChordLoss(nn.Module):
    """Weighted Binary Cross Entropy Loss
    Each row vector of the chord frame will be weighted by loss_weights and summed up to get the total loss.
    The loss for the correct note's index for each row will be multiplied by wrong_penalty to encourage
    the model to predict the correct note to minimise the loss.
    """
    def __init__(self, loss_weights, argmax_penalty=2):
        super(ChordLoss, self).__init__()
        self.loss_weights = loss_weights
        self.penalty = argmax_penalty

    def forward(self, pred, targ):
        # shape=(batch_size, 4, 13)
        assert pred.shape == targ.shape, f'Ensure "pred{pred.shape}" and "targ{targ.shape}" have the same shape'

        loss_weight_map = torch.zeros_like(targ, dtype=torch.float32)
        loss_weight_map = loss_weight_map.to(pred.device)

        for i, weight in enumerate(self.loss_weights):
            loss_weight_map[:, i] = weight
        
        loss_weight_map = loss_weight_map.to(pred.device)
        loss = F.binary_cross_entropy(pred, targ, reduction='none')
        loss = loss * loss_weight_map
        loss = loss.sum(dim=2)

        for i in range(4):
            loss[:, i] = torch.where(pred[:,i].argmax() == targ[:,i].argmax(),
                                        loss[:, i], loss[:, i] * self.penalty)

        loss = loss.sum(dim=1)
        loss = loss.mean()
       
        return loss


def trainer(model, train_sampler, val_sampler, loss_fn, optim,
            num_epochs, save_interval, halt_epoch, save_path:str, device):

    print('Training model on', device)
    model.to(device)
    running_train_loss = []
    running_val_loss = []

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for epoch in range(num_epochs):
        
        # train set
        model.train()
        train_batch_count = 0
        epoch_loss = 0.0
        for inputs, targets in train_sampler:
            inputs = inputs.to(device)
            inputs = inputs.unsqueeze(1) # to (batch_size, 1, 84, 168)
            targets = targets.to(device)
            optim.zero_grad()
            preds = model(inputs)
            loss = loss_fn(preds, targets)
            epoch_loss += loss.item()

            loss.backward()
            optim.step()
            train_batch_count += 1

        # validate set
        model.eval()
        with torch.no_grad():
            val_batch_count = 0
            val_epoch_loss = 0.0
            for val_inputs, val_targets in val_sampler:
                val_inputs = val_inputs.to(device)
                val_inputs = val_inputs.unsqueeze(1)
                val_targets = val_targets.to(device)
                val_preds = model(val_inputs)
                val_loss = loss_fn(val_preds, val_targets)
                val_epoch_loss += val_loss.item()
                val_batch_count += 1

        # update loss every epoch
        running_train_loss.append(epoch_loss/train_batch_count)
        running_val_loss.append(val_epoch_loss/val_batch_count)
        print(f'Epoch {epoch+1} - Train Loss: {running_train_loss[-1]:.4f} - Val Loss: {running_val_loss[-1]:.4f}')
          
        # compare val loss every save_interval epochs
        # and save model parameters if val loss has decreased
        if (epoch+1) % save_interval == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f"epoch_{epoch+1}.pt"))


        # halt training if val loss has not decreased
        if len(running_val_loss) > halt_epoch:
            if  running_val_loss[-1] > max(running_val_loss[-halt_epoch:-1]):
                print(f'Validation loss has not decreased for {halt_epoch} epochs. Stopping training.')
                break

        train_sampler.shuffle()
        val_sampler.shuffle()

    # plot loss
    fig, ax = plt.subplots()
    ax.plot(range(1, len(running_train_loss)+1), running_train_loss, label='Training Loss')
    ax.plot(range(1, len(running_val_loss)+1), running_val_loss, label='Validation Loss')
    ax.set_title('Running Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average Batch Loss')
    plt.show()

    # save plot as png
    fig.savefig(os.path.join(save_path, 'loss.png'))
    # save final model parameters
    torch.save(model.state_dict(), os.path.join(save_path, 'final.pt'))

    return running_train_loss, running_val_loss
