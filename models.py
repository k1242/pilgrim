import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.1) -> None:
        """
        Residual Block with two fully connected layers, batch normalization, ReLU activation, and dropout.

        :param hidden_dim: Dimensionality of the hidden layers.
        :param dropout_rate: Dropout rate used between layers. Default is 0.1.
        """
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Residual Block.

        :param x: Input tensor of shape (batch_size, hidden_dim).
        :return: Output tensor of the same shape as input.
        """
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class Pilgrim(nn.Module):
    def __init__(self, input_dim: int, hidden_dim1: int = 5000, hidden_dim2: int = 1000, num_residual_blocks: int = 4, output_dim: int = 1, dropout_rate: float = 0.1) -> None:
        """
        Pilgrim Model with input, hidden, and output layers, including residual blocks.

        :param input_dim: Dimensionality of the input data.
        :param hidden_dim1: Dimensionality of the first hidden layer. Default is 5000.
        :param hidden_dim2: Dimensionality of the second hidden layer. Default is 1000.
        :param num_residual_blocks: Number of residual blocks. Default is 4.
        :param output_dim: Dimensionality of the output data. Default is 1.
        :param dropout_rate: Dropout rate used between layers. Default is 0.1.
        """
        super(Pilgrim, self).__init__()
        self.hd1 = hidden_dim1
        self.hd2 = hidden_dim2
        self.nrd = num_residual_blocks
        
        self.input_layer = nn.Linear(input_dim, hidden_dim1)
        self.hidden_layer = nn.Linear(hidden_dim1, hidden_dim2)
        self.residual_blocks = nn.ModuleList([ResidualBlock(hidden_dim2, dropout_rate) for _ in range(num_residual_blocks)])
        self.output_layer = nn.Linear(hidden_dim2, output_dim)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Pilgrim model.

        :param x: Input tensor of shape (batch_size, input_dim).
        :return: Output tensor of shape (batch_size, output_dim).
        """
        x = self.relu(self.input_layer(x))
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.relu(self.hidden_layer(x))
        x = self.bn2(x)
        x = self.dropout(x)
        for layer in self.residual_blocks:
            x = layer(x)
        x = self.output_layer(x)
        return x

def count_parameters(model: nn.Module) -> int:
    """
    Count the trainable parameters in a model.

    :param model: A PyTorch model.
    :return: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
