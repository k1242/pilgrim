import torch

# Define constants for dtype and state size
dtype_int = torch.int32  # Define dtype_int since it's used but not defined in your original snippet
state_size = 6  # Define state_size if it's constant; otherwise, pass it as an argument where needed

def tensor2set(states: torch.Tensor) -> set:
    """
    Convert tensor of states to a set of tuples.

    :param states: Tensor of states.
    :return: Set of state tuples.
    """
    return {tuple(state.tolist()) for state in states}

def set2tensor(states: set) -> torch.Tensor:
    """
    Convert set of state tuples to a tensor.

    :param states: Set of state tuples.
    :return: Tensor of states.
    """
    return torch.tensor(list(states), dtype=dtype_int)

def tensor2list(states: torch.Tensor) -> list:
    """
    Convert tensor of states to a list of tuples.

    :param states: Tensor of states.
    :return: List of state tuples.
    """
    return [tuple(state.tolist()) for state in states]

def list2tensor(states: list) -> torch.Tensor:
    """
    Convert list of state tuples to a tensor.

    :param states: List of state tuples.
    :return: Tensor of states.
    """
    return torch.tensor(states, dtype=dtype_int)

def states2X(states: torch.Tensor) -> torch.Tensor:
    """
    Convert states to input tensor for the model.

    :param states: Tensor of states.
    :return: Input tensor for the model.
    """
    return torch.nn.functional.one_hot(states, num_classes=6).view(-1, state_size * 6).to(torch.float)
