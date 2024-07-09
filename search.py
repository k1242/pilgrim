import torch

class BeamSearch:
    def __init__(self, model: torch.nn.Module, state: torch.Tensor, num_steps: int, generators: torch.Tensor, device: torch.device) -> None:
        """
        Initialize the BeamSearch class.

        :param model: Model to use for predictions.
        :param state: Initial state tensor.
        :param num_steps: Number of steps to perform in the search.
        :param generators: Generators to create new states.
        :param device: Device to perform computations (e.g., 'cuda', 'cpu').
        """
        self.model = model
        self.state = state
        self.num_steps = num_steps
        self.generators = generators
        self.n_gens = generators.size(0)
        self.state_size = generators.size(1)
        self.device = device
        self.hash_vec = torch.randint(0, 1_000_000_000_000, (self.state_size,))
        self.target_val = None

    def get_unique_states(self, states: torch.Tensor) -> torch.Tensor:
        """
        Get unique states by hashing.

        :param states: Tensor of states.
        :return: Tensor of unique states.
        """
        hashed = torch.sum(self.hash_vec * states, dim=1)
        hashed_sorted, idx = torch.sort(hashed)
        mask = torch.cat((torch.tensor([True]), hashed_sorted[1:] - hashed_sorted[:-1] > 0))
        return states[idx][mask]

    def get_neighbors(self, states: torch.Tensor) -> torch.Tensor:
        """
        Get neighboring states.

        :param states: Tensor of states.
        :return: Tensor of neighboring states.
        """
        return torch.gather(
            states.unsqueeze(1).expand(states.size(0), self.n_gens, self.state_size), 
            2, 
            self.generators.unsqueeze(0).expand(states.size(0), self.n_gens, self.state_size))

    def states_to_input(self, states: torch.Tensor) -> torch.Tensor:
        """
        Convert states to input tensor.

        :param states: Tensor of states.
        :return: Input tensor for the model.
        """
        return torch.nn.functional.one_hot(states, num_classes=6).view(-1, self.state_size * 6).to(torch.float)

    def batch_predict(self, model: torch.nn.Module, data: torch.Tensor, device: torch.device, batch_size: int) -> torch.Tensor:
        """
        Perform batch prediction.

        :param model: Model to use for predictions.
        :param data: Input data tensor.
        :param device: Device to perform computations (e.g., 'cuda', 'cpu').
        :param batch_size: Batch size for predictions.
        :return: Predictions tensor.
        """
        model.eval()
        model.to(device)

        n_samples = data.shape[0]
        outputs = []

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            batch = data[start:end].to(device)

            with torch.no_grad():
                batch_output = model(batch).flatten()

            outputs.append(batch_output)

        final_output = torch.cat(outputs, dim=0)
        return final_output

    def predict_values(self, states: torch.Tensor) -> torch.Tensor:
        """
        Predict values for given states.

        :param states: Tensor of states.
        :return: Predicted values tensor.
        """
        return self.batch_predict(self.model, self.states_to_input(states), self.device, 4096).cpu()

    def predict_clipped_values(self, states: torch.Tensor) -> torch.Tensor:
        """
        Predict clipped values for given states.

        :param states: Tensor of states.
        :return: Clipped predicted values tensor.
        """
        return torch.clip(self.predict_values(states) - self.target_val, 0, torch.inf)

    def do_greedy_step(self, states: torch.Tensor, B: int = 1000) -> torch.Tensor:
        """
        Perform a greedy step in the search.

        :param states: Tensor of current states.
        :param B: Beam size.
        :return: Tensor of new states after the greedy step.
        """
        neighbors = self.get_neighbors(states).flatten(end_dim=1)
        neighbors = self.get_unique_states(neighbors)
        y_pred = self.predict_clipped_values(neighbors)
        idx = torch.argsort(y_pred)[:B]
        return neighbors[idx]

    def search(self, V0: torch.Tensor, B: int = 1000) -> int:
        """
        Perform the beam search.

        :param V0: Target state tensor.
        :param B: Beam size.
        :return: Number of steps to reach the target state, or -1 if not found.
        """
        self.target_val = self.predict_values(V0.unsqueeze(0)).item()
        states = self.state.clone()
        for j in range(self.num_steps):
            states = self.do_greedy_step(states, B)
            if (states == V0).all(dim=1).any():
                return j
        return -1
