import torch
from collections import deque
from tqdm import tqdm
from .utils import state2hash, get_unique_states
from .model import batch_process


class Searcher:
    def __init__(self, model, all_moves, V0, device=None, verbose=0):
        self.model = model.to(device)
        self.all_moves = all_moves
        self.V0 = V0
        self.n_gens = all_moves.size(0)
        self.state_size = all_moves.size(1)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hash_vec = torch.randint(0, int(1e15), (self.state_size,), device=self.device)
        self.verbose = verbose
    
    def get_neighbors(self, states, batch_size=2**14):
        """Return neighboring states for each state in the batch."""
        total_size = states.size(0)
        neighbors = torch.empty(total_size, self.n_gens, self.state_size, device=self.device, dtype=states.dtype)
        for i in range(0, total_size, batch_size):
            batch_states = states[i:i + batch_size]
            neighbors[i:i + batch_size] = torch.gather(
                batch_states.unsqueeze(1).expand(batch_states.size(0), self.n_gens, self.state_size), 
                2, 
                self.all_moves.unsqueeze(0).expand(batch_states.size(0), self.n_gens, self.state_size)
            )
        return neighbors
    
    def do_greedy_step(self, states, value_last, states_bad_hashed, B=1000):
        """Perform a greedy step to find the best neighbors."""
        idx0 = torch.arange(states.size(0), device=self.device).repeat_interleave(self.n_gens)
        moves = torch.arange(self.n_gens, device=self.device).repeat(states.size(0))
        
#         print(f'debug (do_greedy_step): {states.shape = }')
        neighbors = self.get_neighbors(states).flatten(end_dim=1)
#         print(f'debug (do_greedy_step): {neighbors.shape = }')
        neighbors, idx1 = get_unique_states(neighbors, states_bad_hashed, self.hash_vec)
        
        # Predict values for the neighboring states
        value = self.pred_d(neighbors)[0]
        idx2 = torch.argsort(value)[:B]
        
        return neighbors[idx2], value[idx2], moves[idx1[idx2]], idx0[idx1[idx2]]
    
#     def check_stagnation(self, states_log):
#         """Check if the process is in a stagnation state."""
#         return torch.isin(torch.concat(states_log[2:]), torch.concat(states_log[:2])).all().item()
    def check_stagnation(self, states_log):
        """Check if the process is in a stagnation state."""
        return torch.isin(torch.concat(list(states_log)[2:]), torch.concat(list(states_log)[:2])).all().item()

    
    def get_solution(self, state, B=2**12, num_steps=200, num_attempts=10):
        """Main solution-finding loop that attempts to solve the cube."""
        states_bad_hashed = torch.tensor([], dtype=torch.int64, device=self.device)
        for J in range(num_attempts):
            states = state.unsqueeze(0).clone()
            tree_move = torch.zeros((num_steps, B), dtype=torch.int64) #why not on the device?
            tree_idx = torch.zeros((num_steps, B), dtype=torch.int64)  #why not on the device?
            y_pred = torch.tensor([0], dtype=torch.float64, device=self.device)
            states_hash_log = deque(maxlen=4)
            
            if self.verbose:
                pbar = tqdm(range(num_steps))
            else:
                pbar = range(num_steps)
            for j in pbar:
#                 print(f'------------- step {j} -------------------')
#                 print(f'debug (get_solution): step {j} before, {states.shape = }')
                states, y_pred, moves, idx = self.do_greedy_step(states, y_pred, states_bad_hashed, B)
#                 print(f'debug (get_solution): step {j} after, {states.shape = }')
                if self.verbose:
                    pbar.set_description(f"  y_min = {y_pred.min().item():.1f}, y_mean = {y_pred.mean().item():.1f}, y_max = {y_pred.max().item():.1f}")
                states_hash_log.append(state2hash(states, self.hash_vec))
#                 print(f'debug (get_solution): states_hash_log size = {[T.shape for T in states_hash_log if T is not 0]}')
                leaves_num = states.size(0)
                tree_move[j, :leaves_num] = moves
                tree_idx[j, :leaves_num] = idx

                if (states == self.V0).all(dim=1).any():
                    break
                elif (j > 3 and self.check_stagnation(states_hash_log)):
#                     print(f'debug: stagnation at step {j}')
                    states_bad_hashed = torch.concat((states_bad_hashed, torch.concat(list(states_hash_log))))
                    states_bad_hashed = torch.unique(states_bad_hashed)
                    break

            if (states == self.V0).all(dim=1).any():
                break
        
        if not (states == self.V0).all(dim=1).any():
            return None, J
        
        # Reverse the tree to reconstruct the path
        tree_idx, tree_move = tree_idx[:j+1].flip((0,)), tree_move[:j+1].flip((0,))
        V0_pos = torch.nonzero((states == self.V0).all(dim=1), as_tuple=True)[0].item()
        
        # Construct the path
        path = [tree_idx[0, V0_pos].item()]
        for k in range(1, j+1):
            path.append(tree_idx[k, path[-1]].item())
        
        moves_seq = torch.tensor([tree_move[k, path[k-1]] if k > 0 else tree_move[k, V0_pos] for k in range(j+1)], dtype=torch.int64)
        return moves_seq.flip((0,)), J
    
    def pred_d(self, states):
        """Predict values for states using the model."""
        return batch_process(self.model, states, self.device, 2**14).unsqueeze(0)
