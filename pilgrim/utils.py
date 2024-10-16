import torch
import json

def load_cube_data(cube_size, cube_type, device):
    """Load cube data based on cube size and type (qtm or all)."""
    file_path = f"generators/{cube_type}_cube{cube_size}.json"
    
    with open(file_path, 'rb') as f:
        data = json.load(f)
    
    actions = data["actions"]
    action_names = data["names"]
    
    return torch.tensor(actions, dtype=torch.int64, device=device), action_names

def generate_inverse_moves(moves):
    """Generate the inverse moves for a given list of moves."""
    inverse_moves = [0] * len(moves)
    for i, move in enumerate(moves):
        if "'" in move:  # It's an a_j'
            inverse_moves[i] = moves.index(move.replace("'", ""))
        else:  # It's an a_j
            inverse_moves[i] = moves.index(move + "'")
    return inverse_moves

def state2hash(states, hash_vec, batch_size=2**14):
    """Convert states to hashes."""
    num_batches = (states.size(0) + batch_size - 1) // batch_size
    result = torch.empty(states.size(0), dtype=torch.int64, device=states.device)
    
    for i in range(num_batches):
        batch = states[i * batch_size:(i + 1) * batch_size].to(torch.int64)
        batch_hash = torch.sum(hash_vec * batch, dim=1)
        result[i * batch_size:(i + 1) * batch_size] = batch_hash
    return result

# def get_unique_states(states, states_bad_hashed, hash_vec, batch_size=2**14):
#     """Filter unique states by removing duplicates based on hash."""
#     idx1 = torch.arange(states.size(0), dtype=torch.int64, device=states.device)
#     hashed = state2hash(states, hash_vec, batch_size)
#     mask1  = ~torch.isin(hashed, states_bad_hashed)
#     hashed = hashed[mask1]
#     hashed_sorted, idx2 = torch.sort(hashed)
#     mask2 = torch.concat((torch.tensor([True], device=states.device), hashed_sorted[1:] - hashed_sorted[:-1] > 0))
#     return states[mask1][idx2[mask2]], idx1[mask1][idx2[mask2]] 

# def get_unique_hashed_states_idx(states, states_bad_hashed):
#     """Filter unique hashed states by removing duplicates"""
#     mask1  = ~torch.isin(hashed, states_bad_hashed)
#     hashed = hashed[mask1]
#     hashed_sorted, idx2 = torch.sort(hashed)
#     mask2 = torch.concat((torch.tensor([True], device=states.device), hashed_sorted[1:] - hashed_sorted[:-1] > 0))
#     return idx1[mask1][idx2[mask2]] 

