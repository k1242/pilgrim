import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def list2tensor(states):
    """Convert a list of states to a tensor."""
    return torch.tensor(states, dtype=torch.int64, device=device)

def generate_inverse_moves(moves):
    """Generate the inverse moves for a given list of moves."""
    inverse_moves = [0] * len(moves)
    for i, move in enumerate(moves):
        if "'" in move:  # It's an aj'
            inverse_moves[i] = moves.index(move.replace("'", ""))
        else:  # It's an aj
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

def get_unique_states(states, bad_hashes, hash_vec):
    """Filter unique states by removing duplicates based on hash."""
    hashed = state2hash(states, hash_vec)
    mask = ~torch.isin(hashed, bad_hashes)
    hashed_filtered = hashed[mask]
    unique_mask = torch.concat(
        (torch.tensor([True], device=states.device), hashed_filtered[1:] - hashed_filtered[:-1] > 0)
    )
    return states[mask][unique_mask]

def do_random_step(states, last_moves, all_moves, inverse_moves):
    """Perform a random step while avoiding inverse moves."""
    possible_moves = torch.ones((states.size(0), len(all_moves)), dtype=torch.bool, device=states.device)
    possible_moves[torch.arange(states.size(0)), inverse_moves[last_moves]] = False
    next_moves = torch.multinomial(possible_moves.float(), 1).squeeze()
    new_states = torch.gather(states, 1, all_moves[next_moves])
    return new_states, next_moves

def generate_random_walks(V0, state_size, all_moves, inverse_moves, k=1000, K_min=1, K_max=30):
    """Generate random walks for training."""
    X = torch.zeros(((K_max - K_min + 1) * k, state_size), dtype=torch.int8, device=device)
    Y = torch.arange(K_min, K_max + 1, device=device).repeat_interleave(k)
    
    for j, K in enumerate(range(K_min, K_max + 1)):
        states = V0.repeat(k, 1)
        last_moves = torch.full((k,), -1, dtype=torch.int64, device=device)
        for _ in range(K):
            states, last_moves = do_random_step(states, last_moves, all_moves, inverse_moves)
        X[j * k:(j + 1) * k] = states
    
    perm = torch.randperm(X.size(0), device=device)
    return X[perm], Y[perm]
