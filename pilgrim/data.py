import json
import torch

def load_cube_data(cube_size, cube_type):
    """Load cube data based on cube size and type (qtm or all)."""
    file_path = f"generators/{cube_type}_cube{cube_size}.json"
    
    with open(file_path, 'rb') as f:
        data = json.load(f)
    
    actions = data["actions"]
    action_names = data["names"]
    
    return torch.tensor(actions, dtype=torch.int64), action_names
