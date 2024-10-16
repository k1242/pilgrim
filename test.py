import argparse
import torch
import os
import json
import time
from pilgrim import Pilgrim, Searcher
from pilgrim import count_parameters, generate_inverse_moves, load_cube_data

def main():
    parser = argparse.ArgumentParser(description="Test Pilgrim Model")
    parser.add_argument("--cube_size", type=int, required=True, help="Cube size")
    parser.add_argument("--cube_type", type=str, choices=["qtm", "all"], required=True, help="Cube type (qtm or all)")
    parser.add_argument("--tests", type=str, default='', help="Path to the tests.")
    parser.add_argument("--weights", type=str, required=True, help="Path to the model weights")
    parser.add_argument("--B", type=int, default=4096, help="Beam size")
    parser.add_argument("--num_attempts", type=int, default=2, help="Number of allowed restarts.")
    parser.add_argument("--num_steps", type=int, default=200, help="Number of allowed steps in one beam search run.")
    parser.add_argument("--tests_num", type=int, default=10, help="Number of tests to run")
    parser.add_argument("--device_id", type=int, default=0, help="Device ID")
    parser.add_argument("--verbose", type=int, default=0, help="Use tqdm if verbose > 0.")
    
    args = parser.parse_args()
    # Load model info
    log_dir = "logs"
    with open(f"{log_dir}/model_{'_'.join(args.weights.split('/')[-1].split('_')[:-1])}.json", "r") as json_file:
        info = json.load(json_file)
    
    # Set device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", args.device_id)
#     device = torch.device("cpu")
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(f"[{timestamp}] Start testing with {device}.")

    # Load cube data (moves and names)
    all_moves, move_names = load_cube_data(args.cube_size, args.cube_type, device)

    # Derive important cube parameters from the loaded data
    n_gens = all_moves.size(0)  # Number of moves
    state_size = all_moves.size(1)  # Size of the state representation
    face_size = state_size // 6  # Size of one face of the cube
    
    # Generate inverse moves
    inverse_moves = torch.tensor(generate_inverse_moves(move_names), dtype=torch.int64, device=device)
    V0 = torch.arange(6, dtype=torch.int8, device=device).repeat_interleave(face_size)

    # Load model and weights
    model = Pilgrim(state_size=state_size, 
                    hd1=info['hd1'], hd2=info['hd2'], nrd=info['nrd'], 
                    activation_function=info.get('activation', 'relu'), 
                    use_batch_norm=info.get('use_batch_norm', True))
    model.load_state_dict(torch.load(args.weights, weights_only=False, map_location=device))
    model.eval()
    
    # Load test dataset
    if len(args.tests) == 0:
        tests_path = f"datasets/{args.cube_type}_cube{args.cube_size}.pt"
    else:
        tests_path = args.tests
    tests = torch.load(tests_path, weights_only=False, map_location=device)
    tests = tests[:args.tests_num]
    tests = tests.to(device)

    # Initialize Searcher object
    searcher = Searcher(model=model, all_moves=all_moves, V0=V0, device=device, verbose=args.verbose)
    
    # Extract epoch information from weights file name
    epoch = args.weights.split('_')[-1][:-4]
    
    # Prepare log file
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/test_{info['model_name']}_{info['model_id']}_{epoch}_B{args.B}.json"

    results = []
    total_length = 0
    t1 = time.time()
    for i, state in enumerate(tests, start=0):
        moves, attempts = searcher.get_solution(state, B=args.B, num_steps=args.num_steps, num_attempts=args.num_attempts)
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        
        if moves is not None:
            solution_length = len(moves)
            total_length += solution_length
            
            result = {
                "test_num": i,
                "solution_length": solution_length,
                "attempts": attempts + 1,
                "moves": moves.tolist()
            }
            
            # Print solution length for each solved cube
            print(f"[{timestamp}] Solution {i}: Length = {solution_length}")
        else:
            # If no solution is found
            result = {
                "test_num": i,
                "solution_length": None,
                "attempts": None,
                "moves": None
            }
            print(f"[{timestamp}] Solution {i} not found")
        
        results.append(result)

        # Append new result to the log file
        with open(log_file, 'w') as f:
            json.dump(results, f, indent=4)

    t2 = time.time()

    # Calculate average solution length
    solved_results = [r for r in results if r["solution_length"] is not None]
    avg_length = total_length / len(solved_results) if solved_results else 0

    # Print completion message with average solution length
    print(f"Test completed in {(t2 - t1):.2f}s.")
    print(f"Average solution length: {avg_length:.2f}.")
    print(f"Solved {len(solved_results)}/{args.tests_num} cubes.")
    print(f"Results saved to {log_file}.")

if __name__ == "__main__":
    main()