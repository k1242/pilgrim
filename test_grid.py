import subprocess
import os
import argparse
import json

# Function to find and read the JSON file for a given model_id and return its contents
def load_model_info(model_id, log_dir='logs'):
    # Search for a file in log_dir that ends with _{model_id}.json
    for filename in os.listdir(log_dir):
        if filename.endswith(f"_{model_id}.json"):
            file_path = os.path.join(log_dir, filename)
            
            # Open and read the JSON file
            with open(file_path, 'r') as file:
                model_info = json.load(file)
            
            return model_info
    
    print(f"Error: Log file for model_id {model_id} not found in {log_dir}.")
    return None

def main():
    parser = argparse.ArgumentParser(description="Run Pilgrim model tests for specified model_id, epochs, and beam sizes.")
    
    parser.add_argument("--cube_size", type=int, default=4, 
                        help="Cube size (default is 4x4x4).")
    parser.add_argument("--cube_type", type=str, choices=["qtm", "all"], default="all", 
                        help="Cube move set: 'qtm' or 'all' moves (default: 'all').")
    parser.add_argument("--tests", type=str, default="datasets/santa_cube4.pt", 
                        help="Path to the test dataset (default: 'datasets/santa_cube4.pt').")
    parser.add_argument("--model_ids", nargs='+', required=True, 
                        help="List of model_id(s) to test.")
    parser.add_argument("--epochs", nargs='+', type=int, required=True, 
                        help="List of epoch numbers to use for testing.")
    parser.add_argument("--B", nargs='+', type=int, required=True, 
                        help="List of beam sizes to use in beam search.")
    parser.add_argument("--num_attempts", type=int, default=2, 
                        help="Number of allowed restarts.")
    parser.add_argument("--num_steps", type=int, default=200, 
                        help="Number of allowed steps in one beam search run.")
    parser.add_argument("--tests_num", type=int, default=10, 
                        help="Number of test cases to run (default is 10).")
    parser.add_argument("--device_id", type=int, default=0, 
                        help="Device ID")
    parser.add_argument("--verbose", type=int, default=0, 
                        help="Use tqdm if verbose > 0.")
    
    args = parser.parse_args()
    
    # Ensure the directory exists
    if not os.path.exists("weights"):
        print(f"Error: Weights directory 'weights' not found.")
        return
    if not os.path.exists("logs"):
        print(f"Error: Logs directory 'logs' not found.")
        return
    
    # Loop through each combination of model_id, epoch, and beam size
    for model_id in args.model_ids:
        # Load model info from the logs to get model_name
        model_info = load_model_info(model_id)
        if model_info is None:
            print(f"Skipping model_id {model_id} due to missing model log file.")
            continue
        
        # Extract model_name from the model info
        model_name = model_info.get("model_name", None)
        if model_name is None:
            print(f"Error: model_name not found in the log file for model_id {model_id}.")
            continue
        
        for epoch in args.epochs:
            for B in args.B:
                # Construct the weights file name based on model_id and epoch
                weights_file = f"{model_name}_{model_id}_e{epoch:05d}.pth"
                weights_path = os.path.join("weights", weights_file)
                
                # Check if the weights file exists before proceeding
                if not os.path.exists(weights_path):
                    print(f"Error: Weights file {weights_path} not found. Skipping this combination.")
                    continue
                
                # Build the command to execute the test
                command = [
                    "python", "test.py",
                    "--cube_size", str(args.cube_size),
                    "--cube_type", args.cube_type,
                    "--weights", weights_path,
                    "--tests_num", str(args.tests_num),
                    "--B", str(B),
                    "--tests", str(args.tests),
                    "--verbose", str(args.verbose),
                    "--device_id", str(args.device_id), 
                    "--num_steps", str(args.num_steps),
                    "--num_attempts", str(args.num_attempts)
                ]
                
                # Log the test execution details
                print(f"Running test for {model_name} (model_id={model_id}, epoch={epoch}, B={B})")
                
                # Execute the test command
                subprocess.run(command)

if __name__ == "__main__":
    main()