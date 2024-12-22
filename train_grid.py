import subprocess
import os
import argparse
import itertools

def main():
    parser = argparse.ArgumentParser(description="Run grid search for training models using train.py.")

    # Add all arguments from train.py
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, nargs='+', default=[10000], help="List of batch sizes for grid search")
    parser.add_argument("--lr", type=float, nargs='+', default=[0.001], help="List of learning rates for grid search")
    parser.add_argument("--dropout", type=float, nargs='+', default=[0.0], help="List of dropout rates for grid search")
    parser.add_argument("--optimizer", type=str, choices=["Adam", "AdamSF"], default="Adam", help="Optimizer (Adam or AdamSF)")
    parser.add_argument("--activation", type=str, choices=["relu", "mish"], default="relu", help="Activation function (relu or mish)")
    parser.add_argument("--use_batch_norm", type=bool, default=True, help="Batch normalization usage (True or False, default True).")
    parser.add_argument("--K_min", type=int, default=1, help="Minimum K value for random walks")
    parser.add_argument("--K_max", type=int, default=30, help="Maximum K value for random walks")
    parser.add_argument("--weights", type=str, default='', help="Path to file with model weights.")
    parser.add_argument("--device_id", type=int, default=0, help="Device ID")
    parser.add_argument("--alpha", type=float, nargs='+', default=[1], help="List of TD-learning parameters, avg 1/α steps.")
    parser.add_argument("--cube_size", type=int, default=4, help="Cube size for grid search")
    parser.add_argument("--cube_type", type=str, choices=["qtm", "all"], default="all", help="Cube type (qtm or all)")
    parser.add_argument("--hd1", type=int, nargs='+', default=[2000], help="List of sizes for the first hidden layer")
    parser.add_argument("--hd2", type=int, nargs='+', default=[1000], help="List of sizes for the second hidden layer")
    parser.add_argument("--nrd", type=int, nargs='+', default=[2], help="List of numbers of residual blocks")
    parser.add_argument("--repeats", type=int, default=1, help="Number of repetitions for each configuration")

    args = parser.parse_args()

    # Ensure the weights and logs directories exist
    os.makedirs("weights", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Generate all combinations of the hyperparameters
    hyperparameter_combinations = list(itertools.product(
        [args.cube_size],
        args.lr,
        args.batch_size,
        args.dropout,
        args.hd1,
        args.hd2,
        args.nrd,
        args.alpha
    ))

    total_models = len(hyperparameter_combinations) * args.repeats
    print(f"Starting train grid. Total models to train: {total_models}")

    # Iterate through the grid and run train.py for each combination
    for cube_size, lr, batch_size, dropout, hd1, hd2, nrd, alpha in hyperparameter_combinations:
        for repeat in range(args.repeats):
            # Build the command to run train.py
            command = [
                "python", "train.py",
                "--cube_size", str(cube_size),
                "--lr", str(lr),
                "--batch_size", str(batch_size),
                "--dropout", str(dropout),
                "--hd1", str(hd1),
                "--hd2", str(hd2),
                "--nrd", str(nrd),
                "--epochs", str(args.epochs),
                "--cube_type", args.cube_type,
                "--optimizer", args.optimizer,
                "--activation", args.activation,
                "--use_batch_norm", str(args.use_batch_norm),
                "--K_min", str(args.K_min),
                "--K_max", str(args.K_max),
                "--weights", args.weights,
                "--device_id", str(args.device_id),
                "--alpha", str(alpha)
            ]

            # Run the command
            subprocess.run(command)

    # Retrieve and print the last trained model IDs
    model_id_file = os.path.join("logs", "model_id.txt")
    if os.path.exists(model_id_file):
        with open(model_id_file, "r") as f:
            all_model_ids = f.readlines()
        last_model_ids = all_model_ids[-total_models:]
        print("Trained model IDs:")
        print("".join(last_model_ids))
    else:
        print("No model_id.txt file found. Unable to retrieve model IDs.")

if __name__ == "__main__":
    main()