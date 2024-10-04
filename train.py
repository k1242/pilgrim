import argparse
import torch
from pilgrim import Trainer, Pilgrim
from pilgrim import count_parameters, generate_inverse_moves, load_cube_data

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train Pilgrim Model")
    
    # Training and architecture hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=10000, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.05, help="Dropout")
    parser.add_argument("--K_min", type=int, default=1, help="Minimum K value for random walks")
    parser.add_argument("--K_max", type=int, default=30, help="Maximum K value for random walks")
    
    # Cube parameters
    parser.add_argument("--cube_size", type=int, default=4, help="Cube size (e.g., 4 for 4x4x4 cube)")
    parser.add_argument("--cube_type", type=str, choices=["qtm", "all"], default="all", help="Cube type (qtm or all)")
    
    # Model architecture parameters
    parser.add_argument("--hd1", type=int, default=2000, help="Size of the first hidden layer")
    parser.add_argument("--hd2", type=int, default=1000, help="Size of the second hidden layer (0 means no second layer)")
    parser.add_argument("--nrd", type=int, default=2, help="Number of residual blocks (0 means no residual blocks)")
    
    args = parser.parse_args()

    # Set device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Start training with {device}.")

    # Load cube data (moves and names)
    all_moves, move_names = load_cube_data(args.cube_size, args.cube_type)

    # Derive important cube parameters from the loaded data
    n_gens = all_moves.size(0)  # Number of moves
    state_size = all_moves.size(1)  # Size of the state representation
    face_size = state_size // 6  # Size of one face of the cube

    # Generate inverse moves
    inverse_moves = torch.tensor(generate_inverse_moves(move_names), dtype=torch.int64, device=device)
    V0 = torch.arange(6, dtype=torch.int8, device=device).repeat_interleave(face_size)

    # Infer model mode based on hd1, hd2, and nrd
    if args.hd2 == 0 and args.nrd == 0:
        mode = "MLP1"
    elif args.hd2 > 0 and args.nrd == 0:
        mode = "MLP2"
    elif args.hd2 > 0 and args.nrd > 0:
        mode = "MLP2RB"
    else:
        raise ValueError("Invalid combination of hd1, hd2, and nrd.")

    # Initialize the Pilgrim model
    model = Pilgrim(
        state_size=state_size,
        hd1=args.hd1,
        hd2=args.hd2,
        nrd=args.nrd,
        dropout_rate=args.dropout
    ).to(device)

    # Calculate the number of model parameters
    num_parameters = count_parameters(model)
    param_million = round(num_parameters / 1_000_000, 2)  # Get the number of parameters in millions

    # Create the training name based on mode, hidden layers, residual blocks, and number of parameters
    name = f"cube{args.cube_size}_{args.cube_type}_{mode}_{args.hd1}_{args.hd2}_{args.nrd}_{param_million:.2f}M"

    # Create the trainer
    trainer = Trainer(
        net=model,
        num_epochs=args.epochs,
        device=device,
        batch_size=args.batch_size,
        lr=args.lr,
        name=name,
        K_min=args.K_min,
        K_max=args.K_max,
        all_moves=all_moves,
        inverse_moves=inverse_moves,
        V0=V0
    )

    # Display model information
    print(f"Model Mode: {mode}")
    print(f"Model Name: {name}")
    print(f"Model has {num_parameters} parameters")

    # Start the training process
    trainer.run()

if __name__ == "__main__":
    main()
