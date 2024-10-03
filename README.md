<!-- # Pilgrim Library

Pilgrim Library is a Python library for efficient state space search and model training using PyTorch. It includes tools for building, training, and utilizing neural networks, particularly suited for solving combinatorial puzzles.

## Features

- **Pilgrim Model**: Neural network with residual blocks for complex state representations.
- **BeamSearch**: Efficient search strategy for exploring state spaces and finding solutions.
- **Training Tools**: Classes for model training and evaluation.
- **Utility Functions**: Helper functions for data manipulation and transformation.
 -->


 # Pilgrim: Random Walk Based Neural Network Training

**Pilgrim** is a Python library designed for training neural networks using random walks. The model architecture can be customized with different layers and residual blocks, and the cube moves are derived from either the **quarter-turn metric (QTM)** or **all moves** (including half turns). This library supports training on various cube sizes (e.g., 3x3x3, 4x4x4) and allows for flexible architecture and hyperparameter configurations.


## Getting Started

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/pilgrim.git
    cd pilgrim
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Data Setup

You will need to populate the `generators/` folder with `.json` files that define the cube moves and states for the desired cube size and type. The files should follow this structure:
- `generators/qtm_cube4.json`
- `generators/all_cube4.json`

These JSON files should contain two keys:
- `actions`: A list of tensors representing the moves.
- `names`: A list of move names.

### Running the Training Script

You can run the `train.py` script to train a model on cube-based data. The model architecture is flexible, allowing different hidden layer sizes and residual blocks to be used.

#### Basic Usage

```bash
python train.py --hd1 2000 --hd2 1000 --nrd 2 --epochs 100 --cube_size 4 --cube_type qtm
```

Parameters:
-hd1: Size of the first hidden layer (e.g., 2000).
-hd2: Size of the second hidden layer (0 means no second layer).
-nrd: Number of residual blocks (0 means no residual blocks).
-epochs: Number of training epochs.
-batch_size: Batch size (default 10000).
-lr: Learning rate for the optimizer (default 0.001).
-K_min and --K_max: Minimum and maximum values for random walks (default 1 and 30).
-cube_size: Cube size (e.g., 3 for 3x3x3 or 4 for 4x4x4).
-cube_type: Cube move set (qtm for quarter-turn metric or all for all moves).
-name: Training session name (optional, auto-generated if not provided).