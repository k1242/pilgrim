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
    git clone https://github.com/k1242/pilgrim.git
    cd pilgrim
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Training Script

You can run the `train.py` script to train a model on cube-based data. The model architecture is flexible, allowing different hidden layer sizes and residual blocks to be used.

#### Basic Usage

```bash
python train.py --hd1 2000 --hd2 1418 --nrd 2 --epochs 100 --cube_size 4 --cube_type all
```

#### Parameters:

*   `--hd1`: Size of the first hidden layer (e.g., `2000`).
*   `--hd2`: Size of the second hidden layer (`0` means no second layer).
*   `--nrd`: Number of residual blocks (`0` means no residual blocks).
*   `--epochs`: Number of training epochs.
*   `--batch_size`: Batch size (default `10000`).
*   `--lr`: Learning rate for the optimizer (default `0.001`).
*   `--optimizer`: Optimizer, `Adam` or `AdamSF` means schedulefree module use (default `Adam`).
*   `--K_min` and `--K_max`: Minimum and maximum values for random walks (default `1` and `30`).
*   `--cube_size`: Cube size (e.g., `3` for 3x3x3 or `4` for 4x4x4).
*   `--cube_type`: Cube move set (`qtm` for quarter-turn metric or `all` for all moves).
*   `--device_id`: Device ID to use different graphics card (default `0`).


### Testing the Model

You can test a trained **Pilgrim** model using the `test.py` script. This script loads the model, applies it to a set of cube states, and attempts to solve them using a beam search.

#### Basic Usage

~~~~bash
python test.py --cube_size 4 --cube_type all --weights weights/cube4_all_MLP2_2000_1418_0_4.00M_1727996220_e2pow14.pth --tests_num 10 --B 4096
~~~~

#### Parameters

*   `--cube_size`: The size of the cube (e.g., `4` for 4x4x4 cube).
*   `--cube_type`: The cube type, either `qtm` (quarter-turn metric) or `all` (all moves).
*   `--weights`: Path to the saved model weights.
*   `--tests`: Path to the test dataset (optional). If not provided, it defaults to the dataset in `datasets/{cube_type}_cube{cube_size}.pt`.
*   `--B`: Beam size for the beam search (default `4096`).
*   `--tests_num`: Number of test cases to run (default `10`).
*   `--device_id`: Device ID to use different graphics card.
*   `--verbose`: Each step of beam search printed as tqdm, default is 0.


#### Output

*   **Log File**: The test results, including solution lengths and attempts, are saved to a log file in the `logs/` directory. The log file is named based on the cube size, cube type, model ID, epoch, and beam size:

	~~~~text
	logs/test_cube4_qtm_123456_10000_B4096.json
	~~~~

*   **Console Output**: The solution length for each solved test case is printed to the console. If a solution is not found, it will print "Solution not found" for that test case.

#### Average Solution Length

At the end of the test, the script calculates the average solution length across all solved cubes and prints it along with the total time taken for testing.
