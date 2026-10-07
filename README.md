# Snake Reinforcement Learning

Hi!, this is a project I am making where I teach a neural network to play snake using a DDQN (Double Deep Q Network). I am using Gymnasium to help the agent interact with the game. Currently I have a simple model and architechure, with a simple observation space, informing the agent what direction it is going, what direction the food is in, and if the tiles around it endanger the snake. I will add two more observation spaces:

- A expanded observation space which allows the agent to see more of what is around the snake, not just the immediate area.

- The full game state (entire board representing the game), and then using a CNN (Convolutional Neural Network) to interpret the game and decide the next best action

## Setup

### Requirements

- Python 3.10 or newer
- `pip`

### Installation

Clone the repository and move into the project directory:

```bash
git clone <repository-url>
cd snakerl
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

### Verify the installation

Run the Gymnasium environment checker:

```bash
python check.py
```

### Train the agent

Start DDQN training with one of the hyperparameter sets defined in `hyperparameters.yml`:

```bash
python agent.py snake1 --train
```

To load a saved model and render it:

```bash
python agent.py snake1
```

Training logs, saved models, and graphs are written to the `runs/` directory.

Provided in this repository is snakebest.pt, which is the best model so far.

To run it:

```bash
python agent.py snakebest
```

Thank you for viewing this repo!
