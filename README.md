# Autonomous Snake Battle

Two AI-controlled snakes compete head-to-head, each learning to survive and defeat their opponent using Deep Q-Networks (DQN). The entire AI system is built from scratch in pure Go—no machine learning libraries required.

![Go](https://img.shields.io/badge/Go-1.21+-00ADD8?style=flat&logo=go)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## What Is This?

This project teaches two snakes to play against each other using reinforcement learning. Each snake learns through trial and error: making moves, seeing what happens, and adjusting its strategy over time. After training, you can watch them battle in real-time.

**Key features:**
- Two snakes compete on a shared board
- Self-play training: both snakes use the same "brain" and learn together
- Pure Go implementation of neural networks and DQN
- Real-time visualization with game controls
- Pre-trained model included—watch battles immediately

## Quick Start

### Prerequisites

- [Go 1.21+](https://golang.org/dl/)

### Watch Trained Snakes Battle

```bash
# Clone the repository
git clone https://github.com/yourusername/autonomous-snake.git
cd autonomous-snake

# Run with the pre-trained model
go run cmd/play/main.go
```

You'll see a window with two snakes (green and blue) competing for food while trying to survive.

### Controls

| Key | Action |
|-----|--------|
| Space | Pause/Resume |
| Up Arrow | Increase speed |
| Down Arrow | Decrease speed |
| R | Reset game |
| Q | Quit |

### Train Your Own Model

```bash
# Train for 10,000 episodes (takes a few minutes)
go run cmd/train/main.go -episodes 10000

# Train with custom options
go run cmd/train/main.go -episodes 5000 -board 25 -save-freq 500
```

## How It Works

### The Game

Two snakes spawn on opposite sides of a 20x20 grid. Each turn:
1. Both snakes choose their next move simultaneously
2. They move one square in their chosen direction
3. If a snake eats food, it grows longer and new food spawns
4. The game ends when a snake hits a wall, itself, or the other snake

A snake wins by being the last one alive. If they collide head-to-head, it's a tie.

### The AI: Deep Q-Networks

The snakes don't follow rules we wrote—they learn from experience. Here's how:

#### What the Snake "Sees" (State)

Each snake perceives its world through 22 numbers that describe:
- **Immediate danger**: Is there a wall or snake body straight ahead? To the left? To the right?
- **Extended danger**: Same checks, but 2 steps ahead
- **Current direction**: Which way am I facing?
- **Food location**: Where's the food relative to my head?
- **Opponent info**: Where's the other snake? How close? How long?
- **My length**: How long am I?

This compact representation lets the snake make decisions without seeing the entire board.

#### What the Snake Can Do (Actions)

Rather than choosing compass directions (north, south, east, west), the snake chooses relative movements:
- **Go Straight**: Continue in the current direction
- **Turn Left**: Rotate 90° counterclockwise and move
- **Turn Right**: Rotate 90° clockwise and move

This prevents impossible moves (a snake can't reverse into itself) and makes learning simpler.

#### How It Learns (Q-Learning)

The snake learns by associating states with action values (Q-values). After each move, it asks: "Was that a good decision?"

**Rewards tell the snake what's good:**
| Event | Reward |
|-------|--------|
| Eating food | +0.5 |
| Opponent dies | +1.0 |
| Dying | -1.0 |
| Surviving a turn | +0.01 |
| Moving toward food | +0.1 |
| Moving away from food | -0.1 |

Over thousands of games, the snake learns which actions lead to higher total rewards.

#### The Neural Network

The "brain" is a simple neural network:

```
Input (22 features) → Hidden Layer (128 neurons) → Hidden Layer (64 neurons) → Output (3 Q-values)
```

For each state, the network outputs three numbers—one for each action. The snake picks the action with the highest value (usually).

#### Exploration vs Exploitation

Early in training, the snake moves randomly (exploration) to discover what works. Gradually, it starts using what it's learned (exploitation). This balance is controlled by epsilon (ε):
- ε = 1.0: Always random moves
- ε = 0.01: Almost always use learned strategy

Epsilon decays from 1.0 to 0.01 during training.

#### Experience Replay

The snake stores memories of past experiences (state, action, reward, next state) in a replay buffer. During training, it samples random batches of memories to learn from. This breaks correlations between consecutive experiences and leads to more stable learning.

#### Target Network

A separate copy of the network (the target network) provides stable Q-value targets during learning. It's updated periodically rather than every step, which prevents the learning process from becoming unstable.

## Project Structure

```
autonomous-snake/
├── cmd/
│   ├── play/          # Visual game runner
│   └── train/         # Headless training loop
├── internal/
│   ├── ai/            # DQN implementation
│   │   ├── agent.go   # Decision-making and learning
│   │   ├── network.go # Neural network from scratch
│   │   ├── state.go   # State encoding (22 features)
│   │   └── replay.go  # Experience replay buffer
│   ├── game/          # Core game logic
│   │   ├── game.go    # Game state and rules
│   │   ├── snake.go   # Snake movement and growth
│   │   └── collision.go
│   ├── render/        # Ebiten visualization
│   └── config/        # Configuration constants
├── models/            # Saved neural network weights
└── Makefile           # Build and run shortcuts
```

## Configuration

### Command Line Options

**Play mode:**
```bash
go run cmd/play/main.go [options]
  -model string    Path to trained model (default "models/snake_dqn.gob")
  -board int       Board size (default 20)
  -grid int        Cell size in pixels (default 20)
  -seed int        Random seed for reproducibility
  -random          Use random actions instead of trained model
```

**Training:**
```bash
go run cmd/train/main.go [options]
  -episodes int    Number of training episodes (default 10000)
  -model string    Save path for trained model (default "models/snake_dqn.gob")
  -load string     Load existing model to continue training
  -board int       Board size (default 20)
  -save-freq int   Save checkpoint every N episodes (default 500)
  -log-freq int    Print stats every N episodes (default 100)
```

### Training Hyperparameters

Found in `internal/config/config.go`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 0.001 | How fast the network adjusts |
| Gamma (γ) | 0.99 | Discount factor for future rewards |
| Epsilon Start | 1.0 | Initial exploration rate |
| Epsilon End | 0.01 | Final exploration rate |
| Epsilon Decay | 0.995 | Decay multiplier per episode |
| Batch Size | 64 | Experiences per training step |
| Buffer Size | 100,000 | Max stored experiences |
| Target Update | 1000 | Steps between target network updates |

## Make Commands

```bash
make build        # Build both binaries
make play         # Run visual game
make play-random  # Run with random actions
make train        # Train for 5000 episodes
make train-quick  # Train for 1000 episodes (testing)
make train-long   # Train for 20000 episodes
make test         # Run tests
make test-cover   # Generate coverage report
make clean        # Remove built binaries
```

## How Self-Play Works

Both snakes use the same neural network, but they see different states (each perceives the other as "opponent"). During training:

1. Both snakes observe their current state
2. Both select actions using the same network
3. The game advances one step
4. Each snake stores its own experience (what it saw, what it did, what happened)
5. The network learns from both perspectives

This creates an emergent curriculum: as one snake improves, it becomes a harder opponent for the other, driving continuous improvement.

## Technical Details

### Neural Network Implementation

The network is implemented from scratch in `internal/ai/network.go`:
- **Forward pass**: Matrix multiplication with ReLU activation
- **Backpropagation**: Computes gradients using cached activations
- **Initialization**: Xavier/Glorot initialization for stable training
- **Serialization**: Saves/loads using Go's `gob` encoding

### State Encoding

The 22-feature state vector (see `internal/ai/state.go`):

```
Features 0-2:   Danger straight, left, right (binary)
Features 3-6:   Current direction (one-hot: up, down, left, right)
Features 7-10:  Food direction (one-hot: up, down, left, right)
Features 11-14: Opponent direction (one-hot)
Feature 15:     Distance to opponent head (normalized 0-1)
Feature 16:     Distance to nearest opponent body (normalized 0-1)
Feature 17:     Own length (normalized)
Feature 18:     Opponent length (normalized)
Features 19-21: Extended danger (2 steps ahead)
```

### Why Pure Go?

Building the neural network from scratch serves educational purposes:
- Understand exactly how forward/backward passes work
- No external ML dependencies to install
- Demonstrates that basic deep learning isn't magic
- Full control over the implementation

It's not meant to be a production level project.

## License

MIT License - See [LICENSE](LICENSE) for details.

## Acknowledgments

- [Ebiten](https://ebiten.org/) - The 2D game library used for visualization
- DeepMind's DQN papers for the foundational algorithms
