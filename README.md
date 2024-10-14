<div align="center">

![Gameplay GIF](https://raw.githubusercontent.com/strakam/Generals-RL/master/generals/assets/gifs/wider_gameplay.gif)

## **Generals.io RL**

[![CodeQL](https://github.com/strakam/Generals-RL/actions/workflows/codeql.yml/badge.svg)](https://github.com/strakam/Generals-RL/actions/workflows/codeql.yml)
[![CI](https://github.com/strakam/Generals-RL/actions/workflows/tests.yml/badge.svg)](https://github.com/strakam/Generals-RL/actions/workflows/tests.yml)




[Installation](#-installation) • [Getting Started](#-getting-started) • [Customization](#-custom-grids) • [Environment](#-environment)
</div>

Generals-RL is a real-time strategy environment where players compete to conquer their opponents' generals on a 2D grid.
While the goal is simple — capture the enemy general — the gameplay combines strategic depth with fast-paced action,
challenging players to balance micro and macro-level decision-making.
The combination of these elements makes the game highly engaging and complex.

This repository is designed to make bot development for [Generals.io]((https://generals.io)) easier,
particularly for Machine Learning agents.

Highlights:
* 🚀 **blazing-fast, lightweight simulator**: run thousands of steps per second with numpy-powered efficiency
* 🤝 **seamless integration**: fully compatible with RL standards 🤸[Gymnasium](https://gymnasium.farama.org/) and 🦁[PettingZoo](https://pettingzoo.farama.org/)
* 🔧 **effortless customization**: easily tailor environments to your specific needs
* 🔬 **analysis tools**: leverage features like replays for deeper insights
> [!Tip]
> This repository is based on the [generals.io](https://generals.io) game. Check it out, it is a lot of fun!

## 📦 Installation
You can install the latest stable version via pip for reliable release
```bash
pip install generals
```
or clone the repo for the most up-to-date features
```bash
git clone https://github.com/strakam/Generals-RL
cd Generals-RL
pip install -e .
```

## Usage example (🤸 Gymnasium)

```python
import gymnasium as gym
from generals import AgentFactory

# Initialize agents
agent = AgentFactory.make_agent("expander")
npc = AgentFactory.make_agent("random")

env = gym.make(
    "gym-generals-v0",
    agent=agent,
    npc=npc,
    render_mode="human",
)

observation, info = env.reset()

terminated = truncated = False
while not (terminated or truncated):
    action = agent.act(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
```
You can also check an example for 🦁[PettingZoo](./examples/pettingzoo_example.py) or
an example with commentary showcasing various features [here](./examples/complete_example.py).

## 🚀 Getting Started
Creating your first agent is very simple.
- Start by subclassing an `Agent` class just like [`RandomAgent`](./generals/agents/random_agent.py) or [`ExpanderAgent`](./generals/agents/expander_agent.py).
- Every agent must have a name as it is his ID by which he is called for actions.
- Every agent must implement `play(observation)` function that takes in `observation` and returns an `action` (both defined below).
- You can start by copying the [Usage Example](#usage-example--gymnasium) and replacing `agent` with your implementation.
- When creating an environment, you can choose out of two `render_modes`:
     - `None` that omits rendering and is suitable for training,
     - `"human"` where you can see the game play out.

> [!TIP]
> Check out `Makefile` and run some examples to get a feel for the game 🤗.

## 🎨 Custom grids
Grids are generated via `GridFactory`. You can instantiate the class with desired grid properties, and it will generate
grid with these properties for each run.
```python
import gymnasium as gym
from generals import GridFactory

grid_factory = GridFactory(
    grid_dims=(10, 10),                    # Dimensions of the grid (height, width)
    mountain_density=0.2,                  # Probability of a mountain in a cell
    city_density=0.05,                     # Probability of a city in a cell
    general_positions=[(0,3),(5,7)],       # Positions of generals (i, j)
)

# Create environment
env = gym.make(
    "gym-generals-v0",
    grid_factory=grid_factory,
    ...
)
```
You can also specify grids manually, as a string via `options` dict:
```python
import gymnasium as gym

env = gym.make("gym-generals-v0", ...)
grid = """
.3.#
#..A
#..#
.#.B
"""

options = {"grid": grid}

# Pass the new grid to the environment (for the next game)
env.reset(options=options)
```
Grids are encoded using these symbols:
- `.` for cells where you can move your army
- `#` for mountains (terrain that can not be passed)
- `A,B` are positions of generals
- digits `0-9` represent cities with the cost calculated as `(40 + digit)`

## 🔬 Replays
We can store replays and then analyze them. `Replay` class handles replay related functionality.
### Storing a replay
```python
import gymnasium as gym

env = gym.make("gym-generals-v0", ...)

options = {"replay": "my_replay"}
env.reset(options=options) # The next game will be encoded in my_replay.pkl
```

### Loading a replay

```python
from generals import Replay

# Initialize Replay instance
replay = Replay.load("my_replay")
replay.play()
```
### 🕹️ Replay controls
- `q` — quit/close the replay
- `r` — restart replay from the beginning
- `←/→` — increase/decrease the replay speed
- `h/l` — move backward/forward by one frame in the replay
- `spacebar` — toggle play/pause
- `mouse` click on the player's row — toggle the FoV (Field of View) of the given player

> [!WARNING]
> We are using the [pickle](https://docs.python.org/3/library/pickle.html) module which is not safe!
> Only open replays you trust.

## 🌍 Environment
### 🔭 Observation
An observation for one agent is a dictionary `{"observation": observation, "action_mask": action_mask}`.

The `observation` is a `Dict`. Values are either `numpy` matrices with shape `(N,M)`, or simple `int` constants:
| Key                  | Shape     | Description                                                                  |
| -------------------- | --------- | ---------------------------------------------------------------------------- |
| `army`               | `(N,M)`   | Number of units in a cell regardless of the owner                            |
| `general`            | `(N,M)`   | Mask indicating cells containing a general                                   |
| `city`               | `(N,M)`   | Mask indicating cells containing a city                                      |
| `visible_cells`      | `(N,M)`   | Mask indicating cells that are visible to the agent                          |
| `owned_cells`        | `(N,M)`   | Mask indicating cells owned by the agent                                     |
| `opponent_cells`     | `(N,M)`   | Mask indicating cells owned by the opponent                                  |
| `neutral_cells`      | `(N,M)`   | Mask indicating cells that are not owned by any agent                        |
| `structure`          | `(N,M)`   | Mask indicating whether cells contain cities or mountains, even out of FoV   |
| `owned_land_count`   |     —     | Number of cells the agent owns                                               |
| `owned_army_count`   |     —     | Total number of units owned by the agent                                     |
| `opponent_land_count`|     —     | Number of cells owned by the opponent                                        |
| `opponent_army_count`|     —     | Total number of units owned by the opponent                                  |
| `is_winner`          |     —     | Indicates whether the agent won                                              |
| `timestep`           |     —     | Current timestep of the game                                                 |

`action_mask` is a mask with shape `(N,M,4)` where value `[i,j,d]` says whether you can move from cell `[i,j]` in a direction `d`.

### ⚡ Action
Action is a `tuple(pass, cell, direction, split)`, where:
- `pass` indicates whether you want to `1 (pass)` or `0 (play)`.
- `cell` is an `np.array([i,j])` where `i,j` are indices of the cell you want to move from
- `direction` indicates whether you want to move `0 (up)`, `1 (down)`, `2 (left)`, or `3 (right)`
- `split` indicates whether you want to `1 (split)` units (send half of them) or `0 (no split)`, which sends all possible units to the next cell.

> [!TIP]
> You can see how actions and observations look like by printing a sample form the environment:
> ```python
> print(env.observation_space.sample())
> print(env.action_space.sample())
> ```

### 🎁 Reward
It is possible to implement custom reward function. The default is `1` for winner and `-1` for loser, otherwise `0`.
```python
def custom_reward_fn(observation, action, done, info):
    # Give agent a reward based on the number of cells they own
    return observation["observation"]["owned_land_count"]

env = gym.make(..., reward_fn=custom_reward_fn)
observations, info = env.reset()
```
