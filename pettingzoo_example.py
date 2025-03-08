import time

from generals import GridFactory
from generals.agents import ExpanderAgent, RandomAgent
from generals.envs import PettingZooGenerals

# Initialize agents
random = RandomAgent()
expander = ExpanderAgent()

# Names are used for the environment
agent_names = [random.id, expander.id]
# Store agents in a dictionary
agents = {random.id: random, expander.id: expander}

# Create environment
grid_factory = GridFactory(mode="generalsio")
env = PettingZooGenerals(agents=agent_names, render_mode=None, grid_factory=grid_factory)


observations, info = env.reset()
i = 0

terminated = truncated = False
while i < 400_000:
    actions = {}
    for agent in env.agents:
        # Ask agent for action
        actions[agent] = [1, 0, 0, 0, 0]
    # All agents perform their actions
    observations, rewards, terminated, truncated, info = env.step(actions)
    env.render()
    i += 1
    if i == 1:
        start = time.time()
print(f"Time taken: {time.time() - start:.2f}s")
