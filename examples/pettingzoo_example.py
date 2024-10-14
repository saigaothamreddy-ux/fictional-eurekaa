from generals.agents import RandomAgent, ExpanderAgent
from generals.envs import PettingZooGenerals

# Initialize agents
random = RandomAgent()
expander = ExpanderAgent()

# Store agents in a dictionary
agents = {
    random.id: random,
    expander.id: expander,
}

# Create environment
env = PettingZooGenerals(agents=agents, render_mode="human")
observations, info = env.reset()

done = False
while not done:
    actions = {}
    for agent in env.agents:
        # Ask agent for action
        actions[agent] = agents[agent].act(observations[agent])
    # All agents perform their actions
    observations, rewards, terminated, truncated, info = env.step(actions)
    done = any(terminated.values()) or any(truncated.values())
    env.render()
