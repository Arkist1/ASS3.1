import gymnasium as gym

from agent import Agent
from transition import Transition


def run_environment(steps, agent):
    env = gym.make("LunarLander-v2", render_mode="human")
    state, info = env.reset(seed=42)

    for _ in range(steps):
        action = agent.select_action(state)
        # action = env.action_space.sample()
        print("Action chosen:", action)  # this is where you would insert your policy

        next_state, reward, terminated, truncated, info = env.step(action)

        agent.memory.store(Transition(state, next_state, reward, action))

        if terminated or truncated:
            observation, info = env.reset()
            print(observation)

        state = next_state
    env.close()


main_agent = Agent(epsilon=0.1)
run_environment(1000, main_agent)
