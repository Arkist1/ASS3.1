import gymnasium as gym
from gymnasium import wrappers

from agent import Agent
from transition import Transition

from collections import deque
from datetime import datetime

import numpy as np

try:

    def run_environment(episodes, max_steps, agent):
        env = gym.make("LunarLander-v2", render_mode="human")
        last_100 = deque(maxlen=100)

        for episode in range(episodes):
            returns = 0
            state, _ = env.reset()
            t1 = datetime.now()

            for _ in range(max_steps):
                action = agent.select_action(state)

                next_state, reward, terminated, truncated, info = env.step(action)

                returns += reward
                agent.memory.store(Transition(state, next_state, reward, action))
                # print(len(agent.memory.transition_deque))
                if terminated or truncated:
                    print(reward)
                    break

                state = next_state
                agent.train(learning_rate=0.001)
                agent.decay()

            t2 = datetime.now()
            last_100.append(returns)

            mean_last_100 = np.mean(last_100)
            time_delta = t2 - t1
            print(
                "Finished episode",
                episode,
                "in",
                str(time_delta.seconds) + "." + str(time_delta.microseconds)[:-3] + "s",
                "Current last_100 mean:",
                mean_last_100,
            )

            if mean_last_100 > 250:
                break

        env.close()

    policy = None
    policy = "./model.pt"

    main_agent = Agent(epsilon=0.1, policy=policy)
    episodes = 2_000
    max_steps = 500
    run_environment(episodes, max_steps, main_agent)

except BaseException as e:
    print(e)

finally:
    print("Saving model")
    main_agent.policy.save_model("./model.pt")
