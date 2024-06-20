import gymnasium as gym
from gymnasium import wrappers

from agent import Agent
from transition import Transition

from collections import deque
from datetime import datetime

import numpy as np


GREEN = '\033[0;37;42m'
RED = '\033[0;37;41m'
BLACK = '\033[0m'

try:
    def run_environment(episodes, max_steps, agent, last_steps_n, stop_score):
        env = gym.make("LunarLander-v2", render_mode="human")
        last_steps = deque(maxlen=last_steps_n)

        for episode in range(episodes):
            returns = 0
            state, _ = env.reset()
            t1 = datetime.now()

            for _ in range(max_steps):
                action = agent.select_action(state)

                next_state, reward, terminated, truncated, info = env.step(action)

                returns += reward
                agent.memory.store(
                    Transition(state, next_state, action, reward, terminated)
                )
                # print(len(agent.memory.transition_deque))
                if terminated or truncated:
                    break

                state = next_state
                agent.train()
                agent.decay()

            t2 = datetime.now()
            last_steps.append(returns)

            mean_last = np.mean(last_steps)
            time_delta = t2 - t1
            colour = (GREEN if returns > mean_last else RED)

            print(
                colour +
                "Finished episode",
                " " * (3 -len(str(episode))),
                episode,
                "in",
                str(time_delta.seconds) + "." + str(time_delta.microseconds)[0:2] + "s\t",
                f"Current last {last_steps_n} mean:",
                np.round(mean_last, 1),
                "\tLast reward:", np.round(reward, 1),
                " Returns:",str(np.round(returns, 2))
                + BLACK
            )

            if mean_last > stop_score:
                break

        env.close()

    policy = None
    # policy = "./model.pt"

    episodes = 2_000
    max_steps = 1_000
    
    memory_size = 100_000
    sample_size = 64

    lr = 0.001
    discount = 0.99
    epsilon = 0.1
    decay = 0.996
    
    last_steps = 20
    stop_score = 200

    main_agent = Agent(epsilon=epsilon, 
                       sample_size=sample_size, 
                       memory_size=memory_size, 
                       discount=discount, 
                       lr=lr, 
                       policy=policy,
                       decay_amt=decay)
    run_environment(episodes=episodes, 
                    max_steps=max_steps, 
                    agent=main_agent, 
                    last_steps_n=last_steps, 
                    stop_score=stop_score)
    



except BaseException as e:
    print(e)

finally:
    print("Saving model")
    main_agent.policy.save_model(f"./{datetime.now().strftime('%m-%d_%H-%M')}.pt")
