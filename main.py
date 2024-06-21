import gymnasium as gym
from gymnasium import wrappers

from agent import Agent
from transition import Transition

from collections import deque
from datetime import datetime

import numpy as np
import json


GREEN = '\033[0;37;42m'
RED = '\033[0;37;41m'
BLACK = '\033[0m'

last_episode = 0
max_returns = 0
max_returns_episode = 0
last_steps = []


def save_report(agent, path):
    report = {"doubleq": DOUBLEQ,  
              "max_episodes": str(episodes),
              "episodes_finished": str(last_episode),
              "max_steps": str(max_steps),
              "max_memory_size": str(memory_size),
              "memory_size": str(len(agent.memory.transition_deque)),
              "sample_size": str(sample_size),
              "lr": str(lr),
              "discount": str(discount),
              "start_epsilon": str(epsilon),
              "last_epsilon": str(agent.epsilon),
              "decay": str(decay),
              "max_n_steps": str(last_steps),
              "last_scores": str([str(np.round(x, 1)) for x in last_steps]),
              "stop_score": str(stop_score),
              "highest_score": str(max_returns),
              "highest_score_episode": str(max_returns_episode),
              "averaging_rate": str(averaging_rate)
              }
    
    with open(path, "w") as f:
        f.write(json.dumps(report, indent=4))

try:
    def run_environment(episodes, max_steps, agent, last_steps_n, stop_score):
        global max_returns
        global max_returns_episode
        global last_episode
        global last_steps

        env = gym.make("LunarLander-v2", render_mode="human")
        last_steps = deque(maxlen=last_steps_n)


        for episode in range(episodes):
            last_episode = episode
            returns = 0
            state, _ = env.reset()
            t1 = datetime.now()

            for _ in range(max_steps):
                action = agent.select_action(state)

                next_state, reward, terminated, truncated, _ = env.step(action)

                returns += reward
                agent.memory.store(
                    Transition(state, next_state, action, reward, terminated)
                )
                # print(len(agent.memory.transition_deque))
                if terminated or truncated:
                    break

                state = next_state

                if DOUBLEQ:
                    agent.double_train()
                else:
                    agent.train()

                agent.decay()

            t2 = datetime.now()
            last_steps.append(returns)
            if returns > max_returns or episode == 0:
                max_returns = returns
                max_returns_episode = episode

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
        return episode, last_steps, max_returns_episode, max_returns


    name = "06-21_13-17" # name of the model

    DOUBLEQ = True # use double q learning?

    episodes = 400 # max number of episodes
    max_steps = 1_000 # max steps in an episode

    memory_size = 100_000 # max memory size
    sample_size = 64 # train samples per step

    lr = 0.001 # learning rate
    discount = 0.99 # discount
    epsilon = 0 # epsilon, value for taking random moves
    decay = 0.996 # decay of epsilon
    averaging_rate = 0.1 # averaging rate for doubleq
    
    last_steps_n = 20 # last number of steps to measure score by
    stop_score = 250 # average score to achieve to finish training prematurely

    main_agent = Agent(epsilon=epsilon, 
                        sample_size=sample_size, 
                        memory_size=memory_size, 
                        discount=discount, 
                        lr=lr, 
                        policy=f".\\models\\model_{name}.pt",
                        dq_policy=f".\\models\\model_target_{name}.pt",
                        memory_path=f".\\memory\\memory_{name}.jsonl",
                        decay_amt=decay,
                        doubleq=DOUBLEQ,
                        averaging_rate=averaging_rate)

    results = run_environment(episodes=episodes, 
                    max_steps=max_steps, 
                    agent=main_agent, 
                    last_steps_n=last_steps_n, 
                    stop_score=stop_score)



except BaseException as e:
    print(e)

finally:
    print("Saving model")
    main_agent.policy.save_model(f"models/model_{datetime.now().strftime('%m-%d_%H-%M')}.pt", f"models/model_target_{datetime.now().strftime('%m-%d_%H-%M')}.pt")
    main_agent.save_memory(f"memory/memory_{datetime.now().strftime('%m-%d_%H-%M')}.jsonl")

    print("saving report")
    save_report(main_agent, f"reports/report_{datetime.now().strftime('%m-%d_%H-%M')}.json")


