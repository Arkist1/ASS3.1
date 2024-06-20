import gymnasium as gym
from gymnasium import wrappers

from agent import Agent
from transition import Transition


try:

    def run_environment(episodes, max_steps, agent):
        for episode in range(episodes):
            if episode % 50 == 0:
                env = gym.make("LunarLander-v2", render_mode="human")
                print(env.metadata["render_modes"])
            # elif episode % 50 == 1:
            #     env = gym.make("LunarLander-v2", render_mode=None)
            #     env.close()

            state, _ = env.reset()
            for _ in range(max_steps):
                action = agent.select_action(state)
                # action = env.action_space.sample()
                # print(
                #     "Action chosen:", action
                # )  # this is where you would insert your policy

                next_state, reward, terminated, truncated, info = env.step(action)

                agent.memory.store(Transition(state, next_state, reward, action))
                if terminated or truncated:
                    print("Finished episode", episode)
                    break

                state = next_state
                agent.train()

        env.close()

    policy = None
    policy = "./model.pt"

    main_agent = Agent(epsilon=0.1, policy=policy)
    episodes = 1_000
    max_steps = 1000
    run_environment(episodes, max_steps, main_agent)

except Exception as e:
    print(e)

finally:
    print("Saving model")
    main_agent.policy.save_model("./model.pt")
