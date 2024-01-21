import inline
import matplotlib
import numpy as np
from RLGlue import RLGlue

from tqdm import tqdm

from qlearningagent import QLearningAgent

matplotlib.use('TkAgg')  # Use TkAgg backend
from environment import GiveUpEnvironment
import matplotlib.pyplot as plt

rep_structure = ['regular', 'dilating ']

if __name__ == '__main__':
    # env = GiveUpEnvironment()
    # env.env_init({"state representation structure": "regular",
    #               "fundamental timestep": 0.01,
    #               "gamma": 0.9,
    #               "gu duration": 28,
    #               "bg duration": 2,
    #               "consumption duration": 0.5,
    #               "exponential distribution scale": 3,
    #               "overall reward probability": 0.9,
    #               "reward amount in pursuit": 10,
    #               "seed": 0})

    env_info = {"state representation structure": "regular",
                  "fundamental timestep": 0.01,
                  "gamma": 0.9,
                  "gu duration": 28,
                  "bg duration": 2,
                  "consumption duration": 0.5,
                  "exponential distribution scale": 3,
                  "overall reward probability": 0.9,
                  "reward amount in pursuit": 10,
                  "seed": 0}

    env = GiveUpEnvironment(env_info=env_info)
   # env.env_init(env_info)
    # matplotlib.use('Agg')
    env.env_init(env_info)
    env.test_reward_function()
    (rou_g, rou_l, t_rou_g, t_rou_l) = env.optimal_policy()
    fig, ax = plt.subplots(1, 1)
    ax.plot(env.PSTimeArray, rou_g)
    ax.plot(env.PSTimeArray, rou_l)
   # ax.legend()
    print(f"According to the global reward rate, the optimal give-up time is {t_rou_g} second!")
    # plt.show()



    num_states = env.num_states

    # run experiment
    MaxRuns = 5
    leave_values = np.zeros([MaxRuns, num_states])
    stay_values = np.zeros([MaxRuns, num_states])
    for j in range(MaxRuns):

        seed = j + 10
        env_info["seed"] = seed

        env = GiveUpEnvironment(env_info=env_info)
        env.env_init(env_info)
        num_states = env.num_states
        print(f'number of states in this world is {num_states}')

        agent_info = {"num_actions": 2,
                      "num_states": num_states,
                      "epsilon": 0.05,
                      "discount": 0.9,
                      "step_size": 0.1,
                      "seed": seed + 5000,
                      "degree of exploration": 0.5,
                      "exploration method": "UCB"}
        agent = QLearningAgent(agent_info=agent_info)

        print("Printing agent_info before QLearningAgent initialization:")
        print(agent_info)
      #  agent_info = {"num_actions": 2, "other_key": "value"}

        # agent.agent_init(agent_info)
        # print(agent.policy)
       # agent_info.update({"policy": policy})

        rl_glue = RLGlue(GiveUpEnvironment, QLearningAgent)
        rl_glue.agent.agent_info = agent_info
        rl_glue.rl_init(agent_info, env_info)
        rl_glue.rl_start(agent_info, env_info)
        values = rl_glue.agent.agent_message("get_action_values")
        MaxSteps = 500000
        TD_error = np.zeros(MaxSteps)
        prev_state = np.zeros(MaxSteps)
        current_state = np.zeros(MaxSteps)
        prev_action = np.zeros(MaxSteps)
        rewarded_bool = np.zeros(MaxSteps)

        for i in tqdm(range(MaxSteps)):
            prev_state[i] = rl_glue.agent.prev_state
            prev_action[i] = rl_glue.agent.prev_action
            roat = rl_glue.rl_step()
            rewarded_bool[i] = (roat[0] != 0)
            TD_error[i] = rl_glue.agent.agent_message("get TD error")
            current_state[i] = rl_glue.agent.prev_state

        values = rl_glue.agent.agent_message("get_action_values")
        leave_values[j] = values[:, 1]
        stay_values[j] = values[:, 0]

        prev_state = prev_state.astype(int)
        current_state = current_state.astype(int)
    fig, ax = plt.subplots(1, 1)
    plot1 = plt.plot(np.mean(leave_values, axis=0), label="q_leave")
    plot2 = plt.plot(np.mean(stay_values, axis=0), label="q_stay")
    plt.legend(loc="upper right")
    plt.show()

    # FindExitIndex = np.where(values[80:120, 0] < values[80:120, 1])
    # IndexArray = FindExitIndex[0] + 80
    # BestExitIndex = min(IndexArray)
    # print(BestExitIndex)
    # print(env.state3d(BestExitIndex))