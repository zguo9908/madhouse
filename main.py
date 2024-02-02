import os

import inline
import matplotlib
import numpy as np
from RLGlue import RLGlue

from tqdm import tqdm

from qlearningagent import QLearningAgent

matplotlib.use('TkAgg')  # Use TkAgg backend
from environment import GiveUpEnvironment
import matplotlib.pyplot as plt

rep_structure = ['regular', 'dilating']

if __name__ == '__main__':

    env_info = {"state representation structure": "regular",
                  "fundamental timestep": 0.1,
                  "gamma": 0.9,
                  "gu duration": 30,
                  "bg duration": 2,
                  "consumption duration": 3,
                  "exponential distribution scale": 3,
                  "overall reward probability": 0.9,
                  "reward amount in pursuit": 10,
                  "seed": 0}

    env = GiveUpEnvironment()
    # matplotlib.use('Agg')
    env.env_init(env_info)
    env.test_reward_function()
    (rou_g, rou_l, t_rou_g, t_rou_l) = env.optimal_policy()
    fig, ax = plt.subplots(1, 1)
    ax.plot(env.PSTimeArray, rou_g)
    ax.plot(env.PSTimeArray, rou_l)
    print(f"According to the global reward rate, the optimal give-up time is {t_rou_g} second!")
    # plt.show()

    num_states = env.num_states

    # run experiment
    MaxRuns = 5
    leave_values = np.zeros([MaxRuns, num_states])
    stay_values = np.zeros([MaxRuns, num_states])
    MaxSteps = 100000000

    for j in range(MaxRuns):

        seed = j + 10
        env_info["seed"] = seed

        env = GiveUpEnvironment()
        env.env_init(env_info)
        num_states = env.num_states
        print(f'number of states in this world is {num_states}')

        agent_info = {"num_actions": 2,
                      "num_states": num_states,
                      "epsilon": 0.1,
                      "discount": 0.9,
                      "step_size": 0.1,
                      "seed": seed + 500000000,
                      "degree of exploration": 0.5,
                      "exploration method": "UCB"}
        agent = QLearningAgent()

        # initialize agent-env interaction
        rl_glue = RLGlue(GiveUpEnvironment, QLearningAgent)
        rl_glue.agent.agent_info = agent_info
        rl_glue.rl_init(agent_info, env_info)
        rl_glue.rl_start(agent_info, env_info)
        values = rl_glue.agent.agent_message("get_action_values")
        TD_error = np.zeros(MaxSteps)
        prev_state = np.zeros(MaxSteps)
        current_state = np.zeros(MaxSteps)
        prev_action = np.zeros(MaxSteps)
        rewarded_bool = np.zeros(MaxSteps)

        for i in tqdm(range(MaxSteps)):
            prev_state[i] = rl_glue.agent.prev_state
            prev_action[i] = rl_glue.agent.prev_action
            roat = rl_glue.rl_step()
            # print(roat)
            rewarded_bool[i] = (roat[0] != 0)
            TD_error[i] = rl_glue.agent.agent_message("get TD error")
            current_state[i] = rl_glue.agent.prev_state

        values = rl_glue.agent.agent_message("get_action_values")
        leave_values[j] = values[:, 1]
        stay_values[j] = values[:, 0]

        prev_state = prev_state.astype(int)
        current_state = current_state.astype(int)

    path = os.path.normpath(r'D:\figures\behplots') + "\\" + "modeling"
    fig, ax = plt.subplots(1, 1)
    plot1 = plt.plot(np.mean(leave_values, axis=0), label="q_leave")
    plot2 = plt.plot(np.mean(stay_values, axis=0), label="q_stay")
    plt.legend(loc="upper right")
    plt.show()
    # plt.savefig(f'{env.dt}_{agent.exploration_method}_{MaxSteps}.svg')
    print(f'number of trials experienced {env.num_trials}')
    FindExitIndex = np.where(values[0:120, 0] < values[0:120, 1])
    IndexArray = FindExitIndex[0]
    BestExitIndex = min(IndexArray)
    print(BestExitIndex)
    print(env.state1d(BestExitIndex))