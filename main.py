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

rep_structure = ['dilating', 'regular']
dt = [0.1, 0.05]
max_steps = [1000000, 5000000]
gamma = np.linspace(0.7, 0.95, 6)


if __name__ == '__main__':
    for i_rep in rep_structure:
        for i_step in max_steps:
            for i_gamma in gamma:

                env_info = {"state representation structure": i_rep,
                              "fundamental timestep": 0.1,
                              "gamma": i_gamma,
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
                num_states = env.num_states

                # run experiment
                MaxRuns = 5
                leave_values = np.zeros([MaxRuns, num_states])
                stay_values = np.zeros([MaxRuns, num_states])
                # curr_max_step = max_steps[i_step]

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
                    agent.agent_init(agent_info)

                    # initialize agent-env interaction
                    rl_glue = RLGlue(GiveUpEnvironment, QLearningAgent)
                    rl_glue.agent.agent_info = agent_info
                    rl_glue.rl_init(agent_info, env_info)
                    rl_glue.rl_start(agent_info, env_info)
                    values = rl_glue.agent.agent_message("get_action_values")
                    TD_error = np.zeros(i_step)
                    prev_state = np.zeros(i_step)
                    current_state = np.zeros(i_step)
                    prev_action = np.zeros(i_step)
                    rewarded_bool = np.zeros(i_step)

                    for i in tqdm(range(i_step)):
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
                    # print(values)
                    prev_state = prev_state.astype(int)
                    current_state = current_state.astype(int)

                path = os.path.normpath(r'D:\figures\behplots') + "\\" + "modeling"
                os.chdir(path)
                fig, ax = plt.subplots(1, 1)
                mean_leave_values = np.mean(leave_values, axis=0)
                mean_stay_values = np.mean(stay_values,axis=0)
                plot1 = plt.plot(mean_leave_values, label="q_leave")
                plot2 = plt.plot(mean_stay_values, label="q_stay")
                plt.legend(loc="upper right")
                plt.xlabel('state')
                exit_indices = np.where(mean_stay_values < mean_leave_values)
                if len(exit_indices[0]) == 0:
                    print('not finding timepoints of exit')
                    plt.savefig(f'gamma{i_gamma}_dt{env.dt}_{agent.exploration_method}_steps{i_step}_{i_rep}.svg')
                    pass
                else:
                    best_exit_index = min(exit_indices[0])
                    best_exit_time = env.PSTimeArray[best_exit_index]
                    print(f'best time to exit is {best_exit_time} which is the {best_exit_index} state')
                    ax.annotate(f'in pursuit time={best_exit_time}s', xy=(best_exit_index,
                                mean_leave_values[best_exit_index]),
                                xytext=(best_exit_index, 0), textcoords='data',
                                arrowprops=dict(arrowstyle="->"), rotation='horizontal')
                    ax.axvspan(0, len(env.BGTimeArray)+0.5, color='yellow', alpha=0.1, label='background')
                    ax.axvspan(len(env.BGTimeArray)+0.5, num_states, color='green', alpha=0.1, label='pursuit')
                    plt.savefig(f'gamma{i_gamma}_dt{env.dt}_{agent.exploration_method}_steps{i_step}_{i_rep}.svg')
        # print(env.state1d(BestExitIndex))