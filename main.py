from environment import GiveUpEnvironment
import matplotlib.pyplot as plt


if __name__ == '__main__':
    env = GiveUpEnvironment()
    env.env_init({"state representation structure": "regular",
                  "fundamental timestep": 0.01,
                  "gamma": 0.9,
                  "gu duration": 28,
                  "bg duration": 2,
                  "consumption duration": 0.5,
                  "exponential distribution scale": 3,
                  "overall reward probability": 0.9,
                  "reward amount in pursuit": 10,
                  "seed": 0})

    env.test_reward_function()
    (rou_g, rou_l, t_rou_g, t_rou_l) = env.optimal_policy()
    # hazard_function = np.zeros(len(env.PSTimeArray))
    # for i in range(len(env.PSTimeArray)):
    #     Pr_NeverRewarded = 1 - env.PSRewardTimeCDF[i]
    #     Pr_NeverRewardedNRewardedAtT = env.PSRewardTimePDF[i]
    #     Pr_RewardedAtTGivenNotYetRewarded = Pr_NeverRewardedNRewardedAtT / Pr_NeverRewarded
    #     hazard_function[i] = Pr_RewardedAtTGivenNotYetRewarded
    fig, ax = plt.subplots(1, 1)
    ax.plot(env.PSTimeArray, rou_g)
    ax.plot(env.PSTimeArray, rou_l)
    ax.legend()
    print(f"According to the global reward rate, the optimal give-up time is {t_rou_g} second!")