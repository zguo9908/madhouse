import numpy as np
from matplotlib import pyplot as plt

import utils


class GiveUpEnvironment():
    def __init__(self, env_info={}):
        self.dt = env_info.get("fundamental timestep", 0.2)
        self.gamma = env_info.get("gamma", 0.9)
        self.rand_generator = np.random.RandomState(env_info.get("seed", 0))
        self.consmp_dur = env_info.get("consumption duration", 1)

        self.mean_reward_time = env_info.get("exponential distribution mean", 3)

    def env_init(self, env_info={}):
        """Setup for the environment called when the experiment first starts.
        Note:
            Initialize a tuple with the reward, first state, boolean
            indicating if it's terminal.
        """
        self.StateStruct = env_info.get("state representation structure", "regular")
        self.giveup_reward = env_info.get("reward amount in pursuit", 10)
        self.dt = env_info.get("fundamental timestep", 0.01)
        self.gamma = env_info.get("gamma", 0.9)
        self.rand_generator = np.random.RandomState(env_info.get("seed", 0))
        self.gu_dur = env_info.get("gu duration", 28)
        self.consmp_dur = env_info.get("consumption duration", 0)
        self.mean_reward_time = env_info.get("exponential distribution mean", 3)
        self.overall_reward_prob = env_info.get("overall reward probability", 0.9)
        self.bg_dur = env_info.get("bg duration", 2)

        if (self.StateStruct == "dilating"):
            self.BGTimeArray = construct_dilating_states(self.bg_dur, self.dt, self.gamma)
            self.PSTimeArray = construct_dilating_states(self.gu_dur, self.dt, self.gamma)
            self.PSMinDelay = FindStateStart(self.PSTimeArray, self.PSMinDelay)
            self.BGRewardTime = FindStateStart(self.BGTimeArray, self.BGRewardTime)
        elif (self.StateStruct == "regular"):
            self.PSTimeArray = np.round(np.arange(0, self.gu_dur + self.bg_dur, self.dt), utils.PrecisionOf(self.dt) + 1)
        else:
            raise Exception(
                str(self.StateStruct) + ' not in recognized state representation structures ["regular", "dilating"]!')

        PDFCDF = utils.pursuit_init(self.PSTimeArray, self.mean_reward_time, self.overall_reward_prob)
        self.PSRewardTimePDF = PDFCDF[0]
        self.PSRewardTimeCDF = PDFCDF[1]
        self.num_states = len(self.PSTimeArray)

        # function that finds the real optimal policy (that maximizes the global reward rate)

    def optimal_policy(self):
        PSRewardRate = np.zeros(len(self.PSTimeArray))
        # BGLeaveTime = self.BGRewardTime
        # csmp = self.consmp_dur
        # BGRewardRate = self.BGRewardAmount / BGLeaveTime
        GlobalRewardRate = np.zeros(len(self.PSTimeArray))
        total_time = self.bg_dur + self.PSTimeArray
        for i in range(0, len(self.PSTimeArray)):
            t = self.PSTimeArray[i]
            Pr_RewardedBeforeT = self.PSRewardTimeCDF[i]
            PSRewardRate[i] = Pr_RewardedBeforeT / t
            GlobalRewardRate[i] = Pr_RewardedBeforeT / total_time[i]
        optimal_giveup_index_rou_g = np.argmax(GlobalRewardRate)
        optimal_giveup_time_rou_g = self.PSTimeArray[optimal_giveup_index_rou_g]
        optimal_giveup_index_rou_l = np.argmax(PSRewardRate)
        optimal_giveup_time_rou_l = self.PSTimeArray[optimal_giveup_index_rou_l]
        return GlobalRewardRate, PSRewardRate, optimal_giveup_time_rou_g, optimal_giveup_time_rou_l

    def test_reward_function(self):
        fig, ax = plt.subplots(1, 1)
        ax.plot(self.PSTimeArray, self.PSRewardTimePDF)
        fig, bx = plt.subplots(1, 1)
        bx.plot(self.PSTimeArray, self.PSRewardTimeCDF)


# construct dilating states
def construct_dilating_states(dur, fdt, gamma):
    p = 1 / gamma

    if (p == 1):
        TimeArray = np.round(np.arange(0, dur, fdt), utils.PrecisionOf(fdt))
    else:
        state_num = 0
        while ((fdt * (1 - p ** (state_num + 1)) / (1 - p)) < dur):
            state_num += 1
        increments = np.ones(state_num + 1) * fdt
        TimeArray = np.zeros(state_num + 1)
        for ii in range(1, len(increments)):
            increments[ii] = increments[ii] / (gamma ** (ii))
            TimeArray[ii] = np.round(sum(increments[0:ii]), utils.PrecisionOf(fdt) + 1)
    #         if (len(TimeArray) == 1):
    #             last_timestamp = np.round(TimeArray[-1] + fdt, PrecisionOf(fdt) + 1)
    #         else:
    #             last_timestamp = np.round(TimeArray[-1] + (TimeArray[-1] - TimeArray[-2]) * p, PrecisionOf(fdt) + 1)

    #         TimeArray = np.append(TimeArray, last_timestamp)

    return TimeArray

def FindStateStart(TimeArray, TargetTimestamp):
    FindIndex = np.where(TimeArray <= TargetTimestamp)
    IndexArray = FindIndex[0]
    index = max(IndexArray)
    T = TimeArray[index]
    return T