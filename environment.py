import numpy as np
from matplotlib import pyplot as plt

import utils


class GiveUpEnvironment:
    def __init__(self, env_info={}):
        print("passed in Keys in env_info:", env_info.keys())
        # self.StateStruct = env_info.get("state representation structure", "regular")
        # self.giveup_reward = env_info.get("reward amount in pursuit", 10)
        # self.dt = env_info.get("fundamental timestep", 0.01)
        # self.gamma = env_info.get("gamma", 0.9)
        # self.rand_generator = np.random.RandomState(env_info.get("seed", 0))
        # self.gu_dur = env_info.get("gu duration", 28)
        # self.consmp_dur = env_info.get("consumption duration", 0)
        # self.mean_reward_time = env_info.get("exponential distribution mean", 3)
        # self.overall_reward_prob = env_info.get("overall reward probability", 0.9)
        # self.bg_dur = env_info.get("bg duration", 2)
        #
        # if (self.StateStruct == "dilating"):
        #     self.BGTimeArray = construct_dilating_states(self.bg_dur, self.dt, self.gamma)
        #     self.PSTimeArray = construct_dilating_states(self.gu_dur, self.dt, self.gamma)
        #     self.PSMinDelay = FindStateStart(self.PSTimeArray, self.PSMinDelay)
        #     self.BGRewardTime = FindStateStart(self.BGTimeArray, self.BGRewardTime)
        # elif (self.StateStruct == "regular"):
        #     self.PSTimeArray = np.round(np.arange(0, self.gu_dur + self.bg_dur, self.dt),
        #                                 utils.PrecisionOf(self.dt) + 1)
        # else:
        #     raise Exception(
        #         str(self.StateStruct) + ' not in recognized state representation structures ["regular", "dilating"]!')
        #
        # PDFCDF = utils.pursuit_init(self.PSTimeArray, self.mean_reward_time, self.overall_reward_prob)
        # self.PSRewardTimePDF = PDFCDF[0]
        # self.PSRewardTimeCDF = PDFCDF[1]
        # self.num_states = len(self.PSTimeArray)

        # function that finds the real optimal policy (that maximizes the global reward rate)

    def env_init(self, env_info={}):
        """Setup for the environment called when the experiment first starts.
        Note:
            Initialize a tuple with the reward, first state, boolean
            indicating if it's terminal.
        """
        print("passed in Keys in env_info:", env_info.keys())
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
        elif (self.StateStruct == "regular"):
            self.BGTimeArray = np.round(np.arange(0, self.bg_dur, self.dt), utils.PrecisionOf(self.dt))
            self.PSTimeArray = np.round(np.arange(0, self.gu_dur + self.bg_dur, self.dt),
                                        utils.PrecisionOf(self.dt) + 1)
        else:
            raise Exception(
                str(self.StateStruct) + ' not in recognized state representation structures ["regular", "dilating"]!')

        PDFCDF = utils.pursuit_init(self.PSTimeArray, self.mean_reward_time, self.overall_reward_prob)
        self.PSRewardTimePDF = PDFCDF[0]
        self.PSRewardTimeCDF = PDFCDF[1]
        self.num_states = len(self.PSTimeArray)

        # function that finds the real optimal policy (that maximizes the global reward rate)
    def env_start(self):
        """The first method called when the episode starts, called before the
        agent starts.
        Returns:
            The first state from the environment.
        """
        reward = 0
        # current state of the agent is a 3-dimensional variable of {WhichPort, HowLongHasTheAgentWaited, RewardCollected}
        self.current_state3d = (0, 0, 0)
        self.current_state1d = self.state1d(self.current_state3d)
        termination = False
        self.reward_state_term = (reward, self.current_state1d, termination)

        return self.reward_state_term[1]

    def env_step(self, action):
        """A step taken by the environment.
        Args:
            action: The action taken by the agent.
            can be 0: wait, or 1: give up
        Returns:
            (float, state, Boolean): a tuple of the reward, state,
                and boolean indicating if it's terminal.
        """

        self.current_state1d = self.state1d(self.current_state3d)
        beh_state, time_passed, reward_collected = self.current_state3d
        # Bg or Pursuit or consumption, Time elasped in that state, Collected reward

        reward = 0  # default reward is 0
        num_rewards = 0  # default number of reward is 0. # The number of rewards has nothing to do with reward amount

        # still in background time
        if beh_state == 0:
            # stay within bg time
            if action == 0:
                if isInBounds(time_passed, self.BGTimeArray[-1]):
                    NextIndex = utils.FindIndexOfTime(self.BGTimeArray, time_passed) + 1
                    time_passed = self.BGTimeArray[NextIndex]
                # if bg passes, go to pursuit state
                if not isInBounds(time_passed, self.BGTimeArray[-1]):
                    beh_state, time_passed, reward_collected = 1, 0, 0

            # if leave/give up, then jump to the first state in the pursuit port
            elif action == 1:
                beh_state, time_passed, reward_collected = 0, 0, 0
            else:
                raise Exception(str(action) + " not in recognized actions [0: Wait, 1: Give up and leave]!")

        # inside pursuit period
        elif beh_state == 1:
            if (action == 0):
                if isInBounds(time_passed, self.PSTimeArray[-1]):
                    NextIndex = utils.FindIndexOfTime(self.PSTimeArray, time_passed) + 1
                    time_passed = self.PSTimeArray[NextIndex]

            # if leave/give up, then calculate reward probability
            # if reward is given, enter consumption state (2), start consumption time count
            elif (action == 1):
                Pr_Reward = self.PSRewardTimeCDF[utils.FindIndexOfTime(self.PSTimeArray, time_passed)]
                rand = self.rand_generator.uniform(low=0, high=1, size=1)
                if (rand < Pr_Reward):
                    reward = self.PSRewardAmount
                    num_rewards = 1
                    beh_state, time_passed, reward_collected = 0, 0, 1
                else:
                    beh_state, time_passed, reward_collected = 0, 0, 0
            else:
                raise Exception(str(action) + " not in recognized actions [0: Wait, 1: Give up and leave]!")


        # assign the new state to the environment object
        self.current_state3d = beh_state, time_passed, reward_collected
        self.current_state1d = self.state1d(self.current_state3d)
        termination = False
        self.reward_state_term = (reward, self.current_state1d, termination)
        return self.reward_state_term

    def env_cleanup(self):
        """Cleanup done after the environment ends"""
        self.current_state3d = (0, 0, 0)
        self.current_state1d = 0

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

    def state1d(self, state3d):
        beh_state = state3d[0]
        time_passed = state3d[1]
        reward_collected = state3d[2]
        if beh_state == 0:
            if reward_collected == 1:
                raise Exception(
                    f"It is not possible that no reward is delivered during background ")
            else:
                return utils.FindIndexOfTime(self.BGTimeArray, time_passed)

        elif beh_state == 1:
            if reward_collected == 0 or reward_collected == 1:
                return utils.FindIndexOfTime(self.PSTimeArray, time_passed) + len(self.BGTimeArray)

            else:
                raise Exception(str(reward_collected) +
                                " not in the possible range of number of collected rewards [0, 1]!")

        else:
            raise Exception(
                str(beh_state) + " not in recognized pursuit [0: Background, 1: Pursuit!")


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
    return TimeArray

def FindStateStart(TimeArray, TargetTimestamp):
    FindIndex = np.where(TimeArray <= TargetTimestamp)
    IndexArray = FindIndex[0]
    index = max(IndexArray)
    T = TimeArray[index]
    return T

def isInBounds(HowLongHasTheAgentWaited, MaxTime):
    return (HowLongHasTheAgentWaited >= 0) and (HowLongHasTheAgentWaited < MaxTime)