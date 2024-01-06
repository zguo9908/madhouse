import math

import numpy as np
from RLGlue import BaseAgent


class QLearningAgent(BaseAgent):
    def __init__(self, agent_info={}):
        """Setup for the agent called when the experiment first starts.

        Args:
        agent_init_info (dict), the parameters used to initialize the agent. The dictionary contains:
        {
            num_states (int): The number of states,
            num_actions (int): The number of actions,
            epsilon (float): The epsilon parameter for exploration,
            step_size (float): The step-size,
            discount (float): The discount factor,
        }

        """
        # Create a random number generator with the provided seed to seed the agent for reproducibility.

        # Store the parameters provided in agent_init_info.
        self.num_actions = agent_info["num_actions"]
        self.num_states = agent_info["num_states"]
        self.epsilon = agent_info["epsilon"]
        self.step_size = agent_info["step_size"]
        self.discount = agent_info["discount"]
        self.rand_generator = np.random.RandomState(agent_info["seed"])
        self.c = agent_info.get("degree of exploration", 0.2)
        self.exploration_method = agent_info.get("exploration method", "UCB")

        # Create an array for action-value estimates and initialize it to zero.
        self.q = 0.5 * np.ones((self.num_states, self.num_actions))  # The array of action-value estimates.
        self.delta = 0
        self.total_steps = 0
        self.SAcounter = np.zeros(
            (self.num_states, self.num_actions))  # The array of times each state-action pair is chosen.
        self.UCB = math.inf * np.ones(
            (self.num_states, self.num_actions))  # The array of upper-confidence-bounds for each state-action pair.

    def agent_init(self, agent_info):
        """Setup for the agent called when the experiment first starts.

        Args:
        agent_init_info (dict), the parameters used to initialize the agent. The dictionary contains:
        {
            num_states (int): The number of states,
            num_actions (int): The number of actions,
            epsilon (float): The epsilon parameter for exploration,
            step_size (float): The step-size,
            discount (float): The discount factor,
        }

        """
        # Store the parameters provided in agent_info.
        self.num_actions = agent_info["num_actions"]
        self.num_states = agent_info["num_states"]
        self.epsilon = agent_info["epsilon"]
        self.step_size = agent_info["step_size"]
        self.discount = agent_info["discount"]
        self.rand_generator = np.random.RandomState(agent_info["seed"])
        self.c = agent_info.get("degree of exploration", 0.2)
        self.exploration_method = agent_info.get("exploration method", "UCB")

        # Create an array for action-value estimates and initialize it to zero.
        self.q = 0.5 * np.ones((self.num_states, self.num_actions))  # The array of action-value estimates.
        self.delta = 0
        self.total_steps = 0
        self.SAcounter = np.zeros(
            (self.num_states, self.num_actions))  # The array of times each state-action pair is chosen.

    def agent_start(self, observation):
        """The first method called when the episode starts, called after
        the environment starts.
        Args:
            observation (int): the state observation from the
                environment's evn_start function.
        Returns:
            action (int): the first action the agent takes.
        """

        state = observation
        current_q = self.q[state, :]

        if (self.exploration_method == "epsilon-greedy"):
            # Choose action using epsilon greedy.
            if self.rand_generator.rand() < self.epsilon:
                action = self.rand_generator.randint(self.num_actions)
            else:
                action = self.argmax(current_q)
        elif (self.exploration_method == "UCB"):
            # Choose action using Upper-Confidence-Bound.
            action = self.argmax(self.UCB[state])
        else:
            raise Exception(
                self.exploration_method + " not in recognized exploration methods: 'epsilon-greedy' or 'UCB'!")

        self.update_UCB(state)
        self.total_steps += 1
        self.SAcounter[state, action] += 1
        self.prev_state = state
        self.prev_action = action
        return action

    def agent_step(self, reward, observation):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            observation (int): the state observation from the
                environment's step based on where the agent ended up after the
                last step.
        Returns:
            action (int): the action the agent is taking.
        """

        state = observation
        current_q = self.q[state, :]

        if (self.exploration_method == "epsilon-greedy"):
            # Choose action using epsilon greedy.
            if self.rand_generator.rand() < self.epsilon:
                action = self.rand_generator.randint(self.num_actions)
            else:
                action = self.argmax(current_q)
        elif (self.exploration_method == "UCB"):
            # Choose action using Upper-Confidence-Bound.
            action = self.argmax(self.UCB[state])
        else:
            raise Exception(
                self.exploration_method + " not in recognized exploration methods: 'epsilon-greedy' or 'UCB'!")

        # Perform an update
        # --------------------------
        # your code here
        alpha = self.step_size
        gamma = self.discount
        r = reward
        MaxQSpa = max(current_q)  # meaning max(Q(S', a))
        QSA = self.q[self.prev_state, self.prev_action]  # meaning Q(S, A)
        self.q[self.prev_state, self.prev_action] = QSA + alpha * (r + gamma * MaxQSpa - QSA)
        self.delta = r + gamma * MaxQSpa - QSA
        # --------------------------

        self.update_UCB(state)
        self.total_steps += 1
        self.SAcounter[state, action] += 1
        self.prev_state = state
        self.prev_action = action
        return action

    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        # Perform the last update in the episode
        # --------------------------
        # your code here
        alpha = self.step_size
        gamma = self.discount
        r = reward
        QSA = self.q[self.prev_state, self.prev_action]  # meaning Q(S, A)
        self.q[self.prev_state, self.prev_action] = QSA + alpha * (r - QSA)
        # --------------------------

    def agent_message(self, message):
        """A function used to pass information from the agent to the experiment.
        Args:
            message: The message passed to the agent.
        Returns:
            The response (or answer) to the message.
        """
        if message == "get_action_values":
            return self.q
        elif message == "get TD error":
            return self.delta
        else:
            raise Exception("TDAgent.agent_message(): Message not understood!")

    def update_UCB(self, state):
        """The method gets the current upper confidence bound of each
        action at a certain state.
        Args:
            state (int): the state of which the UCBs of actions are needed.

        Returns:
            UCB (array): the upper confidence bound of each action
            at a certain state.
        """
        for j in range(self.num_actions):
            if (self.SAcounter[state][j] == 0):
                self.UCB[state][j] = math.inf
            else:
                self.UCB[state][j] = self.q[state][j] + self.c * math.sqrt(
                    np.log(self.total_steps) / self.SAcounter[state][j])
                self.UCB[state][j] = np.round(self.UCB[state][j], 1)

        return self.UCB

    def argmax(self, q_values):
        """argmax with random tie-breaking
        Args:
            q_values (Numpy array): the array of action-values
        Returns:
            action (int): an action with the highest value
        """
        top = float("-inf")
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)

        return self.rand_generator.choice(ties)