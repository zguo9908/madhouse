# ---------------
# Test if the action at a certain state leads to a certain other state and reward
# ---------------
from environment import GiveUpEnvironment


def test_action():
    env_info = {"state representation structure": "regular",
                "fundamental timestep": 0.05,
                "gamma": 0.9,
                "gu duration": 30,
                "bg duration": 2,
                "consumption duration": 3,
                "exponential distribution scale": 1,
                "overall reward probability": 0.9,
                "reward amount in pursuit": 10,
                "seed": 0}

    env = GiveUpEnvironment()
    # matplotlib.use('Agg')
    env.env_init(env_info)

    # env.current_state3d = (0, 1.5, 0)
    # env.env_step(0)
    # print(f'current 3d is {env.current_state3d}')
    # print(f'current 1d is {env.current_state1d}')
    # # assert(env.current_state == (0, env.dt, 0))
    #
    # env.current_state3d = (0, 1.8, 0)
    # env.env_step(1)
    # print(f'current 3d is {env.current_state3d}')
    # print(f'current 1d is {env.current_state1d}')
    # # assert(env.current_state == (0, 2, 0))
    #
    # env.current_state3d = (0, 1.95, 0)
    # env.env_step(0)
    # print(f'current 3d is {env.current_state3d}')
    # print(f'current 1d is {env.current_state1d}')
    # # assert(env.current_state == (0, 2, 0))

    env.current_state3d = (0, 1.1, 0)
    env.env_step(1)
    print(f'current 3d is {env.current_state3d}')
    print(f'current 1d is {env.current_state1d}')
    # assert(env.current_state == (1, 0, 0))

    env.current_state3d = (1, 3.1, 0)
    env.env_step(1)
    print(f'current 3d is {env.current_state3d}')
    print(f'current 1d is {env.current_state1d}')

    env.current_state3d = (1, 1.1, 1)
    env.env_step(1)
    print(f'current 3d is {env.current_state3d}')
    print(f'current 1d is {env.current_state1d}')


#test_action()


def test_reward():
    env_info = {"state representation structure": "regular",
                "fundamental timestep": 0.05,
                "gamma": 0.9,
                "gu duration": 30,
                "bg duration": 2,
                "consumption duration": 3,
                "exponential distribution scale": 1,
                "overall reward probability": 0.9,
                "reward amount in pursuit": 10,
                "seed": 0}

    env = GiveUpEnvironment()
    # matplotlib.use('Agg')
    env.env_init(env_info)
    print(env.num_states)
    env.current_state3d = (0, 1.2, 0)
    reward_state_term = env.env_step(1)
    print(env.current_state3d)
    print(reward_state_term[0], reward_state_term[1], reward_state_term[2])
    # assert(reward_state_term[0] == 0 and reward_state_term[1] == (0, env.dt, 0) and reward_state_term[2] == False)

    env.current_state3d = (1, 2.6, 0)
    # reward_state_term = env.env_step(0)
    # print(env.current_state3d)
    # print(reward_state_term[0], reward_state_term[1], reward_state_term[2])
    reward_state_term = env.env_step(1)
    print(env.current_state3d)
    print(reward_state_term[0], reward_state_term[1], reward_state_term[2])
    # assert(reward_state_term[0] ==  and reward_state_term[1] ==  and reward_state_term[2] == False)

    env.current_state3d = (1, 6.4, 1)
    print(env.current_state3d)
    reward_state_term = env.env_step(1)
    print(env.current_state3d)
    print(reward_state_term[0], reward_state_term[1], reward_state_term[2])
    # assert(reward_state_term[0] == 0 and reward_state_term[1] == (0, 0, 0) and reward_state_term[2] == False)

test_reward()