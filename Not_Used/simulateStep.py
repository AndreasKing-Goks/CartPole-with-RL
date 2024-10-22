import numpy as np
from Environment.envCartPole import envCartPole
from Policy.policyCartPole import EpsilonGreedyPolicy  # Adjust the path if necessary

# Initialize the environment and policy
env = envCartPole()
policy = EpsilonGreedyPolicy(env.action_space, np.zeros((env.action_space.n, env.bin_numbers)), epsilon=1.0)

# Test script to simulate one episode
def test_simulation():
    # Reset the environment
    state = env.reset()
    disc_state = env.discrete_reset()

    print(f"Initial State: {state}")
    print(f"Initial Discrete State: {disc_state}")

    # Simulate for a few steps
    for step in range(100):
        # Sample an action from the policy (for now just sample random action)
        action = policy.sample(disc_state)

        # Perform a step
        next_disc_state, reward, done = env.discrete_step(action)

        print(f"Step {step+1}:")
        print(f"  Action: {action}")
        print(f"  Discrete State: {next_disc_state}")
        print(f"  Reward: {reward}")
        print(f"  Done: {done}")

        if done:
            print("Terminated.")
            break

test_simulation()
