import numpy as np
from Environment.envCartPole import envCartPole
from Policy.policyCartPole import EpsilonGreedyPolicy  # Adjust the path if necessary

def simulate_cartpole_step(env, policy, eval=True, print_action=True):
    # If evaluation mode
    if eval is False:
        policy.train()

    # Test script to simulate one episode
    # Reset the environment
    state = env.reset()
    disc_state = env.discrete_reset()

    if print_action:
        print(f"Initial State: {state}")
        print(f"Initial Discrete State: {disc_state}")

    # Simulate for a few steps
    for step in range(100000):
        # Sample an action from the policy (for now just sample random action)
        action = policy.sample(disc_state)

        # Perform a step
        next_disc_state, reward, done = env.discrete_step(action)

        # Flat lined the next discrete step
        next_disc_state = env.flat_indexing(next_disc_state)

        # Update next discrete step
        disc_state = next_disc_state

        if print_action:
            print(f"Step {step+1}:")
            print(f"  Action: {action}")
            print(f"  Discrete State: {next_disc_state}")
            print(f"  Reward: {reward}")
            print(f"  Done: {done}")

        if done:
            if print_action:
                print(f"Step {step+1}:")
                print(f"  Action: {action}")
                print(f"  Discrete State: {next_disc_state}")
                print(f"  Reward: {reward}")
                print(f"  Done: {done}")
                print("Terminated.")
            break
