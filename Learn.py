from Environment.envCartPole import envCartPole
from Policy.policyCartPole import UniformPolicy, GreedyPolicy, EpsilonGreedyPolicy
from EstimationMethod.ActionValueEstMethod import policy_learn_QL
from Util.utilRL import sample_episode, score_policy
from Util.simulate_cartpole import simulate_cartpole
from Util.simulate_cartpole_step import simulate_cartpole_step

import numpy as np
import os
import pickle
import time

# Start timer
start_time = time.time()

# Set the environment
env = envCartPole()

# FOR DEBUGGING
n_episode_value = 10  # Number of episodes required to compute the action value
n_episode_score = 10  # Number of episodes required for socring te policy
bin_numbers=10          # Bin numbers
alpha=0.1               # Learning rate
gamma=0.9               # Reward discount rate
max_n_steps=10        # Maximum number of steps in a single episodes
print_every=2         # The episodes when the score is printed
epsilon = 1.0            # Explorative coefficient
epsilon_decay = 0.9995   # Decays the epsilon as the episodes goes
epsilon_min = 0.01       # The minimum epsilon after decay

# FOR LEARNING
n_episode_value = 20000  # Number of episodes required to compute the action value
n_episode_score = 5000   # Number of episodes required for socring te policy
bin_numbers= 10          # Bin numbers
alpha=0.05               # Learning rate
gamma=0.9                # Reward discount rate
max_n_steps=10000        # Maximum number of steps in a single episodes
print_every=2500         # The episodes when t he score is printed
epsilon = 1.0            # Explorative coefficient
epsilon_decay = 0.9999   # Decays the epsilon as the episodes goes
epsilon_min = 0.01       # The minimum epsilon after decay

# Initialize the action-value table
bin_numbers = env.bin_numbers
observation_space_number = env.observation_space.n
action_space_number = env.action_space.n

init_q = np.zeros((bin_numbers ** observation_space_number, action_space_number))

# Initialize the policy, Epsilon Greedy Policy
QL_policy = EpsilonGreedyPolicy(env.action_space,
                                init_q,
                                epsilon,
                                epsilon_decay,
                                epsilon_min)

# Train the policy
QL_policy.train()
policy_learn_QL(env, 
                QL_policy, 
                n_episode_value, 
                n_episode_score, 
                bin_numbers, 
                alpha, 
                gamma, 
                max_n_steps, 
                print_every)

# Evaluate the policy
QL_policy.eval()

# Print Last episode and epsilon
print(f'Q-Learning episode:{n_episode_value}, epsilon:{QL_policy.epsilon:.2f}')

# Do policy scoring
final_score, final_mean = score_policy(env, QL_policy, n_episode_score, gamma)

# Print score
print('Final Score: {s:.2%}, Final Mean: {m:.2f}'.format(s=final_score, m=final_mean))

# Record time
end_time = time.time()
elapsed_time = -(start_time - end_time)
print()
print('CartPole - Q-Learning')
print(f"Time taken: {elapsed_time:.4f} seconds")

# Assuming the best Q-table is stored during training
best_q_table_path = './Scores/best_q_table.pkl'

# Load the best Q-table if it exists and compare scores
if os.path.exists(best_q_table_path):
    with open(best_q_table_path, 'rb') as f:
        best_data = pickle.load(f)
        best_q_table = best_data['policy_q_table']
        best_score = best_data['score']
        best_mean = best_data['mean']
        episode_best = best_data['episode']
    
    # Compare the current score with the best score
    if final_mean < best_mean:
        print(f"Current Q-table performs worse than the best Q-table from episode {episode_best}. Reverting to the best Q-table.")
        
        # Overwrite the current Q-table with the best Q-table
        QL_policy.q = best_q_table

        # Print the best score
        print(f"Reverted to the best Q-table with score: {best_score:.2%}, Mean: {best_mean:.2f}")
    else:
        print("Current Q-table is the best so far.")

else:
    print("No best Q-table found to compare with.")

# ## Checking the Q-Tables
# # Assuming QL_policy.q is your Q-value table (n x 2)
# q_table = QL_policy.q  # Replace with your actual Q-value table

# # Find rows that are exactly [0, 0]
# zero_rows = np.all(q_table == [0, 0], axis=1)

# # Count how many rows are exactly [0, 0]
# zero_row_count = np.sum(zero_rows)

# # Find rows that are not exactly [0, 0]
# non_zero_rows = np.any(q_table != [0, 0], axis=1)

# # Count how many rows are not [0, 0]
# non_zero_row_count = np.sum(non_zero_rows)

# # Get the indices of rows where at least one column has a non-zero value
# non_zero_row_indices = np.where(np.any(q_table != 0, axis=1))[0]

# # Print row numbers and corresponding rows
# for row_index in non_zero_row_indices:
#     print(f"Row {row_index}: {q_table[row_index]}")

# # Print the count of rows that are [0, 0]
# print(f"Number of rows that are exactly [0, 0]: {zero_row_count}")
# print(f"Number of rows that are not [0, 0]    : {non_zero_row_count}")

# Simulate Cartpole step
# simulate_cartpole_step(env, QL_policy, eval=True, print_action=True)

## Simulate Cartpole
simulate_cartpole(env, QL_policy, eval=True, print_action=False)

# # # Sample episodes
# observations, actions, rewards, dones = sample_episode(env, QL_policy)
# # env.reset()

# step = 0
# for observation, action, reward, done in zip(observations, actions, rewards, dones):
#     step += 1
#     print(f"Step {step}:")
#     print(f"  Action: {action}")
#     print(f"  Discrete State: {observation}")
#     print(f"  Reward: {reward}")
#     print(f"  Done: {done}")

## PRINT DESCRIPTION
# # Check terminal state bound
# print('Cart Position Bound')
# print(env.terminal_state.cartPositionTerminal)
# print('Pole Angle Bound')
# print(env.terminal_state.poleAngleTerminal)
# env.print_bin_ranges()
