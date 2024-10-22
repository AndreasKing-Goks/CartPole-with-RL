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

# Set the environment
env = envCartPole()

# Initialize the action-value table
bin_numbers = env.bin_numbers
observation_space_number = env.observation_space.n
action_space_number = env.action_space.n

init_q = np.zeros((bin_numbers ** observation_space_number, action_space_number))

# Initialize the policy, Epsilon Greedy Policy
QL_policy = EpsilonGreedyPolicy(env.action_space, init_q)

# Use the pre-trained the Q Tables
# Assuming the best Q-table is stored during training
best_q_table_path = './Scores/best_q_table6592.pkl'

# Load the best Q-table if it exists and compare scores
if os.path.exists(best_q_table_path):
    with open(best_q_table_path, 'rb') as f:
        best_data = pickle.load(f)
        best_q_table = best_data['policy_q_table']
        best_score = best_data['score']
        best_mean = best_data['mean']
        episode_best = best_data['episode']
    
        # Overwrite the current Q-table with the best Q-table
        QL_policy.q = best_q_table

        # Print the best score
        print(f"Using the policy with the best Q-table with score: {best_score:.2%}, Mean: {best_mean:.2f}")
else:
    print("Trained data is not found")

# Simulate the cartpole using the selected (improved) policy
simulate_cartpole(env, QL_policy, eval=True, print_action=False)