from Environment.envCartPole import envCartPole
from Policy.policyCartPole import UniformPolicy, GreedyPolicy, EpsilonGreedyPolicy
from Util.utilRL import score_policy

import numpy as np
import os
import pickle

def policy_learn_QL(env, policy, n_episode_value, n_episode_score, bin_numbers=10, alpha=0.1, gamma=0.9, max_n_steps=10000, print_every=500, save_dir='./Scores'):

    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Initialize the best score
    best_mean = -np.inf

    # Intiate training mode
    policy.train()

    # Through each episodes
    for episode in range(n_episode_value):
        # print('Episode ', episode)

        # Set the policy to handle the epsilon decay
        policy.begin_episode(episode)

        # Print for every couple episodes
        if not (episode % print_every):
            # Print current episode and epsilon
            print(f'Q-Learning episode: {episode}, epsilon: {policy.epsilon:.2f}')

            # Switch to evaluation mode
            policy.eval()

            # Do policy scoring
            score, mean = score_policy(env, policy, n_episode_score, gamma)

            # Check if this is the best score so far
            if mean > best_mean:
                best_mean = mean

                # Save the best score and the policy (Q-table)
                save_path = os.path.join(save_dir, 'best_q_table.pkl')
                with open(save_path, 'wb') as f:
                    pickle.dump({
                        'episode': episode,
                        'score': score,
                        'policy_q_table': policy.q,
                        'mean': mean
                    }, f)

                print(f'New best mean of {mean} achieved at episode {episode}. Saved to {save_path}')


            # Print score
            print('Score: {s:.2%}, Mean : {m:.2f}'.format(s=score, m=mean))

            # Switch again to training mode
            policy.train()

            # Print whitespace for readability
            print()


        # Reset the continuous state (Because it is used for the integration scheme)
        env.reset()
        # print(env.states)

        # Get initial discrete state (function already implement the flat indexing)
        disc_state = env.discrete_reset()

        # Sample an action based on the policy using the discrete state
        action = policy.sample(disc_state)

        # Do action value estimation and improvement method using SARSA
        for step in range(max_n_steps):
            # Perform forward DISCRETE step for the agent
            next_disc_state, reward, done = env.discrete_step(action)

            # Compute the flat index (update as the current next_disc_state) to access the Q tables
            next_disc_state = env.flat_indexing(next_disc_state)

            # Get next action
            next_action = policy.sample(next_disc_state)

            # print([disc_state, action])
            # print([next_disc_state, action])

            ## Update the action-value table
            # Obtained reward
            term_1 = reward
            # print('Obtained reward: ', term_1)

            # Maximum expected state-action pair value
            term_2 = gamma * np.max(policy.q[next_disc_state])
            # print('Max epected state-action pair value: ', term_2, 'at', [next_disc_state, action])

            # Current state-action pair value
            term_3 = policy.q[disc_state, action]
            # print('Current state-action pair value: ', term_3, 'at ', [disc_state, action])
        
            policy.q[disc_state, action] += alpha * (term_1 + term_2 - term_3)

            # If the episode has finished, compute the action value and then break the loop
            if done:
                # print('This episode ends')
                break

            # Set the next state-action pair as the current state-action pair for the next action-value update
            # In term of flat index
            disc_state = next_disc_state
            action = next_action
    
    return