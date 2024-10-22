import numpy as np

def sample_episode(env, policy, timestep=100000, reset=True):
    # Intialize transition info container
    observations = []
    actions = []
    rewards = []
    dones = []

    # If reset, initial state is now is the current state
    if reset:
        env.reset()    # The non discrete state also need to be reset
        init_states =  env.discrete_reset()
        # print(init_states)
    else: 
        init_states =  env.disc_states
    
    # Add the state to the container
    observations.append(init_states)

    # Go through the timestep
    for i in range(timestep):
        # Sample am action based on the policy
        action = policy.sample(observations[-1])

        # Obtain the next observation by stepping with action 
        discrete_state, reward, done = env.discrete_step(action)

        # Flat indexing discrete_state
        flat_discrete_state = env.flat_indexing(discrete_state)

        # Store the transition info into the containers
        observations.append(flat_discrete_state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)

        # Terminate the episodes if termination status is active
        if done:
            break

    return observations, actions, rewards, dones
        

def score_policy(env, policy, n_episodes, gamma=0.9):
    # Initialize variables and containers
    episodes_to_goal = 0
    steps_to_goal = []

    # Go through the policy for each episodes
    for episode in range(n_episodes):
        # print('Episode ', episode)
        observations, actions, rewards, _ = sample_episode(env, policy, reset=True)
        
        # print(observations)
        # print(actions)
        
        # # Record the goal and steps to goal
        # if rewards[-1] !=  0: # if the last rewards bigger than zero
        #     episodes_to_goal += 1
        #     steps_to_goal.append(len(rewards))
        
        # Record the goal and
        episodes_to_goal += 1
        steps_to_goal.append(len(rewards))
        
        score = episodes_to_goal / n_episodes if episodes_to_goal > 0 else 0
        mean = np.mean(steps_to_goal) if steps_to_goal else 0 # If steps_to_goal not empty compute the mean
        # print(steps_to_goal)
    return score, mean