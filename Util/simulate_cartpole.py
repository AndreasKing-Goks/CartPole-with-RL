# REPLAY AND SAVE BUTTON

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.widgets import Button
from Environment.envCartPole import envCartPole
from Policy.policyCartPole import EpsilonGreedyPolicy

def simulate_cartpole(env, policy, eval=True, print_action=True, save_dir='./SavedAnimations'):
    # Initially set as evaluation mode
    policy.eval()

    # If not evaluation mode
    if eval is False:
        policy.train()

    # Initialize state variables in the outer scope
    disc_state = None
    total_reward = 0
    step = 0
    ani = None
    is_done = False  # Track if the simulation has finished
    is_saving = False  # Flag to indicate if we're saving the animation

    # Time settings
    t_total = env.tspan[1]  # Total time for animation (seconds)
    dt = env.timestep  # Time step for each frame (computed with fps)

    # Set up the figure and axis
    fig, ax = plt.subplots()

    # Set the limits for the cart and pole visualization
    ax.set_xlim(-5, 5)
    ax.set_ylim(-3, 3)

    # Create the cart and pole objects
    cart_width = 1
    cart_height = 0.5
    pole_length = 0.5  # Length of the pole

    # Draw the cart as a rectangle and the pole as a line
    cart = plt.Rectangle((-cart_width / 2, 0), cart_width, cart_height, fill=True, color='blue')
    pole, = ax.plot([], [], lw=5, color='red')

    # Add the cart to the plot
    ax.add_patch(cart)

    # Add a replay button
    replay_ax = plt.axes([0.7, 0.02, 0.1, 0.04])  # Position of the replay button
    replay_button = Button(replay_ax, 'Replay')

    # Add a save button
    save_ax = plt.axes([0.82, 0.02, 0.1, 0.04])  # Position of the save button
    save_button = Button(save_ax, 'Save')

    # Initialization function for the animation
    def init():
        cart.set_xy((-cart_width / 2, -cart_height))  # Initial cart position
        pole.set_data([], [])  # Initial pole position (empty)
        return cart, pole

    # Function to update the animation
    def update(frame, ani):
        nonlocal disc_state, total_reward, step, is_done, t_total  # Use nonlocal to access outer function variables and track t_total

        # If the simulation has already finished, return None to stop updating
        if is_done and not is_saving:  # Skip early return during saving
            return cart, pole

        # Sample an action from the policy
        action = policy.sample(disc_state)

        # Perform a step in the environment
        next_disc_state, reward, done = env.discrete_step(action)

        # Flat lined the next discrete step
        next_disc_state = env.flat_indexing(next_disc_state)

        # Count step
        step += 1

        # Extract the new states
        cart_pos = env.states[0]
        pole_angle = env.states[2]

        # Update cart position
        cart.set_x(cart_pos - 0.5)

        # Update pole position (it's a line starting at the cart and extending up to the pole's tip)
        pole_x = [cart_pos, cart_pos + np.sin(pole_angle)]
        pole_y = [0, np.cos(pole_angle)]
        pole.set_data(pole_x, pole_y)

        # Update the discrete state
        disc_state = next_disc_state

        # Track reward
        total_reward += reward

        # Print action taken status (only during normal simulation, not during saving)
        if print_action and not is_saving:
            print(f"Step {step} - Move {env.action_taken[-1]}")

        # Check for termination
        if done:
            is_done = True  # Set the done flag to True
            t_total = step * env.timestep
            if not is_saving:  # Only print and stop during normal simulation
                print("Terminated, Total reward: ", total_reward)
                ani.event_source.stop()  # Stop the animation source

        return cart, pole

    # Function to reset and replay the simulation
    def replay(event):
        nonlocal ani, disc_state, total_reward, step, is_done, t_total
        # Reset the environment and animation state
        disc_state = None
        total_reward = 0
        step = 0
        is_done = False  # Reset the done flag

        # Reset environment and state
        state = env.reset()
        disc_state = env.discrete_reset()

        if print_action:
            print(f"Initial State: {state}")
            print(f"Initial Discrete State: {disc_state}")

        # Create the animation, pass ani into update via lambda
        ani = animation.FuncAnimation(
            fig, lambda frame: update(frame, ani), frames=np.arange(0, t_total, dt), init_func=init, blit=True, interval=dt*1000
        )

        # Redraw the plot
        plt.draw()

    # Function to save the animation
    def save(event):
        nonlocal ani, is_saving, disc_state, total_reward, step, is_done, t_total
        if ani is not None:
            # Set default max_tspan if t_total hasn't been dynamically set
            if t_total == 0:
                t_total = env.tspan[1]  # Use initial max_tspan if not calculated
            
            # Reset the environment for saving mode
            is_saving = True  # Indicate that we're in saving mode
            disc_state = None
            total_reward = 0
            step = 0
            is_done = False  # Reset the done flag

            # Reset environment and state before saving
            state = env.reset()
            disc_state = env.discrete_reset()

            # Ensure that the saving animation uses the correct state updates
            ani = animation.FuncAnimation(
                fig, lambda frame: update(frame, ani), frames=np.arange(0, t_total, dt), init_func=init, blit=True, interval=dt*1000
            )

            save_file_dir = save_dir + '/cartpole_simulation.mp4'
            ani.save(save_file_dir, writer='ffmpeg', fps=env.fps)
            is_saving = False  # Reset the saving flag after saving
            print("Animation saved as 'cartpole_simulation.mp4'.")

    # Connect the replay button to the replay function
    replay_button.on_clicked(replay)

    # Connect the save button to the save function
    save_button.on_clicked(save)

    # Start the first simulation
    replay(None)

    # Display the animation
    plt.show()

###########################################################################################################################
# DUNNO

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import animation
# from matplotlib.widgets import Button
# from Environment.envCartPole import envCartPole
# from Policy.policyCartPole import EpsilonGreedyPolicy

# def simulate_cartpole(env, policy, eval=True, print_action=True, save_dir='./SavedAnimations'):
# # Initially set as evaluation mode
#     policy.eval()
    
#     # If not evaluation mode
#     if eval is False:
#         policy.train()

#     # Initialize state variables in the outer scope
#     disc_state = None
#     total_reward = 0
#     step = 0
#     ani = None

#     # Time settings
#     t_total = env.tspan[1]  # Total time for animation (seconds)
#     dt = env.timestep  # Time step for each frame (computed with fps)

#     # Set up the figure and axis
#     fig, ax = plt.subplots()

#     # Set the limits for the cart and pole visualization
#     ax.set_xlim(-5, 5)
#     ax.set_ylim(-3, 3)

#     # Create the cart and pole objects
#     cart_width = 1
#     cart_height = 0.5
#     pole_length = 0.5  # Length of the pole

#     # Draw the cart as a rectangle and the pole as a line
#     cart = plt.Rectangle((-cart_width / 2, 0), cart_width, cart_height, fill=True, color='blue')
#     pole, = ax.plot([], [], lw=5, color='red')

#     # Add the cart to the plot
#     ax.add_patch(cart)

#     # Add a replay button
#     replay_ax = plt.axes([0.7, 0.02, 0.1, 0.04])  # Position of the replay button
#     replay_button = Button(replay_ax, 'Replay')

#     # Add a save button
#     save_ax = plt.axes([0.82, 0.02, 0.1, 0.04])  # Position of the save button
#     save_button = Button(save_ax, 'Save')

#     # Initialize writer
#     FFwriter = animation.FFMpegWriter(fps=10)

#     # Initialization function for the animation
#     def init():
#         cart.set_xy((-cart_width / 2, -cart_height))  # Initial cart position
#         pole.set_data([], [])  # Initial pole position (empty)
#         return cart, pole

#     # Function to update the animation
#     def update(frame, ani):
#         nonlocal disc_state, total_reward, step  # Use nonlocal to access outer function variables
#         # Sample an action from the policy
#         action = policy.sample(disc_state)

#         # Perform a step in the environment
#         next_disc_state, reward, done = env.discrete_step(action)

#         # Flat lined the next discrete step
#         next_disc_state = env.flat_indexing(next_disc_state)

#         # Count step
#         step += 1

#         # Extract the new states
#         cart_pos = env.states[0]
#         pole_angle = env.states[2]

#         # Update cart position
#         cart.set_x(cart_pos - 0.5)

#         # Update pole position (it's a line starting at the cart and extending up to the pole's tip)
#         pole_x = [cart_pos, cart_pos + np.sin(pole_angle)]
#         pole_y = [0, np.cos(pole_angle)]
#         pole.set_data(pole_x, pole_y)

#         # Update the discrete state
#         disc_state = next_disc_state

#         # Track reward
#         total_reward += reward

#         # Print action taken status
#         if print_action:
#             print(f"Step {step} - Move {env.action_taken[-1]}")

#         # Check for termination
#         if done:
#             # Print total gained reward
#             print("Terminated, Total reward: ", total_reward)
#             ani.event_source.stop()

#         return cart, pole

#     # Function to reset and replay the simulation
#     def replay(event):
#         nonlocal ani, disc_state, total_reward, step
#         # Reset the environment and animation state
#         disc_state = None
#         total_reward = 0
#         step = 0

#         # Reset environment and state
#         state = env.reset()
#         disc_state = env.discrete_reset()

#         if print_action:
#             print(f"Initial State: {state}")
#             print(f"Initial Discrete State: {disc_state}")

#         # Create the animation, pass ani into update via lambda
#         ani = animation.FuncAnimation(
#             fig, lambda frame: update(frame, ani), frames=np.arange(0, t_total, dt), init_func=init, blit=True, interval=dt*1000
#         )

#         # Redraw the plot
#         plt.draw()

#     # Function to save the animation
#     def save(event):
#         nonlocal ani
#         if ani is not None:
#             file_dir = save_dir +'/cartpole_simulation.mp4'
#             print(file_dir)
#             ani.save(file_dir, writer=FFwriter)
#             print("Animation saved as 'cartpole_simulation.mp4'.")

#     # Connect the replay button to the replay function
#     replay_button.on_clicked(replay)

#     # Connect the save button to the save function
#     save_button.on_clicked(save)

#     # Start the first simulation
#     replay(None)

#     # Display the animation
#     plt.show()

##########################################################################################################################
# NO BUTTON FUNCTIONALITY

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import animation
# from Environment.envCartPole import envCartPole
# from Policy.policyCartPole import EpsilonGreedyPolicy

# def simulate_cartpole(env, policy, eval=True, print_action=True):
#     # Initially set as evaluation mode
#     policy.eval()
    
#     # If not evaluation mode
#     if eval is False:
#         policy.train()

#     # Initialize state variables in the outer scope
#     disc_state = None
#     total_reward = 0
#     step = 0

#     # Time settings
#     t_total = env.tspan[1]  # Total time for animation (seconds)
#     dt = env.timestep  # Time step for each frame (computed with fps)

#     # Set up the figure and axis
#     fig, ax = plt.subplots()

#     # Set the limits for the cart and pole visualization
#     ax.set_xlim(-5, 5)
#     ax.set_ylim(-3, 3)

#     # Create the cart and pole objects
#     cart_width = 1
#     cart_height = 0.5
#     pole_length = 0.5  # Length of the pole

#     # Draw the cart as a rectangle and the pole as a line
#     cart = plt.Rectangle((-cart_width / 2, 0), cart_width, cart_height, fill=True, color='blue')
#     pole, = ax.plot([], [], lw=5, color='red')

#     # Add the cart to the plot
#     ax.add_patch(cart)

#     # Initialization function for the animation
#     def init():
#         cart.set_xy((-cart_width / 2, -cart_height))  # Initial cart position
#         pole.set_data([], [])  # Initial pole position (empty)
#         return cart, pole

#     # Function to update the animation
#     def update(frame):
#         nonlocal disc_state, total_reward, step  # Use nonlocal to access outer function variables
#         # Sample an action from the policy
#         action = policy.sample(disc_state)

#         # Perform a step in the environment
#         next_disc_state, reward, done = env.discrete_step(action)

#         # Flat lined the next discrete step
#         next_disc_state = env.flat_indexing(next_disc_state)

#         # Count step
#         step += 1

#         # Extract the new states
#         cart_pos = env.states[0]
#         pole_angle = env.states[2]

#         # Update cart position
#         cart.set_x(cart_pos - 0.5)

#         # Update pole position (it's a line starting at the cart and extending up to the pole's tip)
#         pole_x = [cart_pos, cart_pos + np.sin(pole_angle)]
#         pole_y = [0, np.cos(pole_angle)]
#         pole.set_data(pole_x, pole_y)

#         # Update the discrete state
#         disc_state = next_disc_state

#         # Track reward
#         total_reward += reward

#         # Print action taken status
#         if print_action:
#             print(f"Step {step} - Move {env.action_taken[-1]}")

#         # Check for termination
#         if done:
#             # Print total gained reward
#             print("Terminated, Total reward: ", total_reward)
#             ani.event_source.stop()  # Stop the animation when done

#         return cart, pole

#     # Start the first simulation
#     state = env.reset()
#     disc_state = env.discrete_reset()

#     if print_action:
#         print(f"Initial State: {state}")
#         print(f"Initial Discrete State: {disc_state}")

#     # Create the animation
#     ani = animation.FuncAnimation(
#         fig, update, frames=np.arange(0, t_total, dt), init_func=init, blit=True, interval=dt*1000
#     )

#     # Display the animation
#     plt.show()

#########################################################################################################################
# ONLY REPLAY BUTTON

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import animation
# from matplotlib.widgets import Button
# from Environment.envCartPole import envCartPole
# from Policy.policyCartPole import EpsilonGreedyPolicy

# def simulate_cartpole(env, policy, eval=True, print_action=True):
#     # If evaluation mode
#     if eval is False:
#         policy.train()

#     # Initialize state variables in the outer scope
#     disc_state = None
#     total_reward = 0
#     step = 0
#     ani = None

#     # Time settings
#     t_total = env.tspan[1]  # Total time for animation (seconds)
#     dt = env.timestep  # Time step for each frame (computed with fps)

#     # Set up the figure and axis
#     fig, ax = plt.subplots()

#     # Set the limits for the cart and pole visualization
#     ax.set_xlim(-5, 5)
#     ax.set_ylim(-3, 3)

#     # Create the cart and pole objects
#     cart_width = 1
#     cart_height = 0.5
#     pole_length = 0.5  # Length of the pole

#     # Draw the cart as a rectangle and the pole as a line
#     cart = plt.Rectangle((-cart_width / 2, 0), cart_width, cart_height, fill=True, color='blue')
#     pole, = ax.plot([], [], lw=5, color='red')

#     # Add the cart to the plot
#     ax.add_patch(cart)

#     # Add a replay button
#     replay_ax = plt.axes([0.7, 0.02, 0.1, 0.04])  # Position of the replay button
#     replay_button = Button(replay_ax, 'Replay')

#     # Initialization function for the animation
#     def init():
#         cart.set_xy((-cart_width / 2, -cart_height))  # Initial cart position
#         pole.set_data([], [])  # Initial pole position (empty)
#         return cart, pole

#     # Function to update the animation
#     def update(frame, ani):
#         nonlocal disc_state, total_reward, step  # Use nonlocal to access outer function variables
#         # Sample an action from the policy
#         action = policy.sample(disc_state)

#         # Perform a step in the environment
#         next_disc_state, reward, done = env.discrete_step(action)

#         # Flat lined the next discrete step
#         next_disc_state = env.flat_indexing(next_disc_state)

#         # Count step
#         step += 1

#         # Extract the new states
#         cart_pos = env.states[0]
#         pole_angle = env.states[2]

#         # Update cart position
#         cart.set_x(cart_pos - 0.5)

#         # Update pole position (it's a line starting at the cart and extending up to the pole's tip)
#         pole_x = [cart_pos, cart_pos + np.sin(pole_angle)]
#         pole_y = [0, np.cos(pole_angle)]
#         pole.set_data(pole_x, pole_y)

#         # Update the discrete state
#         disc_state = next_disc_state

#         # Track reward
#         total_reward += reward

#         # Print action taken status
#         if print_action:
#             print(f"Step {step} - Move {env.action_taken[-1]}")

#         # Check for termination
#         if done:
#             # Print total gained reward
#             print("Terminated, Total reward: ", total_reward)
#             ani.event_source.stop()

#         return cart, pole

#     # Function to reset and replay the simulation
#     def replay(event):
#         nonlocal ani, disc_state, total_reward, step
#         # Reset the environment and animation state
#         disc_state = None
#         total_reward = 0
#         step = 0

#         # Reset environment and state
#         state = env.reset()
#         disc_state = env.discrete_reset()

#         if print_action:
#             print(f"Initial State: {state}")
#             print(f"Initial Discrete State: {disc_state}")

#         # Create the animation, pass ani into update via lambda
#         ani = animation.FuncAnimation(
#             fig, lambda frame: update(frame, ani), frames=np.arange(0, t_total, dt), init_func=init, blit=True, interval=dt*1000
#         )

#         # Redraw the plot
#         plt.draw()

#     # Connect the replay button to the replay function
#     replay_button.on_clicked(replay)

#     # Start the first simulation
#     replay(None)

#     # Display the animation
#     plt.show()
