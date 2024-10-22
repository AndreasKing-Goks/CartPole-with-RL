import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from Environment.envCartPole import envCartPole
from Policy.policyCartPole import EpsilonGreedyPolicy

# Initialize the environment and policy
env = envCartPole()
policy = EpsilonGreedyPolicy(env.action_space, np.zeros((env.action_space.n, env.bin_numbers)), epsilon=1.0)

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

# Initiate total reward counter
total_reward = 0

# Initiate step counter
step = 0

# Initialization function for the animation
def init():
    cart.set_xy((-cart_width / 2, -cart_height))  # Initial cart position
    pole.set_data([], [])  # Initial pole position (empty)
    return cart, pole

# Function to update the animation
def update(frame):
    global disc_state, ani, total_reward, action_taken, step
    # Sample an action from the policy
    action = policy.sample(disc_state)

    # Perform a step in the environment
    next_disc_state, reward, done = env.discrete_step(action)

    # Count step
    step += 1

    # Extract the new states
    cart_pos = env.states[0]
    pole_angle = env.states[2]

    # print(cart_pos, pole_angle)

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

    # Print action taken status
    print("Step {s} - Move {m}".format(s=(step+1), m=env.action_taken[-1]))

    # Check for termination
    if done:
        # Print total gained reward
        print("Terminated, Total reward: ", total_reward)
        ani.event_source.stop()

    return cart, pole

# Test script to simulate one episode and animate
def test_simulation():
    global disc_state, ani, total_reward, action_taken, step
    # Reset the environment and initialize discrete state
    state = env.reset()
    disc_state = env.discrete_reset()

    print(f"Initial State: {state}")
    print(f"Initial Discrete State: {disc_state}")

    # Create the animation
    ani = animation.FuncAnimation(
        fig, update, frames=np.arange(0, t_total, dt), init_func=init, blit=True, interval=dt*1000
    )

    # Display the animation
    plt.show()

# Run the test simulation with animation
test_simulation()
