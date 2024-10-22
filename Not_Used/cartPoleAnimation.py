from Environment.envCartPole import envCartPole
from Policy.policyCartPole import UniformPolicy, GreedyPolicy, EpsilonGreedyPolicy
from EstimationMethod.ActionValueEstMethod import policy_learn_QL
from Util.utilRL import sample_episode, score_policy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Assuming envCartPole class from your provided code

# Initialize environment
env = envCartPole()

# Time settings
t_total = 10  # Total time for animation (seconds)
fps = 60      # Frames per second
dt = 1 / fps  # Time step for each frame

# Force applied to the system (constant for simplicity)
env.action_space.forceRight = 0
F = env.action_space.forceRight  # You can choose either forceRight or forceLeft

# Set up figure and axis for animation
fig, ax = plt.subplots()
ax.set_xlim(-5, 5)  # X-axis limits for the cart's position
ax.set_ylim(-3, 3)  # Y-axis limits for the pole's movement

# Create the cart and pole objects
cart_width = 1
cart_height = .5
pole_length = 1  # Length of the pole

# Draw the cart as a rectangle and the pole as a line
cart = plt.Rectangle((-cart_width / 2, 0), cart_width, cart_height, fill=True, color='blue')
pole, = ax.plot([], [], lw=5, color='red')

# Add the cart to the plot
ax.add_patch(cart)

# Function to initialize the animation
def init():
    cart.set_xy((-cart_width / 2, 0))  # Initial cart position
    pole.set_data([], [])  # Initial pole position (empty)
    return cart, pole

# Function to update the cart and pole position at each frame
def update(frame):
    global env, F

    # Update the dynamics using the force applied
    dynamics = env.dynamicsCartPole(env.states, F)
    env.states += dt * dynamics  # Euler integration for state update

    # Extract the state variables
    x, _, theta, _ = env.states  # Cart position (x) and pole angle (theta)

    # Update the cart position
    cart.set_xy((x - cart_width / 2, 0))

    # Compute the pole's end point
    pole_x = x + pole_length * np.sin(theta)
    pole_y = cart_height + pole_length * np.cos(theta)

    # Update the pole position (start at cart, end at pole tip)
    pole.set_data([x, pole_x], [cart_height, pole_y])

    return cart, pole

# Create the animation
ani = FuncAnimation(fig, update, frames=np.arange(0, t_total, dt), init_func=init, blit=True, interval=dt*1000)

# Show the animation
plt.show()
