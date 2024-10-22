import numpy as np

# Classes
class observationSpaceCartPole:
    def __init__(self, obsPos=4.8, obsVel=10, obsAng=4.8, obsAngVel=24):
        # Observation Space
        self.cartPosition = [-obsPos, obsPos]
        self.cartVelocity = [-obsVel, obsVel]
        self.poleAngle = [-obsAng/180*np.pi, obsAng/180*np.pi]          # Radians
        self.poleAngVelo = [-obsAngVel, obsAngVel]
        self.n = 4 # [x, x_dot, theta, theta_dot]

class terminalStateBoundCartPole:
    def __init__(self, tsPos=2.4, tsAng=12): # TRUE BOUND, tsPos=2.4, tsAng=12, make it easy to simulate: tsAnf=np.inf
        self.cartPositionTerminal = [-tsPos, tsPos]
        self.poleAngleTerminal = [-tsAng/180*np.pi, tsAng/180*np.pi]    # Radians

class actionSpaceCartPole:
    def __init__(self, Force=10):
        self.forceLeft = -Force                                 # Newton
        self.forceRight = Force                                 # Newton
        self.all_force = [self.forceLeft, self.forceRight]
        self.n = len(self.all_force)                            # Number of action space

    def n(self):
        n_action = len(self.all_force)
        return n_action
    
    def sample(self):
        magnitude = np.random.choice(self.all_force)
        action = self.all_force.index(magnitude)
        return action

class envCartPole:
    def __init__(self,
                 max_tspan=20,
                 fps = 60,
                 bin_numbers=10,
                 initCartPos = 0,
                 initCartVel = 0,
                 initPoleAng = 0.1,
                 initPoleAngVel = 0,):
        ### Based on Gymnasium API
        ## Instantiate the action space class as an attribute of envCartPole
        ## COMPOSITION method

        # CLASS COMPOSITION
        # Observation Space
        self.observation_space = observationSpaceCartPole()

        # Terminal State Bound
        self.terminal_state = terminalStateBoundCartPole()

        # Action Space
        self.action_space = actionSpaceCartPole()

        # Action Taken
        self.action_taken = []

        # Environment conditions
        self.fps = 60
        self.tspan = [0, max_tspan]
        self.timestep = 1/fps
        self.bin_numbers = bin_numbers
        
        # CONTINOUS SPACE
        # Initial states
        self.initial_states = np.array([initCartPos, initCartVel, initPoleAng, initPoleAngVel])

        # States
        self.states = self.initial_states

        # DISCRETE SPACE
        # Initial discrete states
        self.disc_initial_states = self.flat_indexing(self.discretizeState(self.initial_states, self.bin_numbers))

        # Discrete states
        self.disc_states = self.disc_initial_states

    def dynamicsCartPole(self, states, F):
        # Parameters 
        m = 0.1     # Mass of the pole (kg)
        M = 1     # Mass of the cart (kg)
        L = 0.5     # Length of the pole (m)
        g = -9.81    # Gravity acceleration (m/s2)
        b = 0.1     # Damping coefficient
        mu_c = 0.0005 # friction coefficient of cart on track
        mu_p = 0.000002 # friction coefficient of pole on cart

        # State variables
        x = states[0]
        x_dot = states[1]
        theta = states[2]
        theta_dot = states[3]

        # Equation of motion
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        total_mass = M + m
        pole_moment = m * L

        # Compute accelerations
        temp1 = (-F - (pole_moment * theta_dot**2 * sin_theta) + (mu_c * np.sign(x_dot))) / total_mass
        temp2 = (mu_p * theta_dot) / (m * L)
        theta_ddot = (g * sin_theta + cos_theta * temp1 - temp2) / (L * (4/3 - (m * cos_theta**2 / total_mass)))

        temp3 = ((theta_dot ** 2) * sin_theta) - (theta_ddot * cos_theta)
        x_ddot = (F + pole_moment * temp3 - mu_c * np.sign(x_dot))  / total_mass

        # Encase the dynamics results
        cartPoleDynamics = np.array([x_dot, x_ddot, theta_dot, theta_ddot])

        return cartPoleDynamics

    def discretizeState(self, states, bin_numbers):
        # Unpacks states container
        x, x_dot, theta, theta_dot = states

        # Define bin edges for discretization
        cartPosBins = np.linspace(self.observation_space.cartPosition[0], self.observation_space.cartPosition[1], bin_numbers)
        cartVelBins = np.linspace(self.observation_space.cartVelocity[0], self.observation_space.cartVelocity[1], bin_numbers)
        poleAngBins = np.linspace(self.observation_space.poleAngle[0], self.observation_space.poleAngle[1], bin_numbers)
        poleAngVelBins = np.linspace(self.observation_space.poleAngVelo[0], self.observation_space.poleAngVelo[1], bin_numbers)

        # Discretize each part of the state
        x_discrete = np.clip(np.digitize(x, cartPosBins),0 ,9)
        x_dot_discrete = np.clip(np.digitize(x_dot, cartVelBins), 0, 9)
        theta_discrete = np.clip(np.digitize(theta, poleAngBins), 0, 9)
        theta_dot_discrete = np.clip(np.digitize(theta_dot, poleAngVelBins), 0, 9)

        discrete_states = [x_discrete, x_dot_discrete, theta_discrete, theta_dot_discrete] # Made a tuple

        return discrete_states
    
    def rewardFunction(self, states):
        # Check the done (termination) status
        done = (
            states[0] < self.terminal_state.cartPositionTerminal[0] or     # x < -x_bound
            states[0] > self.terminal_state.cartPositionTerminal[1] or     # x > x_bound
            states[2] < self.terminal_state.poleAngleTerminal[0] or        # theta < -theta_bound
            states[2] > self.terminal_state.poleAngleTerminal[1]           # theta > theta_bound
        )

        # Compute reward
        reward = 1 if not done else 0 # If not in terminal state, keep giving reward, else stop
        
        return reward, done
    
    def flat_indexing(self, disc_states):
        # Continous state shape (number of bins per state variable)
        state_shape = (self.bin_numbers, self.bin_numbers, self.bin_numbers, self.bin_numbers)

        # Flat indexing
        flat_index = np.ravel_multi_index((disc_states[0], disc_states[1], disc_states[2], disc_states[3]), state_shape)

        return flat_index

    def discrete_step(self, action):
        # ONLY FOR DISCRETE OBSERVATION SPACE
        # Report the action taken
        self.action_taken.append("Left" if action==0 else "Right")

        # Compute the dynamics using current state and action
        states = self.states
        F = self.action_space.all_force[action]
        dynamics = self.dynamicsCartPole(states, F)

        # Update next states by doing integration 
        next_states = states + self.timestep * dynamics

        # Discretize the continuous next state into discrete bins
        discrete_states = self.discretizeState(next_states, self.bin_numbers)

        # Compute reward
        (reward, done) = self.rewardFunction(next_states)

        # Set next states as current states for both continuous and discrete space
        self.states = next_states
        self.disc_states = discrete_states

        # print(self.states)
        # print(reward)

        return discrete_states, reward, done
    
    def step(self, action):
        # Report the action taken
        self.action_taken.append("Left" if action==0 else "Right")

        # Compute the dynamics using current state and action
        states = self.states
        F = self.action_space.all_force[action]
        dynamics = self.dynamicsCartPole(states, F)

        # Update next states by doing integration 
        next_states = states + self.timesteps * dynamics

        # Compute reward
        (reward, done) = self.rewardFunction(next_states)

        # Set next states as current states
        self.states = next_states

        return next_states, reward, done
    
    def reset(self):
        # Reset the states to the initial states
        self.states = self.initial_states

        # Reset te action taken record
        self.action_taken = []

        # Store the resetted states
        states = self.states

        return states
    
    def discrete_reset(self):
        # Reset the states to the initial states
        self.disc_states = self.disc_initial_states

        # Reset te action taken record
        self.action_taken = []

        # Store the resetted states
        disc_states = self.disc_states

        return disc_states
    
    def preview_bins(self):
        # Define bin edges for each state variable
        cartPosBins = np.linspace(self.observation_space.cartPosition[0], self.observation_space.cartPosition[1], self.bin_numbers)
        cartVelBins = np.linspace(self.observation_space.cartVelocity[0], self.observation_space.cartVelocity[1], self.bin_numbers)
        poleAngBins = np.linspace(self.observation_space.poleAngle[0], self.observation_space.poleAngle[1], self.bin_numbers)
        poleAngVelBins = np.linspace(self.observation_space.poleAngVelo[0], self.observation_space.poleAngVelo[1], self.bin_numbers)

        # Print the bins
        print("Cart Position Bins:", cartPosBins)
        print("Cart Velocity Bins:", cartVelBins)
        print("Pole Angle Bins:", poleAngBins)
        print("Pole Angular Velocity Bins:", poleAngVelBins)

    def print_bin_ranges(self):
        # Define bin edges for each state variable
        cartPosBins = np.linspace(self.observation_space.cartPosition[0], self.observation_space.cartPosition[1], self.bin_numbers)
        cartVelBins = np.linspace(self.observation_space.cartVelocity[0], self.observation_space.cartVelocity[1], self.bin_numbers)
        poleAngBins = np.linspace(self.observation_space.poleAngle[0], self.observation_space.poleAngle[1], self.bin_numbers)
        poleAngVelBins = np.linspace(self.observation_space.poleAngVelo[0], self.observation_space.poleAngVelo[1], self.bin_numbers)

        # Helper function to print bin ranges
        def print_bin_ranges_for_variable(var_name, bins):
            print(f"\n{var_name} Bin Ranges:")
            for i in range(len(bins) - 1):
                print(f"Bin {i+1}: {bins[i]:.6f} to {bins[i+1]:.6f}")

        # Print the bin ranges for each variable
        print_bin_ranges_for_variable("Cart Position", cartPosBins)
        print_bin_ranges_for_variable("Cart Velocity", cartVelBins)
        print_bin_ranges_for_variable("Pole Angle", poleAngBins)
        print_bin_ranges_for_variable("Pole Angular Velocity", poleAngVelBins)    