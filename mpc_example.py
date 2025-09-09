import numpy as np
import matplotlib.pyplot as plt
import do_mpc

# Define the model (nonlinear pendulum dynamics)
model_type = 'continuous'  # continuous-time system
model = do_mpc.model.Model(model_type)

# States: angle θ, angular velocity ω
theta = model.set_variable('_x', 'theta')
omega = model.set_variable('_x', 'omega')

# Control input: torque u
u = model.set_variable('_u', 'u')

# Parameters
g = 9.81
l = 1.0
m = 1.0

# Dynamics
model.set_rhs('theta', omega)
model.set_rhs('omega', (u - m*g*l*np.sin(theta)) / (m*l**2))

model.setup()

# MPC controller
mpc = do_mpc.controller.MPC(model)
setup_mpc = {
    'n_horizon': 20,
    't_step': 0.05,
    'n_robust': 0,
    'store_full_solution': True,
}
mpc.set_param(**setup_mpc)

# Cost function: swing up to theta = pi
mterm = (theta)**2 + 0.1*omega**2   # terminal cost
lterm = (theta)**2 + 0.1*omega**2 + 0.01*u**2  # stage cost
mpc.set_objective(mterm=mterm, lterm=lterm)

# Input bounds
mpc.bounds['lower','_u','u'] = -5.0
mpc.bounds['upper','_u','u'] = 5.0

mpc.setup()

# Simulator for closed-loop execution
simulator = do_mpc.simulator.Simulator(model)
simulator.set_param(t_step=0.05)
simulator.setup()

# Initial state (pendulum hanging down)
x0 = np.random.normal(loc=[[np.pi/2], [np.pi/2]], scale=0.1, size=(2,1))
mpc.x0 = x0
simulator.x0 = x0
mpc.set_initial_guess()

# Run closed-loop simulation
n_steps = 100
x_hist = []
u_hist = []

for k in range(n_steps):
    u0 = mpc.make_step(x0)
    x0 = simulator.make_step(u0)
    x_hist.append(x0)
    u_hist.append(u0)

# Convert to arrays for plotting
x_hist = np.array(x_hist).squeeze()
u_hist = np.array(u_hist).squeeze()

# Plot results
time = np.arange(n_steps)*0.05
plt.figure(figsize=(10,5))

plt.subplot(2,1,1)
plt.plot(time, x_hist[:,0], label='theta')
plt.plot(time, x_hist[:,1], label='omega')
plt.axhline(np.pi, color='r', linestyle='--', label='target')
plt.ylabel("States")
plt.legend()

plt.subplot(2,1,2)
plt.plot(time, u_hist, label='torque u')
plt.ylabel("Input")
plt.xlabel("Time [s]")
plt.legend()

plt.tight_layout()
plt.show()
