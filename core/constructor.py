import numpy as np
from matplotlib import pyplot as plt

from do_mpc.model import Model
from do_mpc.controller import MPC
from do_mpc.simulator import Simulator

from typing import Tuple

valid_environments = ['pendulum', 'min_time']

def constructor(name: str) -> Tuple[Model, MPC, Simulator]:
    """
    Problem constructor. Given an environment name, creates:
        - the model of the system,
        - the MPC controller,
        - the environment simulator. 

    Args:
        name (str): The name of the environment. E.g. 'pendulum', 'min_time'   
    """
    
    model: Model
    mpc: MPC
    simulator: Simulator

    assert name in valid_environments, f"Environment '{name}' not implemented."
    
    # Create a continuous time model.
    model = Model('continuous')
    if name == 'pendulum':
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
        mpc = MPC(model)
        setup_mpc = {
            'n_horizon': 20,
            't_step': 0.05,
            'n_robust': 0,
            'store_full_solution': False,
            'nlpsol_opts':{
                "ipopt.print_level": 1,  # 0-> no output. 1 -> errors only. 2-> default.
                "ipopt.sb": "yes",
                "print_time": 0,
                }
        }
        mpc.set_param(**setup_mpc)

        # Cost function: swing up to theta = 0
        mterm = (theta)**2 + 0.1*omega**2   # terminal cost
        lterm = (theta)**2 + 0.1*omega**2 + 0.01*u**2  # stage cost
        mpc.set_objective(mterm=mterm, lterm=lterm)

        # Input bounds
        mpc.bounds['lower','_u','u'] = -5.0
        mpc.bounds['upper','_u','u'] = 5.0

        mpc.setup()

        # Simulator for closed-loop execution
        simulator = Simulator(model)
        simulator.set_param(t_step=setup_mpc['t_step'])
        simulator.setup()
    
    elif name == 'min_time':        
        p = model.set_variable('_x', 'p')
        q = model.set_variable('_x', 'q')
        u = model.set_variable('_u', 'u')

        # a > 0 means unstable system
        a = .2
        lambd = 1.0
        # Dynamics
        model.set_rhs('p', a*p + q)
        model.set_rhs('q', u)
        model.setup()

        # MPC controller
        mpc = MPC(model)
        setup_mpc = {
            'n_horizon': 100,
            't_step': 0.01,
            'n_robust': 0,
            'store_full_solution': True,
            'nlpsol_opts':{
                "ipopt.print_level": 1,  # 0-> no output. 1 -> errors only. 2-> default.
                "ipopt.sb": "yes",
                "print_time": 0,
                }        
        }

        lterm = 10 * u ** 2 + lambd 
        mpc.set_objective(lterm=lterm, mterm=0*p)
        mpc.set_param(**setup_mpc)

        # Control constraints: -1 <= u <= 1
        mpc.bounds['lower','_u','u'] = -1.
        mpc.bounds['upper','_u','u'] = 1.

        # Terminal constraint: p == 0, q == 0 at end of horizon
        mpc.terminal_bounds['lower', 'p'] = 0.
        mpc.terminal_bounds['upper', 'p'] = 0.
        mpc.terminal_bounds['lower', 'q'] = 0.        
        mpc.terminal_bounds['upper', 'q'] = 0.

        mpc.setup()

        # Simulator for closed-loop execution
        simulator = Simulator(model)
        simulator.set_param(t_step=setup_mpc['t_step'])
        simulator.setup()

    return model, mpc, simulator



# Let's try it out!
if __name__ == "__main__":
    for name in valid_environments:
        model, mpc, simulator = constructor(name)
        
        x0_shape = model.x.shape
        print(f"Environment: {name}, x0 shape: {x0_shape}")
        x0 = np.random.normal(loc=0.0, scale=1, size=x0_shape)
        # x0 = np.random.normal(loc=[[np.pi/2], [np.pi/2]], scale=0.1, size=(2,1))
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

            print(model.x)
            raise ValueError
            

        # Convert to arrays for plotting
        x_hist = np.array(x_hist).squeeze()
        u_hist = np.array(u_hist).squeeze()

        # Plot results
        time = np.arange(n_steps)*0.05
        plt.figure(figsize=(10,5))

        plt.subplot(2,1,1)
        plt.plot(time, x_hist[:,0], label='theta')
        plt.plot(time, x_hist[:,1], label='omega')
        # plt.axhline(np.pi, color='r', linestyle='--', label='target')
        plt.ylabel("States")
        plt.legend()

        plt.subplot(2,1,2)
        plt.plot(time, u_hist, label='torque u')
        plt.ylabel("Input")
        plt.xlabel("Time [s]")
        plt.legend()

        plt.tight_layout()
        plt.show()