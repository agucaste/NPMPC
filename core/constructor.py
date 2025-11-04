import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy

from do_mpc.model import Model
from do_mpc.controller import MPC
from do_mpc.simulator import Simulator

from config import Config, get_default_kwargs_yaml

from typing import Tuple, List

valid_environments = ['pendulum', 'min_time', 'constrained_lqr_1', 'constrained_lqr_2']

def constructor(name: str, cfgs: Config) -> Tuple[Model, List[MPC], Simulator]:
    """
    Problem constructor. Given an environment name, creates:
        - model: the model of the system,
        - mpc_T: MPC controller for the _full_ horizon.
        - mpc_H: MPC controller for the receeding horizon problem.
        - the environment simulator. 

    Args:
        name (str): The name of the environment. E.g. 'pendulum', 'min_time'   
    """
    
    model: Model
    mpcs: List[MPC]
    simulator: Simulator

    assert name in valid_environments, f"Environment '{name}' not implemented."    
    
    # Create a continuous time model.
    model = Model('continuous') if not name.startswith('constrained_lqr') else Model('discrete')
    if name == 'pendulum':
        # States: angle θ, angular velocity ω
        theta = model.set_variable('_x', 'theta')
        omega = model.set_variable('_x', 'omega')

        # Control input: torque u
        u = model.set_variable('_u', 'u')

        # Parameters
        g, m, l = cfgs.g, cfgs.m, cfgs.l

        # Dynamics
        model.set_rhs('theta', omega)
        model.set_rhs('omega', (u - m*g*l*np.sin(theta)) / (m*l**2))

        model.setup()

        # MPC controller
        mpc_T = MPC(model)
        print(f'config mpc is')
        print(cfgs.mpc)
        cfgs.mpc.n_horizon = cfgs.N  # First MPC controller has the full horizon.
        mpc_T.set_param(**cfgs.mpc.todict())
        
        # Cost function: swing up to theta = 0
        mterm = (theta)**2 + 0.1*omega**2   # terminal cost
        lterm = (theta)**2 + 0.1*omega**2 + 0.01*u**2  # stage cost
        mpc_T.set_objective(mterm=mterm, lterm=lterm)

        # Input bounds
        mpc_T.bounds['lower','_u','u'] = cfgs.u_lb
        mpc_T.bounds['upper','_u','u'] = cfgs.u_ub

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
        mpc_T = MPC(model)

        lterm = 10 * u ** 2 + lambd 
        mpc_T.set_objective(lterm=lterm, mterm=0*p)

        cfgs.mpc.n_horizon = cfgs.N  # First MPC controller has the full horizon.
        mpc_T.set_param(**cfgs.mpc.todict())
        # mpc.set_param(**setup_mpc)

        # Control constraints: -1 <= u <= 1
        mpc_T.bounds['lower','_u','u'] = cfgs.u_lb
        mpc_T.bounds['upper','_u','u'] = cfgs.u_ub

        # Terminal constraint: p == 0, q == 0 at end of horizon
        mpc_T.terminal_bounds['lower', 'p'] = 0.
        mpc_T.terminal_bounds['upper', 'p'] = 0.
        mpc_T.terminal_bounds['lower', 'q'] = 0.        
        mpc_T.terminal_bounds['upper', 'q'] = 0.

    elif name.startswith('constrained_lqr'):        
        x1 = model.set_variable('_x', 'x1')
        x2 = model.set_variable('_x', 'x2')
        u = model.set_variable('_u', 'u')
        
        # Get system matrices
        A = cfgs.A
        B = cfgs.B

        # Dynamics
        model.set_rhs('x1', A[0][0] * x1 + A[0][1] * x2 + B[0] *u)
        model.set_rhs('x2', A[1][0] * x1 + A[1][1] * x2 + B[1] *u)
        model.setup()

        # MPC controller
        mpc_T = MPC(model)

        # lterm = 1 * x1 * x1 + x2 * x2 + u * u
        lterm = 1.0 * x1**2 + 1.0 * x2**2 + 1.0 * u**2
        mpc_T.set_objective(lterm=lterm, mterm=0*x1)

        cfgs.mpc.n_horizon = cfgs.N  # First MPC controller has the full horizon.
        mpc_T.set_param(**cfgs.mpc.todict())

        # Control constraints
        mpc_T.bounds['lower','_u','u'] = cfgs.u_lb
        mpc_T.bounds['upper','_u','u'] = cfgs.u_ub

        # State constraints
        mpc_T.bounds['lower','_x','x1'] = cfgs.x_lb[0]
        mpc_T.bounds['lower','_x','x2'] = cfgs.x_lb[1]
        mpc_T.bounds['upper','_x','x1'] = cfgs.x_ub[0]
        mpc_T.bounds['upper','_x','x2'] = cfgs.x_ub[1]

        

    
    mpc_list = [mpc_T] # List of MPC controllers (different H)
    # Before setting up, copy to full horizon MPC
    for h in cfgs.mpc.n_horizons:
        mpc_h = deepcopy(mpc_T)
        mpc_h.set_param(n_horizon=h)
        mpc_list.append(mpc_h)
    
    if hasattr(cfgs, 'x_lb') and hasattr(cfgs, 'x_ub'):     
        assert cfgs.N == cfgs.mpc.n_horizons[0], "Full horizon N must match first MPC horizon."
        # First MPC solves the conservative problem   
        e = cfgs.epsilon
        mpc_T.bounds['lower','_x','x1'] += e
        mpc_T.bounds['lower','_x','x2'] += e
        mpc_T.bounds['upper','_x','x1'] -= e
        mpc_T.bounds['upper','_x','x2'] -= e
    
    # Set them up.
    for mpc in mpc_list:
        mpc.setup()

    # Simulator for closed-loop execution
    simulator = Simulator(model)
    simulator.set_param(t_step=cfgs.mpc.t_step)
    simulator.setup()

    return model, mpc_list, simulator

# Let's try it out!
if __name__ == "__main__":
    for name in valid_environments:
        cfgs = get_default_kwargs_yaml(algo='', env_id=name)        
        model, mpcs, simulator = constructor(name, cfgs)
        
        x0_shape = model.x.shape
        print(f"Environment: {name}, x0 shape: {x0_shape}")
        x0 = np.random.normal(loc=0.0, scale=1, size=x0_shape)
        # x0 = np.random.normal(loc=[[np.pi/2], [np.pi/2]], scale=0.1, size=(2,1))
        mpc_t = mpcs[0]
        mpc_t.x0 = x0
        simulator.x0 = x0
        mpc_t.set_initial_guess()

        # Run closed-loop simulation
        n_steps = 100
        x_hist = []
        u_hist = []

        for k in range(n_steps):
            u0 = mpc_t.make_step(x0)
            x0 = simulator.make_step(u0)
            x_hist.append(x0)
            u_hist.append(u0)

            print(model.x)
            

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