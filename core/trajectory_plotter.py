import os, time
import numpy as np
from constructor import constructor
from constructor import valid_environments
from evaluator import Evaluator, Sampler
from config import get_default_kwargs_yaml



if __name__ == "__main__":
    # for env in valid_environments:
    for env in ['constrained_lqr']:
        cfgs = get_default_kwargs_yaml(algo='', env_id=env)
        model, mpcs, simulator = constructor(env, cfgs)

        n = model.x.shape[0]  # state dimension
        G = 3  # grid points per dimension
        M = G ** n  # total number of trajectories

        anchors = [np.linspace(-1, 1, G) for _ in range(n)]
        grid = np.meshgrid(*anchors)
        X0 = np.column_stack([g.ravel() for g in grid]).reshape(M, n, 1)
        sampler = Sampler(X0=X0)

        mpc_t = mpcs[0]  # full horizon MPC
        evaluator = Evaluator(mpc=mpc_t)
        for mpc in mpcs:
            stats = evaluator.evaluate(n, mpc, simulator, cfgs, sampler, M)
            # stats = evaluator.evaluate(n, mpc, simulator, cfgs, sampler, M)


        # Path to save files -> Create based on time.
        hms_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'results',
                            env,
                            hms_time)    
        os.makedirs(path, exist_ok=True)
        
        evaluator.plot_boxplots(path=path, filename=f'bp_only_mpc_{env}_M_{M}.pdf')
        evaluator.plot_tradeoff(path=path, filename=f'tradeoff_only_mpc_{env}_M_{M}.pdf')
        evaluator.plot_trajectories(path=path, filename=f'trajectories_{env}_G_{G}_M_{M}.pdf')

        
