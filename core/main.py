


if __name__ == "__main__":
    from constructor import constructor
    from data_collector import DataCollector
    from config import get_default_kwargs_yaml
    from tqdm import trange
    import os
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate controllers on the given environment.")
    parser.add_argument('--env', type=str, default='pendulum',
                        choices=['pendulum', 'min_time', 'constrained_lqr'],
                        help='The environment to evaluate on.')
    parser.add_argument('--G', type=int, nargs='+', default=[3, 5, 7, 9],
                        help='List of grid anchors per dimension for data collection.')