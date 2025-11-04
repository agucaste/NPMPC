"""Implementation of Config."""

from __future__ import annotations

import json
import os
from typing import Any

import yaml


def load_yaml(path: str) -> dict[str, Any]:
    """Get the default kwargs from ``yaml`` file.

    .. note::
        This function search the ``yaml`` file by the algorithm name and environment name. Make sure
        your new implemented algorithm or environment has the same name as the yaml file.

    Args:
        path (str): The path of the ``yaml`` file.

    Returns:
        The default kwargs.

    Raises:
        AssertionError: If the ``yaml`` file is not found.
    """

    # The following 4 lines convert #/path/SAC_config.yaml -> /path/sac_config.yaml
    # This makes it work both in mac and ubuntu
    path = make_yaml_path_lower_case(path)
    with open(path, encoding='utf-8') as file:
        try:
            kwargs = yaml.load(file, Loader=yaml.FullLoader)  # noqa: S506
        except FileNotFoundError as exc:
            raise FileNotFoundError(f'{path} error: {exc}') from exc

    return kwargs

def make_yaml_path_lower_case(path: str) -> str:
    dir_, file = os.path.split(path)
    try:
        prefix, rest = file.split("_", 1)
        file = f"{prefix.lower()}_{rest}"
    except ValueError:
        pass
    return os.path.join(dir_, file)

def recursive_check_config(
    config: dict[str, Any],
    default_config: dict[str, Any],
    exclude_keys: tuple[str, ...] = (),
) -> None:
    """Check whether config is valid in default_config.

    Args:
        config (dict[str, Any]): The config to be checked.
        default_config (dict[str, Any]): The default config.
        exclude_keys (tuple of str, optional): The keys to be excluded. Defaults to ().

    Raises:
        AssertionError: If the type of the value is not the same as the default value.
        KeyError: If the key is not in default_config.
    """
    assert isinstance(config, dict), 'custom_cfgs must be a dict!'
    for key in config:
        if key not in default_config and key not in exclude_keys:
            raise KeyError(f'Invalid key: {key}')
        if isinstance(config[key], dict):
            recursive_check_config(config[key], default_config[key])



class Config(dict):
    """Config class for storing hyperparameters.

    Hyperparameters are stored in a yaml file and loaded into a Config object.
    Then the Config class will check the hyperparameters are valid, then pass them to the algorithm class.

    Attributes:
        seed (int): seed for rng
        total_steps (int): total steps for the algorithm
        horizon (int): the mdp horizon
        gamma (float): discount factor in (0, 1)
        mu (float): suboptimality proportionality threshold in (0, 1)
        sub_opt (str): suboptimality condition: 'proportional' 'absolute' 'scaled'
        epsilon (float):  absolute suboptimality
        lipschitz (float): initial guess of lipschitz constant
        adapt_lipschitz (bool): whether to re-compute the lipschitz constant after every episode
        local_lipschitz (bool): whether to use a localized lip. constant among n-nearest neighbors or not.


    Keyword Args:
        kwargs (Any): keyword arguments to set the attributes.
    """

    seed: int
    total_steps: int
    horizon: int
    gamma: float
    mu: float
    sub_opt: str
    epsilon: float
    lipschitz: float
    adapt_lipschitz: bool
    local_lipschitz: bool
    n_neighbors: int

    def __init__(self, **kwargs: Any) -> None:
        """Initialize an instance of :class:`Config`."""
        for key, value in kwargs.items():
            if isinstance(value, dict):
                self[key] = Config.dict2config(value)
            else:
                self[key] = value

    def __getattr__(self, name: str) -> Any:
        """Get attribute."""
        try:
            return self[name]
        except KeyError:
            return super().__getattribute__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute."""
        self[name] = value

    def todict(self) -> dict[str, Any]:
        """Convert Config to dictionary.

        Returns:
            The dictionary of Config.
        """
        config_dict: dict[str, Any] = {}
        for key, value in self.items():
            if isinstance(value, Config):
                config_dict[key] = value.todict()
            else:
                config_dict[key] = value
        return config_dict

    def tojson(self) -> str:
        """Convert Config to json string.

        Returns:
            The json string of Config.
        """
        return json.dumps(self.todict(), indent=4)

    @staticmethod
    def dict2config(config_dict: dict[str, Any]) -> Config:
        """Convert dictionary to Config.

        Args:
            config_dict (dict[str, Any]): The dictionary to be converted.

        Returns:
            The algorithm config.
        """
        config = Config()
        for key, value in config_dict.items():
            if isinstance(value, dict):
                config[key] = Config.dict2config(value)
            else:
                config[key] = value
        return config

    def recursive_update(self, update_args: dict[str, Any]) -> None:
        """Recursively update args.

        Args:
            update_args (dict[str, Any]): Args to be updated.
        """
        for key, value in self.items():
            if key in update_args:
                if isinstance(update_args[key], dict):
                    if isinstance(value, Config):
                        value.recursive_update(update_args[key])
                        self[key] = value
                    else:
                        self[key] = Config.dict2config(update_args[key])
                else:
                    self[key] = update_args[key]
        for key, value in update_args.items():
            if key not in self:
                if isinstance(value, dict):
                    self[key] = Config.dict2config(value)
                else:
                    self[key] = value


def get_default_kwargs_yaml(env_id: str, algo: str = '') -> Config:
    """Get the default kwargs from ``yaml`` file.

    .. note::
        This function search the ``yaml`` file by the algorithm name and environment name. Make
        sure your new implemented algorithm or environment has the same name as the yaml file.

    Args:
        algo (str): The algorithm name.
        env_id (str): The environment name.
        algo_type (str): The algorithm type.

    Returns:
        The default kwargs.
    """
    path = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(path, '..', 'config.yaml')
    print(f'Loading {algo}.yaml from {cfg_path}')
    kwargs = load_yaml(cfg_path)
    default_kwargs = kwargs['defaults']
    env_spec_kwargs = kwargs[env_id] if env_id in kwargs else None
    # print(f"particular env specs are {env_spec_kwargs}")

    default_kwargs = Config.dict2config(default_kwargs)

    if env_spec_kwargs is not None:
        default_kwargs.recursive_update(env_spec_kwargs)

    return default_kwargs


def check_all_configs(configs: Config, algo_type: str) -> None:
    """Check all configs.

    This function is used to check the configs.

    Args:
        configs (Config): The configs to be checked.
        algo_type (str): The algorithm type.
    """
    # __check_algo_configs(configs.algo_cfgs, algo_type)
    __check_logger_configs(configs.logger_cfgs)


def __check_algo_configs(configs: Config, algo_type: str) -> None:
    """Check algorithm configs.

    This function is used to check the algorithm configs.

    .. note::
        - ``update_iters`` must be greater than 0 and must be int.
        - ``steps_per_epoch`` must be greater than 0 and must be int.
        - ``batch_size`` must be greater than 0 and must be int.
        - ``target_kl`` must be greater than 0 and must be float.
        - ``entropy_coeff`` must be in [0, 1] and must be float.
        - ``gamma`` must be in [0, 1] and must be float.
        - ``cost_gamma`` must be in [0, 1] and must be float.
        - ``lam`` must be in [0, 1] and must be float.
        - ``lam_c`` must be in [0, 1] and must be float.
        - ``clip`` must be greater than 0 and must be float.
        - ``penalty_coeff`` must be greater than 0 and must be float.
        - ``reward_normalize`` must be bool.
        - ``cost_normalize`` must be bool.
        - ``obs_normalize`` must be bool.
        - ``kl_early_stop`` must be bool.
        - ``use_max_grad_norm`` must be bool.
        - ``use_cost`` must be bool.
        - ``max_grad_norm`` must be greater than 0 and must be float.
        - ``adv_estimation_method`` must be in [``gae``, ``v-trace``, ``gae-rtg``, ``plain``].
        - ``standardized_rew_adv`` must be bool.
        - ``standardized_cost_adv`` must be bool.

    Args:
        configs (Config): The configs to be checked.
        algo_type (str): The algorithm type.
    """
    if algo_type == 'on-policy':
        assert (
            isinstance(configs.update_iters, int) and configs.update_iters > 0
        ), 'update_iters must be int and greater than 0'
        assert (
            isinstance(configs.steps_per_epoch, int) and configs.steps_per_epoch > 0
        ), 'steps_per_epoch must be int and greater than 0'
        assert (
            isinstance(configs.batch_size, int) and configs.batch_size > 0
        ), 'batch_size must be int and greater than 0'
        assert (
            isinstance(configs.target_kl, float) and configs.target_kl >= 0.0
        ), 'target_kl must be float and greater than 0.0'
        assert (
            isinstance(configs.entropy_coef, float)
            and configs.entropy_coef >= 0.0
            and configs.entropy_coef <= 1.0
        ), 'entropy_coef must be float, and it values must be [0.0, 1.0]'
        assert isinstance(configs.reward_normalize, bool), 'reward_normalize must be bool'
        assert isinstance(configs.cost_normalize, bool), 'cost_normalize must be bool'
        assert isinstance(configs.obs_normalize, bool), 'obs_normalize must be bool'
        assert isinstance(configs.kl_early_stop, bool), 'kl_early_stop must be bool'
        assert isinstance(configs.use_max_grad_norm, bool), 'use_max_grad_norm must be bool'
        assert isinstance(configs.use_critic_norm, bool), 'use_critic_norm must be bool'
        assert isinstance(configs.max_grad_norm, float) and isinstance(
            configs.critic_norm_coef,
            float,
        ), 'norm must be bool'
        assert (
            isinstance(configs.gamma, float) and configs.gamma >= 0.0 and configs.gamma <= 1.0
        ), 'gamma must be float, and it values must be [0.0, 1.0]'
        assert (
            isinstance(configs.cost_gamma, float)
            and configs.cost_gamma >= 0.0
            and configs.cost_gamma <= 1.0
        ), 'cost_gamma must be float, and it values must be [0.0, 1.0]'
        assert (
            isinstance(configs.lam, float) and configs.lam >= 0.0 and configs.lam <= 1.0
        ), 'lam must be float, and it values must be [0.0, 1.0]'
        assert (
            isinstance(configs.lam_c, float) and configs.lam_c >= 0.0 and configs.lam_c <= 1.0
        ), 'lam_c must be float, and it values must be [0.0, 1.0]'
        if hasattr(configs, 'clip'):
            assert (
                isinstance(configs.clip, float) and configs.clip >= 0.0
            ), 'clip must be float, and it values must be [0.0, infty]'
        assert isinstance(configs.adv_estimation_method, str) and configs.adv_estimation_method in [
            'gae',
            'gae-rtg',
            'vtrace',
            'plain',
        ], "adv_estimation_method must be string, and it values must be ['gae','gae-rtg','vtrace','plain']"
        assert isinstance(configs.standardized_rew_adv, bool) and isinstance(
            configs.standardized_cost_adv,
            bool,
        ), 'standardized_<>_adv must be bool'
        assert (
            isinstance(configs.penalty_coef, float)
            and configs.penalty_coef >= 0.0
            and configs.penalty_coef <= 1.0
        ), 'penalty_coef must be float, and it values must be [0.0, 1.0]'
        assert isinstance(configs.use_cost, bool), 'penalty_coef must be bool'


def __check_logger_configs(configs: Config) -> None:
    """Check logger configs.

    Args:
        configs (Config): The configs to be checked.
        algo_type (str): The algorithm type.
    """
    assert isinstance(configs.use_wandb, bool) and isinstance(
        configs.wandb_project,
        str,
    ), 'use_wandb and wandb_project must be bool and string'
    # assert isinstance(configs.use_tensorboard, bool), 'use_tensorboard must be bool'
    assert isinstance(configs.save_model_freq, int), 'save_model_freq must be int'
    if window_lens := configs.get('window_lens'):
        assert isinstance(window_lens, int), 'window_lens must be int'
    assert isinstance(configs.log_dir, str), 'log_dir must be string'