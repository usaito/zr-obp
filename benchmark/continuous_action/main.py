# Bandwidth
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.utils import check_random_state

# import open bandit pipeline (obp)
from obp.dataset import (
    SyntheticContinuousBanditDataset,
    linear_reward_funcion_continuous,
    quadratic_reward_funcion_continuous,
    pricing_reward_funcion_continuous,
    linear_behavior_policy_funcion_continuous,
)
from obp.policy import NNPolicyLearnerForContinuousAction

pg_methods = ["dpg", "ipw", "dr-d"]

reward_function_dict = dict(
    linear=linear_reward_funcion_continuous,
    quadratic=quadratic_reward_funcion_continuous,
    pricing=pricing_reward_funcion_continuous,
)

behavior_policy_function_dict = dict(
    linear=linear_behavior_policy_funcion_continuous,
    uniform=None,
)


def sample_hyperparameters(
    policy_hyperparams_: dict,
    experiment: str,
    random_: np.random.RandomState,
) -> dict:
    hyperparams_dict_ = dict()
    if experiment == "bandwidth":
        bandwidth = 0.1 ** random_.uniform(0, 3)
        policy_hyperparams_["bandwidth"] = bandwidth
        hyperparams_dict_["bandwidth"] = bandwidth

    elif "architecture" in experiment:
        num_layers = random_.randint(1, 5)
        num_neurons = 10 * random_.randint(1, 50)
        alpha = random_.choice(1e-6 * np.arange(0, 10001, 50))
        activation = random_.choice(["logistic", "tanh", "relu"])
        if experiment == "policy_architecture":
            policy_hyperparams_["hidden_layer_size"] = (num_neurons,) * num_layers
            policy_hyperparams_["activation"] = activation
            policy_hyperparams_["alpha"] = alpha
        elif experiment == "q_func_architecture":
            policy_hyperparams_["q_func_estimator_hyperparams"]["hidden_layer_size"] = (
                num_neurons,
            ) * num_layers
            policy_hyperparams_["q_func_estimator_hyperparams"][
                "activation"
            ] = activation
            policy_hyperparams_["q_func_estimator_hyperparams"]["alpha"] = alpha
        hyperparams_dict_["num_layers"] = num_layers
        hyperparams_dict_["num_neurons"] = num_neurons
        hyperparams_dict_["activation"] = activation
        hyperparams_dict_["alpha"] = alpha.round(6)

    elif "optimizer" in experiment:
        batch_size = int(2 ** random_.randint(4, 10))
        learning_rate_init = random_.choice(1e-6 * np.arange(0, 10001, 50))
        max_iter = int(random_.choice([100, 200, 500, 1000, 2000]))
        solver = random_.choice(["sgd", "adam"])
        if experiment == "policy_optimizer":
            policy_hyperparams_["batch_size"] = batch_size
            policy_hyperparams_["learning_rate_init"] = learning_rate_init
            policy_hyperparams_["max_iter"] = max_iter
            policy_hyperparams_["solver"] = solver
        elif experiment == "q_func_optimizer":
            policy_hyperparams_["q_func_estimator_hyperparams"][
                "batch_size"
            ] = batch_size
            policy_hyperparams_["q_func_estimator_hyperparams"][
                "learning_rate_init"
            ] = learning_rate_init
            policy_hyperparams_["q_func_estimator_hyperparams"]["max_iter"] = max_iter
            policy_hyperparams_["q_func_estimator_hyperparams"]["solver"] = solver
        hyperparams_dict_["batch_size"] = batch_size
        hyperparams_dict_["learning_rate_init"] = learning_rate_init.round(6)
        hyperparams_dict_["max_iter"] = max_iter
        hyperparams_dict_["solver"] = solver

    return policy_hyperparams_, hyperparams_dict_


def save_outputs(policy_value_dict: dict, hyperparameters_dict: dict) -> None:
    log_path = Path("./outputs")
    log_path.mkdir(exist_ok=True, parents=True)

    policy_value_df = DataFrame(policy_value_dict)
    policy_value_df.to_csv(log_path / "policy_value_df.csv")
    policy_value_df.describe().round(6).to_csv(log_path / "policy_value_summary_df.csv")
    sampled_hyperparams_df = DataFrame(hyperparameters_dict)
    for pg_method in pg_methods:
        sampled_hyperparams_df_ordered_by_policy_value = sampled_hyperparams_df.loc[
            np.argsort(-policy_value_df[pg_method].values)
        ]
        sampled_hyperparams_df_ordered_by_policy_value["policy_value"] = -np.sort(
            -policy_value_df[pg_method]
        )
        sampled_hyperparams_df_ordered_by_policy_value.reset_index(
            drop=True, inplace=True
        )
        sampled_hyperparams_df_ordered_by_policy_value.to_csv(
            log_path / f"{pg_method}_sampled_hyperparameters_df.csv"
        )

    font_size = 16
    fig_width = 10
    fig_height = 8
    xmax = 6.0
    policy_value_df = pd.read_csv(log_path / "policy_value_df.csv", index_col=0)
    for pg_method in pg_methods:
        plt.clf()
        plt.style.use("ggplot")
        plt.rcParams.update({"font.size": font_size})
        _, ax = plt.subplots(figsize=(fig_width, fig_height))
        for pg_method in pg_methods:
            sns.ecdfplot(
                policy_value_df[pg_method],
                ax=ax,
                label=pg_method,
                linewidth=3.5,
                alpha=0.7,
                complementary=True,
            )
        plt.legend(loc="upper right", fontsize=24)
        plt.xlabel("Policy Value")
        plt.ylabel("1 - CDF(Policy Value)")
        plt.xlim(0, xmax)
        plt.ylim(0, 1.1)
        plt.savefig(log_path / "cdf.png", dpi=100)


@hydra.main(config_path="./conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(cfg)
    print(f"The current working directory is {Path().cwd()}")

    # configurations
    n_runs = cfg.setting.n_runs
    n_rounds_train = cfg.setting.n_rounds_train
    n_rounds_test = cfg.setting.n_rounds_test
    dim_context = cfg.setting.dim_context
    reward_noise = cfg.setting.reward_noise
    action_noise = cfg.setting.action_noise
    min_action = cfg.setting.min_action
    max_action = cfg.setting.max_action
    n_jobs = cfg.setting.n_jobs
    sampled_hyperparameters = dict(
        policy_optimizer=["solver", "learning_rate_init", "max_iter", "batch_size"],
        q_func_optimizer=["solver", "learning_rate_init", "max_iter", "batch_size"],
        policy_architecture=["num_layers", "num_neurons", "activation", "alpha"],
        q_func_architecture=["num_layers", "num_neurons", "activation", "alpha"],
        bandwidth=["bandwidth"],
    )

    # add info to default policy hyperparameters
    policy_hyperparams = dict(cfg.policy_hyperparams)
    policy_hyperparams["hidden_layer_size"] = (50, 50)
    policy_hyperparams["dim_context"] = dim_context
    policy_hyperparams["output_space"] = (min_action, max_action)
    q_func_estimator_hyperparams = dict(cfg.q_func_hyperparams)
    q_func_estimator_hyperparams["hidden_layer_size"] = (50, 50)
    policy_hyperparams["q_func_estimator_hyperparams"] = q_func_estimator_hyperparams

    def process(i: int, policy_hyperparams: dict):
        # generate synthetic data with continuous action
        dataset = SyntheticContinuousBanditDataset(
            dim_context=dim_context,
            reward_function=reward_function_dict[cfg.setting.reward_function],
            behavior_policy_function=behavior_policy_function_dict[
                cfg.setting.behavior_policy_function
            ],
            random_state=i,
        )
        # sample new training and test sets of synthetic logged bandit feedback
        bandit_feedback_train = dataset.obtain_batch_bandit_feedback(
            n_rounds=n_rounds_train,
            reward_noise=reward_noise,
            action_noise=action_noise,
            min_action=min_action,
            max_action=max_action,
        )
        bandit_feedback_test = dataset.obtain_batch_bandit_feedback(
            n_rounds=n_rounds_test,
            reward_noise=reward_noise,
            action_noise=action_noise,
            min_action=min_action,
            max_action=max_action,
        )

        policy_value_dict_ = dict()
        # sample and set hyperparameters
        policy_hyperparams["random_state"] = int(i)
        policy_hyperparams, hyperparams_dict_ = sample_hyperparameters(
            policy_hyperparams_=policy_hyperparams,
            experiment=cfg.setting.experiment,
            random_=check_random_state(i),
        )
        for pg_method in pg_methods:
            policy_hyperparams["pg_method"] = pg_method
            # define a neural network policy
            nn_policy_learner = NNPolicyLearnerForContinuousAction(**policy_hyperparams)
            # train neural network policies on the training set of the synthetic logged bandit feedback
            nn_policy_learner.fit(
                context=bandit_feedback_train["context"],
                action_by_behavior_policy=bandit_feedback_train[
                    "action_by_behavior_policy"
                ],
                reward=bandit_feedback_train["reward"],
                pscore=bandit_feedback_train["pscore"],
            )
            action_by_nn_policy = nn_policy_learner.predict(
                context=bandit_feedback_test["context"]
            )
            policy_value_dict_[pg_method] = (
                dataset.calc_ground_truth_policy_value(
                    context=bandit_feedback_test["context"], action=action_by_nn_policy
                )
                / bandit_feedback_train["expected_reward"].mean()
            )

        return policy_value_dict_, hyperparams_dict_

    processed = Parallel(
        n_jobs=n_jobs,
        verbose=50,
    )([delayed(process)(i, policy_hyperparams) for i in np.arange(n_runs)])
    policy_value_dict = {pg_method: dict() for pg_method in pg_methods}
    hyperparameters_dict = {
        hyperparam: dict()
        for hyperparam in sampled_hyperparameters[cfg.setting.experiment]
    }
    for i, (policy_value_i, hyperparam_i) in enumerate(processed):
        for (
            pg_method_,
            value_,
        ) in policy_value_i.items():
            policy_value_dict[pg_method_][i] = value_
        for (
            hyperparam_,
            value_,
        ) in hyperparam_i.items():
            hyperparameters_dict[hyperparam_][i] = value_

    save_outputs(
        policy_value_dict=policy_value_dict,
        hyperparameters_dict=hyperparameters_dict,
    )


if __name__ == "__main__":
    main()
