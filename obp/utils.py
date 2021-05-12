# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Useful Tools."""
from typing import Dict, Optional, Union

import numpy as np
from sklearn.utils import check_random_state
import torch


def check_confidence_interval_arguments(
    alpha: float = 0.05,
    n_bootstrap_samples: int = 10000,
    random_state: Optional[int] = None,
) -> Optional[ValueError]:
    """Check confidence interval arguments.

    Parameters
    ----------
    alpha: float, default=0.05
        Significant level of confidence intervals.

    n_bootstrap_samples: int, default=10000
        Number of resampling performed in the bootstrap procedure.

    random_state: int, default=None
        Controls the random seed in bootstrap sampling.

    Returns
    ----------
    estimated_confidence_interval: Dict[str, float]
        Dictionary storing the estimated mean and upper-lower confidence bounds.

    """
    if not (isinstance(alpha, float) and (0.0 < alpha < 1.0)):
        raise ValueError(
            f"alpha must be a positive float (< 1.0), but {alpha} is given"
        )
    if not (isinstance(n_bootstrap_samples, int) and n_bootstrap_samples > 0):
        raise ValueError(
            f"n_bootstrap_samples must be a positive integer, but {n_bootstrap_samples} is given"
        )
    if random_state is not None and not isinstance(random_state, int):
        raise ValueError(
            f"random_state must be an integer, but {random_state} is given"
        )


def estimate_confidence_interval_by_bootstrap(
    samples: np.ndarray,
    alpha: float = 0.05,
    n_bootstrap_samples: int = 10000,
    random_state: Optional[int] = None,
) -> Dict[str, float]:
    """Estimate confidence interval by nonparametric bootstrap-like procedure.

    Parameters
    ----------
    samples: array-like
        Empirical observed samples to be used to estimate cumulative distribution function.

    alpha: float, default=0.05
        Significant level of confidence intervals.

    n_bootstrap_samples: int, default=10000
        Number of resampling performed in the bootstrap procedure.

    random_state: int, default=None
        Controls the random seed in bootstrap sampling.

    Returns
    ----------
    estimated_confidence_interval: Dict[str, float]
        Dictionary storing the estimated mean and upper-lower confidence bounds.

    """
    check_confidence_interval_arguments(
        alpha=alpha, n_bootstrap_samples=n_bootstrap_samples, random_state=random_state
    )

    boot_samples = list()
    random_ = check_random_state(random_state)
    for _ in np.arange(n_bootstrap_samples):
        boot_samples.append(np.mean(random_.choice(samples, size=samples.shape[0])))
    lower_bound = np.percentile(boot_samples, 100 * (alpha / 2))
    upper_bound = np.percentile(boot_samples, 100 * (1.0 - alpha / 2))
    return {
        "mean": np.mean(boot_samples),
        f"{100 * (1. - alpha)}% CI (lower)": lower_bound,
        f"{100 * (1. - alpha)}% CI (upper)": upper_bound,
    }


def convert_to_action_dist(
    n_actions: int,
    selected_actions: np.ndarray,
) -> np.ndarray:
    """Convert selected actions (output of `run_bandit_simulation`) to distribution over actions.

    Parameters
    ----------
    n_actions: int
        Number of actions.

    selected_actions: array-like, shape (n_rounds, len_list)
            Sequence of actions selected by evaluation policy
            at each round in offline bandit simulation.

    Returns
    ----------
    action_dist: array-like, shape (n_rounds, n_actions, len_list)
        Action choice probabilities (can be deterministic).

    """
    n_rounds, len_list = selected_actions.shape
    action_dist = np.zeros((n_rounds, n_actions, len_list))
    for pos in np.arange(len_list):
        selected_actions_ = selected_actions[:, pos]
        action_dist[
            np.arange(n_rounds),
            selected_actions_,
            pos * np.ones(n_rounds, int),
        ] = 1
    return action_dist


def check_bandit_feedback_inputs(
    context: np.ndarray,
    action: np.ndarray,
    reward: np.ndarray,
    expected_reward: Optional[np.ndarray] = None,
    position: Optional[np.ndarray] = None,
    pscore: Optional[np.ndarray] = None,
    action_context: Optional[np.ndarray] = None,
) -> Optional[ValueError]:
    """Check inputs for bandit learning or simulation.

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors in each round, i.e., :math:`x_t`.

    action: array-like, shape (n_rounds,)
        Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

    reward: array-like, shape (n_rounds,)
        Observed rewards (or outcome) in each round, i.e., :math:`r_t`.

    expected_reward: array-like, shape (n_rounds, n_actions), default=None
        Expected rewards (or outcome) in each round, i.e., :math:`\\mathbb{E}[r_t]`.

    position: array-like, shape (n_rounds,), default=None
        Positions of each round in the given logged bandit feedback.

    pscore: array-like, shape (n_rounds,), default=None
        Propensity scores, the probability of selecting each action by behavior policy,
        in the given logged bandit feedback.

    action_context: array-like, shape (n_actions, dim_action_context)
        Context vectors characterizing each action.

    """
    if not isinstance(context, np.ndarray):
        raise ValueError("context must be ndarray")
    if context.ndim != 2:
        raise ValueError("context must be 2-dimensional")
    if not isinstance(action, np.ndarray):
        raise ValueError("action must be ndarray")
    if action.ndim != 1:
        raise ValueError("action must be 1-dimensional")
    if not isinstance(reward, np.ndarray):
        raise ValueError("reward must be ndarray")
    if reward.ndim != 1:
        raise ValueError("reward must be 1-dimensional")
    if not (np.issubdtype(action.dtype, np.integer) and action.min() >= 0):
        raise ValueError("action elements must be non-negative integers")

    if expected_reward is not None:
        if not isinstance(expected_reward, np.ndarray):
            raise ValueError("expected_reward must be ndarray")
        if expected_reward.ndim != 2:
            raise ValueError("expected_reward must be 2-dimensional")
        if not (
            context.shape[0]
            == action.shape[0]
            == reward.shape[0]
            == expected_reward.shape[0]
        ):
            raise ValueError(
                "context, action, reward, and expected_reward must be the same size."
            )
        if action.max() >= expected_reward.shape[1]:
            raise ValueError(
                "action elements must be smaller than the size of the second dimension of expected_reward"
            )
    if pscore is not None:
        if not isinstance(pscore, np.ndarray):
            raise ValueError("pscore must be ndarray")
        if pscore.ndim != 1:
            raise ValueError("pscore must be 1-dimensional")
        if not (
            context.shape[0] == action.shape[0] == reward.shape[0] == pscore.shape[0]
        ):
            raise ValueError(
                "context, action, reward, and pscore must be the same size."
            )
        if np.any(pscore <= 0):
            raise ValueError("pscore must be positive")

    if position is not None:
        if not isinstance(position, np.ndarray):
            raise ValueError("position must be ndarray")
        if position.ndim != 1:
            raise ValueError("position must be 1-dimensional")
        if not (
            context.shape[0] == action.shape[0] == reward.shape[0] == position.shape[0]
        ):
            raise ValueError(
                "context, action, reward, and position must be the same size."
            )
        if not (np.issubdtype(position.dtype, np.integer) and position.min() >= 0):
            raise ValueError("position elements must be non-negative integers")
    else:
        if not (context.shape[0] == action.shape[0] == reward.shape[0]):
            raise ValueError("context, action, and reward must be the same size.")
    if action_context is not None:
        if not isinstance(action_context, np.ndarray):
            raise ValueError("action_context must be ndarray")
        if action_context.ndim != 2:
            raise ValueError("action_context must be 2-dimensional")
        if action.max() >= action_context.shape[0]:
            raise ValueError(
                "action elements must be smaller than the size of the first dimension of action_context"
            )

def check_continuous_bandit_feedback_inputs(
    context: np.ndarray,
    action_by_behavior_policy: np.ndarray,
    reward: np.ndarray,
    expected_reward: Optional[np.ndarray] = None,
    pscore: Optional[np.ndarray] = None,
) -> Optional[ValueError]:
    """Check inputs for bandit learning or simulation.

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors in each round, i.e., :math:`x_t`.

    action_by_behavior_policy: array-like, shape (n_rounds,)
        Continuous action values sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

    reward: array-like, shape (n_rounds,)
        Observed rewards (or outcome) in each round, i.e., :math:`r_t`.

    expected_reward: array-like, shape (n_rounds, n_actions), default=None
        Expected rewards (or outcome) in each round, i.e., :math:`\\mathbb{E}[r_t]`.

    pscore: array-like, shape (n_rounds,), default=None
        Probability densities of the continuous action values sampled by a behavior policy
        (generalized propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

    """
    if not isinstance(context, np.ndarray):
        raise ValueError("context must be ndarray")
    if context.ndim != 2:
        raise ValueError("context must be 2-dimensional")
    if not isinstance(action_by_behavior_policy, np.ndarray):
        raise ValueError("action_by_behavior_policy must be ndarray")
    if action_by_behavior_policy.ndim != 1:
        raise ValueError("action_by_behavior_policy must be 1-dimensional")
    if not isinstance(reward, np.ndarray):
        raise ValueError("reward must be ndarray")
    if reward.ndim != 1:
        raise ValueError("reward must be 1-dimensional")

    if expected_reward is not None:
        if not isinstance(expected_reward, np.ndarray):
            raise ValueError("expected_reward must be ndarray")
        if expected_reward.ndim != 1:
            raise ValueError("expected_reward must be 1-dimensional")
        if not (
            context.shape[0]
            == action_by_behavior_policy.shape[0]
            == reward.shape[0]
            == expected_reward.shape[0]
        ):
            raise ValueError(
                "context, action, reward, and expected_reward must be the same size."
            )
    if pscore is not None:
        if not isinstance(pscore, np.ndarray):
            raise ValueError("pscore must be ndarray")
        if pscore.ndim != 1:
            raise ValueError("pscore must be 1-dimensional")
        if not (
            context.shape[0] == action_by_behavior_policy.shape[0] == reward.shape[0] == pscore.shape[0]
        ):
            raise ValueError(
                "context, action, reward, and pscore must be the same size."
            )
        if np.any(pscore <= 0):
            raise ValueError("pscore must be positive")


def check_ope_inputs(
    action_dist: np.ndarray,
    position: Optional[np.ndarray] = None,
    action: Optional[np.ndarray] = None,
    reward: Optional[np.ndarray] = None,
    pscore: Optional[np.ndarray] = None,
    estimated_rewards_by_reg_model: Optional[np.ndarray] = None,
) -> Optional[ValueError]:
    """Check inputs for bandit learning or simulation.

    Parameters
    -----------
    action_dist: array-like, shape (n_rounds, n_actions, len_list)
        Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

    position: array-like, shape (n_rounds,), default=None
        Positions of each round in the given logged bandit feedback.

    action: array-like, shape (n_rounds,), default=None
        Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

    reward: array-like, shape (n_rounds,), default=None
        Observed rewards (or outcome) in each round, i.e., :math:`r_t`.

    pscore: array-like, shape (n_rounds,), default=None
        Propensity scores, the probability of selecting each action by behavior policy,
        in the given logged bandit feedback.

    estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list), default=None
        Expected rewards for each round, action, and position estimated by a regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

    """
    # action_dist
    if not isinstance(action_dist, np.ndarray):
        raise ValueError("action_dist must be ndarray")
    if action_dist.ndim != 3:
        raise ValueError(
            f"action_dist.ndim must be 3-dimensional, but is {action_dist.ndim}"
        )
    if not np.allclose(action_dist.sum(axis=1), 1):
        raise ValueError("action_dist must be a probability distribution")

    # position
    if position is not None:
        if not isinstance(position, np.ndarray):
            raise ValueError("position must be ndarray")
        if position.ndim != 1:
            raise ValueError("position must be 1-dimensional")
        if not (position.shape[0] == action_dist.shape[0]):
            raise ValueError(
                "the first dimension of position and the first dimension of action_dist must be the same"
            )
        if not (np.issubdtype(position.dtype, np.integer) and position.min() >= 0):
            raise ValueError("position elements must be non-negative integers")
        if position.max() >= action_dist.shape[2]:
            raise ValueError(
                "position elements must be smaller than the third dimension of action_dist"
            )
    elif action_dist.shape[2] > 1:
        raise ValueError(
            "position elements must be given when the third dimension of action_dist is greater than 1"
        )

    # estimated_rewards_by_reg_model
    if estimated_rewards_by_reg_model is not None:
        if not isinstance(estimated_rewards_by_reg_model, np.ndarray):
            raise ValueError("estimated_rewards_by_reg_model must be ndarray")
        if estimated_rewards_by_reg_model.shape != action_dist.shape:
            raise ValueError(
                "estimated_rewards_by_reg_model.shape must be the same as action_dist.shape"
            )

    # action, reward
    if action is not None or reward is not None:
        if not isinstance(action, np.ndarray):
            raise ValueError("action must be ndarray")
        if action.ndim != 1:
            raise ValueError("action must be 1-dimensional")
        if not isinstance(reward, np.ndarray):
            raise ValueError("reward must be ndarray")
        if reward.ndim != 1:
            raise ValueError("reward must be 1-dimensional")
        if not (action.shape[0] == reward.shape[0]):
            raise ValueError("action and reward must be the same size.")
        if not (np.issubdtype(action.dtype, np.integer) and action.min() >= 0):
            raise ValueError("action elements must be non-negative integers")
        if action.max() >= action_dist.shape[1]:
            raise ValueError(
                "action elements must be smaller than the second dimension of action_dist"
            )

    # pscpre
    if pscore is not None:
        if not isinstance(pscore, np.ndarray):
            raise ValueError("pscore must be ndarray")
        if pscore.ndim != 1:
            raise ValueError("pscore must be 1-dimensional")
        if not (action.shape[0] == reward.shape[0] == pscore.shape[0]):
            raise ValueError("action, reward, and pscore must be the same size.")
        if np.any(pscore <= 0):
            raise ValueError("pscore must be positive")

def check_continuous_ope_inputs(
    action_by_evaluation_policy: np.ndarray,
    action_by_behavior_policy: Optional[np.ndarray] = None,
    reward: Optional[np.ndarray] = None,
    pscore: Optional[np.ndarray] = None,
    estimated_rewards_by_reg_model: Optional[np.ndarray] = None,
) -> Optional[ValueError]:
    """Check inputs for bandit learning or simulation.

    Parameters
    -----------
    action_by_behavior_policy: array-like, shape (n_rounds,)
        Continuous action values sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

    action_by_evaluation_policy: array-like, shape (n_rounds,), default=None
        Continuous action values given by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(x_t)`.

    reward: array-like, shape (n_rounds,), default=None
        Observed rewards (or outcome) in each round, i.e., :math:`r_t`.

    pscore: array-like, shape (n_rounds,), default=None
        Probability densities of the continuous action values sampled by a behavior policy
        (generalized propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

    estimated_rewards_by_reg_model: array-like, shape (n_rounds,), default=None
            Expected rewards given context and action estimated by a regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

    """
    # action_by_evaluation_policy
    if not isinstance(action_by_evaluation_policy, np.ndarray):
        raise ValueError("action_by_evaluation_policy must be ndarray")
    if action_by_evaluation_policy.ndim != 1:
        raise ValueError(
            f"action_by_evaluation_policy must be 1-dimensional"
        )

    # estimated_rewards_by_reg_model
    if estimated_rewards_by_reg_model is not None:
        if not isinstance(estimated_rewards_by_reg_model, np.ndarray):
            raise ValueError("estimated_rewards_by_reg_model must be ndarray")
        if estimated_rewards_by_reg_model.shape != action_by_evaluation_policy.shape:
            raise ValueError(
                "estimated_rewards_by_reg_model.shape must be the same as action_by_evaluation_policy.shape"
            )

    # action, reward
    if action_by_behavior_policy is not None or reward is not None:
        if not isinstance(action_by_behavior_policy, np.ndarray):
            raise ValueError("action must be ndarray")
        if action_by_behavior_policy.ndim != 1:
            raise ValueError("action_by_behavior_policy must be 1-dimensional")
        if not isinstance(reward, np.ndarray):
            raise ValueError("reward must be ndarray")
        if reward.ndim != 1:
            raise ValueError("reward must be 1-dimensional")
        if not (action_by_behavior_policy.shape[0] == reward.shape[0]):
            raise ValueError("action_by_behavior_policy and reward must be the same size")

    # pscpre
    if pscore is not None:
        if not isinstance(pscore, np.ndarray):
            raise ValueError("pscore must be ndarray")
        if pscore.ndim != 1:
            raise ValueError("pscore must be 1-dimensional")
        if not (action_by_behavior_policy.shape[0] == reward.shape[0] == pscore.shape[0]):
            raise ValueError("action_by_behavior_policy, reward, and pscore must be the same size")
        if np.any(pscore <= 0):
            raise ValueError("pscore must be positive")


def check_ope_inputs_tensor(
    action_dist: torch.Tensor,
    position: Optional[torch.Tensor] = None,
    action: Optional[torch.Tensor] = None,
    reward: Optional[torch.Tensor] = None,
    pscore: Optional[torch.Tensor] = None,
    estimated_rewards_by_reg_model: Optional[torch.Tensor] = None,
) -> Optional[ValueError]:
    """Check inputs for bandit learning or simulation.
    This is intended for being used with NNPolicyLearner.

    Parameters
    -----------
    action_dist: Tensor, shape (n_rounds, n_actions, len_list)
        Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

    position: Tensor, shape (n_rounds,), default=None
        Positions of each round in the given logged bandit feedback.

    action: Tensor, shape (n_rounds,), default=None
        Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

    reward: Tensor, shape (n_rounds,), default=None
        Observed rewards (or outcome) in each round, i.e., :math:`r_t`.

    pscore: Tensor, shape (n_rounds,), default=None
        Propensity scores, the probability of selecting each action by behavior policy,
        in the given logged bandit feedback.

    estimated_rewards_by_reg_model: Tensor, shape (n_rounds, n_actions, len_list), default=None
        Expected rewards for each round, action, and position estimated by a regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

    """
    # action_dist
    if not isinstance(action_dist, torch.Tensor):
        raise ValueError("action_dist must be Tensor")
    if action_dist.ndim != 3:
        raise ValueError(
            f"action_dist.ndim must be 3-dimensional, but is {action_dist.ndim}"
        )
    action_dist_sum = action_dist.sum(axis=1)
    action_dist_ones = torch.ones_like(action_dist_sum)
    if not torch.allclose(action_dist_sum, action_dist_ones):
        raise ValueError("action_dist must be a probability distribution")

    # position
    if position is not None:
        if not isinstance(position, torch.Tensor):
            raise ValueError("position must be Tensor")
        if position.ndim != 1:
            raise ValueError("position must be 1-dimensional")
        if not (position.shape[0] == action_dist.shape[0]):
            raise ValueError(
                "the first dimension of position and the first dimension of action_dist must be the same"
            )
        if not (position.dtype == torch.int64 and position.min() >= 0):
            raise ValueError("position elements must be non-negative integers")
        if position.max() >= action_dist.shape[2]:
            raise ValueError(
                "position elements must be smaller than the third dimension of action_dist"
            )
    elif action_dist.shape[2] > 1:
        raise ValueError(
            "position elements must be given when the third dimension of action_dist is greater than 1"
        )

    # estimated_rewards_by_reg_model
    if estimated_rewards_by_reg_model is not None:
        if not isinstance(estimated_rewards_by_reg_model, torch.Tensor):
            raise ValueError("estimated_rewards_by_reg_model must be Tensor")
        if estimated_rewards_by_reg_model.shape != action_dist.shape:
            raise ValueError(
                "estimated_rewards_by_reg_model.shape must be the same as action_dist.shape"
            )

    # action, reward
    if action is not None or reward is not None:
        if not isinstance(action, torch.Tensor):
            raise ValueError("action must be Tensor")
        if action.ndim != 1:
            raise ValueError("action must be 1-dimensional")
        if not isinstance(reward, torch.Tensor):
            raise ValueError("reward must be Tensor")
        if reward.ndim != 1:
            raise ValueError("reward must be 1-dimensional")
        if not (action.shape[0] == reward.shape[0]):
            raise ValueError("action and reward must be the same size.")
        if not (action.dtype == torch.int64 and action.min() >= 0):
            raise ValueError("action elements must be non-negative integers")
        if action.max() >= action_dist.shape[1]:
            raise ValueError(
                "action elements must be smaller than the second dimension of action_dist"
            )

    # pscpre
    if pscore is not None:
        if not isinstance(pscore, torch.Tensor):
            raise ValueError("pscore must be Tensor")
        if pscore.ndim != 1:
            raise ValueError("pscore must be 1-dimensional")
        if not (action.shape[0] == reward.shape[0] == pscore.shape[0]):
            raise ValueError("action, reward, and pscore must be the same size.")
        if torch.any(pscore <= 0):
            raise ValueError("pscore must be positive")


def check_continuous_ope_inputs_tensor(
    action_by_evaluation_policy: torch.Tensor,
    action_by_behavior_policy: Optional[torch.Tensor] = None,
    reward: Optional[torch.Tensor] = None,
    pscore: Optional[torch.Tensor] = None,
    estimated_rewards_by_reg_model: Optional[torch.Tensor] = None,
) -> Optional[ValueError]:
    """Check inputs for bandit learning or simulation.

    Parameters
    -----------
    action_by_behavior_policy: Tensor, shape (n_rounds,)
            Continuous action values sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

    action_by_evaluation_policy: Tensor, shape (n_rounds,), default=None
        Continuous action values given by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(x_t)`.

    reward: Tensor, shape (n_rounds,), default=None
        Observed rewards (or outcome) in each round, i.e., :math:`r_t`.

    pscore: Tensor, shape (n_rounds,), default=None
        Probability densities of the continuous action values sampled by a behavior policy
        (generalized propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

    estimated_rewards_by_reg_model: Tensor, shape (n_rounds,), default=None
            Expected rewards given context and action estimated by a regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

    """
    # action_by_evaluation_policy
    if not isinstance(action_by_evaluation_policy, torch.Tensor):
        raise ValueError("action_by_evaluation_policy must be Tensor")
    if action_by_evaluation_policy.ndim != 1:
        raise ValueError(
            f"action_by_evaluation_policy must be 1-dimensional"
        )

    # estimated_rewards_by_reg_model
    if estimated_rewards_by_reg_model is not None:
        if not isinstance(estimated_rewards_by_reg_model, torch.Tensor):
            raise ValueError("estimated_rewards_by_reg_model must be ndarray")
        if estimated_rewards_by_reg_model.shape != action_by_evaluation_policy.shape:
            raise ValueError(
                "estimated_rewards_by_reg_model.shape must be the same as action_by_evaluation_policy.shape"
            )

    # action, reward
    if action_by_behavior_policy is not None or reward is not None:
        if not isinstance(action_by_behavior_policy, torch.Tensor):
            raise ValueError("action must be Tensor")
        if action_by_behavior_policy.ndim != 1:
            raise ValueError("action_by_behavior_policy must be 1-dimensional")
        if not isinstance(reward, torch.Tensor):
            raise ValueError("reward must be Tensor")
        if reward.ndim != 1:
            raise ValueError("reward must be 1-dimensional")
        if not (action_by_behavior_policy.shape[0] == reward.shape[0]):
            raise ValueError("action_by_behavior_policy and reward must be the same size")

    # pscpre
    if pscore is not None:
        if not isinstance(pscore, torch.Tensor):
            raise ValueError("pscore must be Tensor")
        if pscore.ndim != 1:
            raise ValueError("pscore must be 1-dimensional")
        if not (action_by_behavior_policy.shape[0] == reward.shape[0] == pscore.shape[0]):
            raise ValueError("action_by_behavior_policy, reward, and pscore must be the same size")
        if torch.any(pscore <= 0):
            raise ValueError("pscore must be positive")


def sigmoid(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Calculate sigmoid function."""
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Calculate softmax function."""
    b = np.max(x, axis=1)[:, np.newaxis]
    numerator = np.exp(x - b)
    denominator = np.sum(numerator, axis=1)[:, np.newaxis]
    return numerator / denominator
