# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Class for Generating Synthetic Continuous Logged Bandit Feedback."""
from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np
from scipy.stats import uniform, truncnorm
from sklearn.utils import check_random_state

from .base import BaseBanditDataset
from ..types import BanditFeedback


@dataclass
class SyntheticContinuousBanditDataset(BaseBanditDataset):
    """Class for generating synthetic continuous bandit dataset.

    Note
    -----
    By calling the `obtain_batch_bandit_feedback` method several times,
    we have different bandit samples with the same setting.
    This can be used to estimate confidence intervals of the performances of OPE estimators for continuous actions.

    If None is set as `behavior_policy_function`, the synthetic data will be context-free bandit feedback.

    Parameters
    -----------
    dim_context: int, default=1
        Number of dimensions of context vectors.

    reward_function: Callable[[np.ndarray, np.ndarray], np.ndarray]], default=None
        Function generating expected reward for each given action-context pair,
        i.e., :math:`\\mu: \\mathcal{X} \\times \\mathcal{A} \\rightarrow \\mathbb{R}`.
        If None is set, context **independent** expected reward for each action will be
        sampled from the uniform distribution automatically.

    behavior_policy_function: Callable[[np.ndarray, np.ndarray], np.ndarray], default=None
        Function generating logit value of each action in action space,
        i.e., :math:`\\f: \\mathcal{X} \\rightarrow \\mathbb{R}^{\\mathcal{A}}`.
        If None is set, context **independent** uniform distribution will be used (uniform behavior policy).

    random_state: int, default=12345
        Controls the random seed in sampling synthetic slate bandit dataset.

    dataset_name: str, default='synthetic_slate_bandit_dataset'
        Name of the dataset.

    Examples
    ----------

    .. code-block:: python

    """

    dim_context: int = 1
    reward_function: Optional[
        Callable[
            [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray
        ]
    ] = None
    behavior_policy_function: Optional[
        Callable[[np.ndarray, np.ndarray], np.ndarray]
    ] = None
    random_state: int = 12345
    dataset_name: str = "synthetic_continuous_bandit_dataset"

    def __post_init__(self) -> None:
        """Initialize Class."""
        if not isinstance(self.dim_context, int) or self.dim_context <= 0:
            raise ValueError(
                f"dim_context must be a positive integer, but {self.dim_context} is given"
            )
        if self.random_state is None:
            raise ValueError("random_state must be given")
        self.random_ = check_random_state(self.random_state)

    def _contextfree_reward_function(self, action: np.ndarray) -> np.ndarray:
        """Calculate context-free expected rewards given only continuous action values."""
        return 2 * (action ** 1.5)

    def obtain_batch_bandit_feedback(
        self,
        n_rounds: int,
        action_noise: float = 1.0,
        reward_noise: float = 1.0,
        min_action: float = -np.inf,
        max_action: float = np.inf,
    ) -> BanditFeedback:
        """Obtain batch logged bandit feedback.

        Parameters
        ----------
        n_rounds: int
            Number of rounds for synthetic bandit feedback data.

        action_noise: float, default=1.0
            Standard deviation of the Gaussian noise on the continuous action value.

        reward_noise: float, default=1.0
            Standard deviation of the Gaussian noise on the reward.

        Returns
        ---------
        bandit_feedback: BanditFeedback
            Generated synthetic bandit feedback dataset.

        """
        if not isinstance(n_rounds, int) or n_rounds <= 0:
            raise ValueError(
                f"n_rounds must be a positive integer, but {n_rounds} is given"
            )
        if not isinstance(action_noise, (int, float)) or action_noise <= 0:
            raise ValueError(
                f"action_noise must be a positive integer or a positive float, but {action_noise} is given"
            )
        if not isinstance(reward_noise, (int, float)) or reward_noise <= 0:
            raise ValueError(
                f"reward_noise must be a positive integer or a positive float, but {reward_noise} is given"
            )

        context = self.random_.normal(size=(n_rounds, self.dim_context))
        # sample actions for each round based on the behavior policy
        if self.behavior_policy_function is not None:
            expected_actions = self.behavior_policy_function(
                context=context,
                random_state=self.random_state,
            )
            a = (min_action - expected_actions) / action_noise
            b = (max_action - expected_actions) / action_noise
            action = truncnorm.rvs(
                a,
                b,
                loc=expected_actions,
                scale=action_noise,
                random_state=self.random_state,
            )
            pscore = truncnorm.pdf(
                action, a, b, loc=expected_actions, scale=action_noise
            )
        else:
            action = uniform.rvs(
                loc=min_action,
                scale=(max_action - min_action),
                size=n_rounds,
                random_state=self.random_state,
            )
            pscore = uniform.pdf(
                action, loc=min_action, scale=(max_action - min_action)
            )

        if self.reward_function is None:
            expected_reward_ = self._contextfree_reward_function(action=action)
        else:
            expected_reward_ = self.reward_function(
                context=context, action=action, random_state=self.random_state
            )
        reward = expected_reward_ + self.random_.normal(
            scale=reward_noise, size=n_rounds
        )

        return dict(
            n_rounds=n_rounds,
            context=context,
            action_by_behavior_policy=action,
            reward=reward,
            pscore=pscore,
            expected_reward=expected_reward_,
        )

    def calc_ground_truth_policy_value(
        self,
        context: np.ndarray,
        action: np.ndarray,
    ) -> float:
        """Calculate the policy value of given action distribution on the given expected_reward.

        Parameters
        -----------
        context: array-like, shape (n_rounds_of_test_data, dim_context)
            Context vectors of test data.

        action: array-like, shape (n_rounds_of_test_data,)
            Continuous action values for test data given by the evaluation policy, i.e., :math:`\\pi_e(a_t|x_t)`.

        Returns
        ----------
        policy_value: float
            The policy value of the evaluation policy on the given test bandit feedback data.

        """
        if not isinstance(context, np.ndarray) or context.ndim != 2:
            raise ValueError("context must be 2-dimensional ndarray")
        if not isinstance(action, np.ndarray):
            raise ValueError("action must be ndarray")
        if context.shape[0] != action.shape[0]:
            raise ValueError(
                "the size of axis 0 of context must be the same as that of action"
            )

        return self.reward_function(
            context=context, action=action, random_state=self.random_state
        ).mean()


# some functions to generate synthetic bandit feedback with continuous actions


def linear_reward_funcion_continuous(
    context: np.ndarray,
    action: np.ndarray,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Linear reward function to generate synthetic continuous bandit datasets.

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each round (such as user information).

    action: array-like, shape (n_rounds,)
        Continuous action values.

    random_state: int, default=None
        Controls the random seed in sampling parameters.

    Returns
    ---------
    expected_reward: array-like, shape (n_rounds,)
        Expected reward given context (:math:`x`) and continuous action (:math:`a`).

    """
    if not isinstance(context, np.ndarray) or context.ndim != 2:
        raise ValueError("context must be 2-dimensional ndarray")
    if not isinstance(action, np.ndarray):
        raise ValueError("action must be ndarray")

    random_ = check_random_state(random_state)
    coef_ = random_.normal(size=context.shape[1])
    pow_, bias = random_.uniform(size=2)
    return (np.abs(context @ coef_ - action) ** pow_) + bias


def quadratic_reward_funcion_continuous(
    context: np.ndarray,
    action: np.ndarray,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Quadratic reward function to generate synthetic continuous bandit datasets.

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each round (such as user information).

    action: array-like, shape (n_rounds,)
        Continuous action values.

    random_state: int, default=None
        Controls the random seed in sampling parameters.

    Returns
    ---------
    expected_reward: array-like, shape (n_rounds,)
        Expected reward given context (:math:`x`) and continuous action (:math:`a`).

    """
    if not isinstance(context, np.ndarray) or context.ndim != 2:
        raise ValueError("context must be 2-dimensional ndarray")
    if not isinstance(action, np.ndarray):
        raise ValueError("action must be ndarray")

    random_ = check_random_state(random_state)
    coef_x = random_.normal(size=context.shape[1])
    coef_x_a = random_.normal(size=context.shape[1])
    coef_x_a_squared = random_.normal(size=context.shape[1])
    coef_a = random_.normal(size=1)

    expected_reward = (coef_a * action) * (context @ coef_x)
    expected_reward += (context @ coef_x_a) * action
    expected_reward += (action - context @ coef_x_a_squared) ** 2
    return expected_reward


def pricing_reward_funcion_continuous(
    context: np.ndarray,
    action: np.ndarray,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Reward function imitating the personalized pricing problem to generate synthetic continuous bandit datasets.

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each round (such as user information).

    action: array-like, shape (n_rounds,)
        Continuous action values.

    random_state: int, default=None
        Controls the random seed in sampling parameters.

    Returns
    ---------
    expected_reward: array-like, shape (n_rounds,)
        Expected reward given context (:math:`x`) and continuous action (:math:`a`).
        In this function, expected reward is the expected revenue given by a personalized pricing policy.

    """
    if not isinstance(context, np.ndarray) or context.ndim != 2:
        raise ValueError("context must be 2-dimensional ndarray")
    if not isinstance(action, np.ndarray):
        raise ValueError("action must be ndarray")

    random_ = check_random_state(random_state)
    coef_1 = random_.normal(size=context.shape[1])
    coef_2 = random_.normal(size=context.shape[1])

    expected_reward = context @ coef_1
    expected_reward -= (context @ coef_2) * action
    expected_reward *= action
    return expected_reward


def linear_behavior_policy_funcion_continuous(
    context: np.ndarray,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Linear behavior policy function to generate synthetic continuous bandit datasets.

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each round (such as user information).

    random_state: int, default=None
        Controls the random seed in sampling parameters.

    Returns
    ---------
    expected_action_value: array-like, shape (n_rounds,)
        Expected continuous action values given context (:math:`x`).

    """
    if not isinstance(context, np.ndarray) or context.ndim != 2:
        raise ValueError("context must be 2-dimensional ndarray")

    random_ = check_random_state(random_state)
    coef_ = random_.normal(size=context.shape[1])
    bias = random_.uniform(size=1)
    return context @ coef_ + bias


# some functions to generate synthetic policies for continuous actions


def linear_synthetic_policy_continuous(context: np.ndarray) -> np.ndarray:
    """Linear synthtic policy

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each round (such as user information).

    Returns
    ---------
    action_by_evaluation_policy: array-like, shape (n_rounds,)
        Continuous action values given by the evaluation policy, i.e., :math:`\\pi_e(x_t)`.

    """
    return context.mean(1)


def threshold_synthetic_policy_continuous(context: np.ndarray) -> np.ndarray:
    """Threshold synthtic policy

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each round (such as user information).

    Returns
    ---------
    action_by_evaluation_policy: array-like, shape (n_rounds,)
        Continuous action values given by the evaluation policy, i.e., :math:`\\pi_e(x_t)`.

    """
    return 1.0 + np.sign(context.mean(1) - 1.5)


def sin_synthetic_policy_continuous(context: np.ndarray) -> np.ndarray:
    """Sign synthtic policy

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each round (such as user information).

    Returns
    ---------
    action_by_evaluation_policy: array-like, shape (n_rounds,)
        Continuous action values given by the evaluation policy, i.e., :math:`\\pi_e(x_t)`.

    """
    return np.sin(context.mean(1))
