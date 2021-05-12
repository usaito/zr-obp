# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Off-Policy Estimators for Continuous Actions."""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
import torch

from ..utils import (
    estimate_confidence_interval_by_bootstrap,
    check_continuous_ope_inputs,
)

# kernel functions
def triangular_kernel(u: np.ndarray) -> np.ndarray:
    return np.clip(1 - np.abs(u), -1.0, 1.0)


def gaussian_kernel(u: np.ndarray) -> np.ndarray:
    return np.exp(-(u ** 2) / 2) / np.sqrt(2 * np.pi)


def epanechnikov_kernel(u: np.ndarray) -> np.ndarray:
    return np.clip(0.75 * (1 - u ** 2), -1.0, 1.0)


kernel_functions = dict(
    gaussian=gaussian_kernel,
    epanechnikov=epanechnikov_kernel,
    triangular=triangular_kernel,
)


@dataclass
class BaseOffPolicyEstimatorForContinuousAction(metaclass=ABCMeta):
    """Base class for OPE estimators for continuous actions."""

    @abstractmethod
    def _estimate_round_rewards(self) -> Union[np.ndarray, torch.Tensor]:
        """Estimate rewards for each round."""
        raise NotImplementedError

    @abstractmethod
    def estimate_policy_value(self) -> float:
        """Estimate policy value of an evaluation policy."""
        raise NotImplementedError

    @abstractmethod
    def estimate_interval(self) -> Dict[str, float]:
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure."""
        raise NotImplementedError


@dataclass
class KernelizedInverseProbabilityWeighting(BaseOffPolicyEstimatorForContinuousAction):
    """Kernelized Inverse Probability Weighting.

    Note
    -------
    Kernelized Inverse Probability Weighting (KernelizedIPW)
    estimates the policy value of a given evaluation policy :math:`\\pi_e` by

    .. math::


    Parameters
    ------------

    kernel: str
        Choice of kernel function.
        Must be one of "gaussian", "epanechnikov", and "triangular".

    bandwidth: float
        A bandwidth hyperparameter.
        A larger value increases bias instead of reducing variance.
        A smaller value increases variance instead of reducing bias.

    estimator_name: str, default='kernelized_ipw'.
        Name of off-policy estimator.

    References
    ------------
    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments", 2018.

    """

    kernel: str
    bandwidth: float
    estimator_name: str = "kernelized_ipw"

    def __post_init__(self) -> None:
        if self.kernel not in ["gaussian", "epanechnikov", "triangular"]:
            raise ValueError(
                f"kernel must be one of 'gaussian', 'epanechnikov', and 'triangular', but {self.kernel} is given"
            )
        if not isinstance(self.bandwidth, (int, float)) or self.bandwidth <= 0:
            raise ValueError(
                f"bandwidth must be a positive integer or a positive float, but {self.bandwidth} is given"
            )

    def _estimate_round_rewards(
        self,
        reward: Union[np.ndarray, torch.Tensor],
        action_by_behavior_policy: Union[np.ndarray, torch.Tensor],
        pscore: Union[np.ndarray, torch.Tensor],
        action_by_evaluation_policy: Union[np.ndarray, torch.Tensor],
        **kwargs,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Estimate rewards for each round.

        Parameters
        ----------
        reward: array-like or Tensor, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action_by_behavior_policy: array-like or Tensor, shape (n_rounds,)
            Continuous action values sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like or Tensor, shape (n_rounds,)
            Probability densities of the continuous action values sampled by a behavior policy
            (generalized propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_by_evaluation_policy: array-like or Tensor, shape (n_rounds,)
            Continuous action values given by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(x_t)`.

        Returns
        ----------
        estimated_rewards: array-like or Tensor, shape (n_rounds,)
            Rewards estimated by KernelizedIPW for each round.

        """
        kernel_func = kernel_functions[self.kernel]
        u = action_by_evaluation_policy - action_by_behavior_policy
        u /= self.bandwidth
        estimated_rewards = kernel_func(u) * reward / pscore
        estimated_rewards /= self.bandwidth
        return estimated_rewards

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action_by_behavior_policy: np.ndarray,
        pscore: np.ndarray,
        action_by_evaluation_policy: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Estimate policy value of an evaluation policy.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action_by_behavior_policy: array-like, shape (n_rounds,)
            Continuous action values sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like, shape (n_rounds,)
            Probability densities of the continuous action values sampled by a behavior policy
            (generalized propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_by_evaluation_policy: array-like, shape (n_rounds,)
            Continuous action values given by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(x_t)`.

        Returns
        ----------
        V_hat: float
            Estimated policy value (performance) of a given evaluation policy.

        """
        check_continuous_ope_inputs(
            reward=reward,
            action_by_behavior_policy=action_by_behavior_policy,
            pscore=pscore,
            action_by_evaluation_policy=action_by_evaluation_policy,
        )

        return self._estimate_round_rewards(
            reward=reward,
            action_by_behavior_policy=action_by_behavior_policy,
            pscore=pscore,
            action_by_evaluation_policy=action_by_evaluation_policy,
        ).mean()

    def estimate_interval(
        self,
        reward: np.ndarray,
        action_by_behavior_policy: np.ndarray,
        pscore: np.ndarray,
        action_by_evaluation_policy: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action_by_behavior_policy: array-like, shape (n_rounds,)
            Continuous action values sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like, shape (n_rounds,)
            Probability densities of the continuous action values sampled by a behavior policy
            (generalized propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_by_evaluation_policy: array-like, shape (n_rounds,)
            Continuous action values given by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(x_t)`.

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
        check_continuous_ope_inputs(
            reward=reward,
            action_by_behavior_policy=action_by_behavior_policy,
            pscore=pscore,
            action_by_evaluation_policy=action_by_evaluation_policy,
        )

        estimated_round_rewards = self._estimate_round_rewards(
            reward=reward,
            action_by_behavior_policy=action_by_behavior_policy,
            pscore=pscore,
            action_by_evaluation_policy=action_by_evaluation_policy,
        )

        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class KernelizedSelfNormalizedInverseProbabilityWeighting(
    KernelizedInverseProbabilityWeighting
):
    """Kernelized Self-Normalized Inverse Probability Weighting.

    Note
    -------
    Kernelized Self-Normalized Inverse Probability Weighting (KernelizedSNIPW)
    estimates the policy value of a given evaluation policy :math:`\\pi_e` by

    .. math::


    Parameters
    ------------

    kernel: str
        Choice of kernel function.
        Must be one of "gaussian", "epanechnikov", and "triangular".

     bandwidth: float
        A bandwidth hyperparameter.
        A larger value increases bias instead of reducing variance.
        A smaller value increases variance instead of reducing bias.

    estimator_name: str, default='kernelized_snipw'.
        Name of off-policy estimator.

    References
    ------------
    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments", 2018.

    """

    kernel: str
    bandwidth: float
    estimator_name: str = "kernelized_snipw"

    def __post_init__(self) -> None:
        if not self.kernel in ["gaussian", "epanechnikov", "triangular"]:
            raise ValueError(
                f"kernel must be one of 'gaussian', 'epanechnikov', and 'triangular', but {self.kernel} is given"
            )
        self.kernel_function = kernel_functions[self.kernel]
        if not isinstance(self.bandwidth, (int, float)) or self.bandwidth <= 0:
            raise ValueError(
                f"bandwidth must be a positive integer or a positive float, but {self.bandwidth} is given"
            )

    def _estimate_round_rewards(
        self,
        reward: Union[np.ndarray, torch.Tensor],
        action_by_behavior_policy: Union[np.ndarray, torch.Tensor],
        pscore: Union[np.ndarray, torch.Tensor],
        action_by_evaluation_policy: Union[np.ndarray, torch.Tensor],
        **kwargs,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Estimate rewards for each round.

        Parameters
        ----------
        reward: array-like or Tensor, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action_by_behavior_policy: array-like or Tensor, shape (n_rounds,)
            Continuous action values sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like or Tensor, shape (n_rounds,)
            Probability densities of the continuous action values sampled by a behavior policy
            (generalized propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_by_evaluation_policy: array-like or Tensor, shape (n_rounds,)
            Continuous action values given by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(x_t)`.

        Returns
        ----------
        estimated_rewards: array-like or Tensor, shape (n_rounds,)
            Rewards estimated by KernelizedSNIPW for each round.

        """
        kernel_func = kernel_functions[self.kernel]
        u = action_by_evaluation_policy - action_by_behavior_policy
        u /= self.bandwidth
        estimated_rewards = kernel_func(u) * reward / pscore
        estimated_rewards /= (kernel_func(u) / pscore).sum() / reward.shape[0]
        return estimated_rewards


@dataclass
class KernelizedDoublyRobust(BaseOffPolicyEstimatorForContinuousAction):
    """Kernelized Doubly Robust Estimator.

    Note
    -------
    Kernelized Doubly Robust (KernelizedDR) estimates the policy value of a given evaluation policy :math:`\\pi_e` by

    .. math::


    Parameters
    ------------

    kernel: str
        Choice of kernel function.
        Must be one of "gaussian", "epanechnikov", and "triangular".

     bandwidth: float
        A bandwidth hyperparameter.
        A larger value increases bias instead of reducing variance.
        A smaller value increases variance instead of reducing bias.

    estimator_name: str, default='kernelized_dr'.
        Name of off-policy estimator.

    References
    ------------
    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments", 2018.

    """

    kernel: str
    bandwidth: float
    estimator_name: str = "kernelized_dr"

    def __post_init__(self) -> None:
        if not self.kernel in ["gaussian", "epanechnikov", "triangular"]:
            raise ValueError(
                f"kernel must be one of 'gaussian', 'epanechnikov', and 'triangular', but {self.kernel} is given"
            )
        self.kernel_function = kernel_functions[self.kernel]
        if not isinstance(self.bandwidth, (int, float)) or self.bandwidth <= 0:
            raise ValueError(
                f"bandwidth must be a positive integer or a positive float, but {self.bandwidth} is given"
            )

    def _estimate_round_rewards(
        self,
        reward: Union[np.ndarray, torch.Tensor],
        action_by_behavior_policy: Union[np.ndarray, torch.Tensor],
        pscore: Union[np.ndarray, torch.Tensor],
        action_by_evaluation_policy: Union[np.ndarray, torch.Tensor],
        estimated_rewards_by_reg_model: Union[np.ndarray, torch.Tensor],
        **kwargs,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Estimate rewards for each round.

        Parameters
        ----------
        reward: array-like or Tensor, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action_by_behavior_policy: array-like or Tensor, shape (n_rounds,)
            Continuous action values sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like or Tensor, shape (n_rounds,)
            Probability densities of the continuous action values sampled by a behavior policy
            (generalized propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_by_evaluation_policy: array-like or Tensor, shape (n_rounds,)
            Continuous action values given by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(x_t)`.

        estimated_rewards_by_reg_model: array-like or Tensor, shape (n_rounds,)
            Expected rewards given context and action estimated by a regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        Returns
        ----------
        estimated_rewards: array-like or Tensor, shape (n_rounds,)
            Rewards estimated by KernelizedDR for each round.

        """
        kernel_func = kernel_functions[self.kernel]
        u = action_by_evaluation_policy - action_by_behavior_policy
        u /= self.bandwidth
        estimated_rewards = (
            kernel_func(u) * (reward - estimated_rewards_by_reg_model) / pscore
        )
        estimated_rewards /= self.bandwidth
        estimated_rewards += estimated_rewards_by_reg_model
        return estimated_rewards

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action_by_behavior_policy: np.ndarray,
        pscore: np.ndarray,
        action_by_evaluation_policy: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Estimate policy value of an evaluation policy.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action_by_behavior_policy: array-like, shape (n_rounds,)
            Continuous action values sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like, shape (n_rounds,)
            Probability densities of the continuous action values sampled by a behavior policy
            (generalized propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_by_evaluation_policy: array-like, shape (n_rounds,)
            Continuous action values given by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds,)
            Expected rewards given context and action estimated by a regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        Returns
        ----------
        V_hat: float
            Estimated policy value (performance) of a given evaluation policy.

        """
        check_continuous_ope_inputs(
            reward=reward,
            action_by_behavior_policy=action_by_behavior_policy,
            pscore=pscore,
            action_by_evaluation_policy=action_by_evaluation_policy,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )

        return self._estimate_round_rewards(
            reward=reward,
            action_by_behavior_policy=action_by_behavior_policy,
            pscore=pscore,
            action_by_evaluation_policy=action_by_evaluation_policy,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        ).mean()

    def estimate_interval(
        self,
        reward: np.ndarray,
        action_by_behavior_policy: np.ndarray,
        pscore: np.ndarray,
        action_by_evaluation_policy: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action_by_behavior_policy: array-like, shape (n_rounds,)
            Continuous action values sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like, shape (n_rounds,)
            Probability densities of the continuous action values sampled by a behavior policy
            (generalized propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_by_evaluation_policy: array-like, shape (n_rounds,)
            Continuous action values given by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds,)
            Expected rewards given context and action estimated by a regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

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
        check_continuous_ope_inputs(
            reward=reward,
            action_by_behavior_policy=action_by_behavior_policy,
            pscore=pscore,
            action_by_evaluation_policy=action_by_evaluation_policy,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )

        estimated_round_rewards = self._estimate_round_rewards(
            reward=reward,
            action_by_behavior_policy=action_by_behavior_policy,
            pscore=pscore,
            action_by_evaluation_policy=action_by_evaluation_policy,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )

        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )
