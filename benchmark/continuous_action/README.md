# Evaluating Off-Policy Learning for Continuous Actions

## Description

Here, we use synthetic bandit datasets to evaluate several off-policy learning methods. Specifically, we evaluate the performances of the deep-learning based off-policy learners using the ground-truth policy value calculable with synthetic data.

In the following, we evaluate the performances of

- Deterministic Policy Gradient (DPG)
- Importance Sampling Policy Gradient (ISPG)
- Doubly Robust DPG ('kernel')
- Doubly Robust DPG ('deterministic')

these off-policy policy gradient methods are summarized in [[Kallus and Uehara 2020](https://papers.nips.cc/paper/2020/hash/75df63609809c7a2052fdffe5c00a84e-Abstract.html)] and implemented in [zr-obp/obp/policy/offline_continuous.py](https://github.com/usaito/zr-obp/blob/continuous/obp/dataset/synthetic_continuous.py).

## Requirements

- hydra-core==1.0.6
- matplotlib==3.4.2
- numpy==1.20.3
- pandas==1.2.4
- python==3.9.5
- pyyaml==5.4.1
- seaborn==0.11.1
- scikit-learn==0.24.2
- scipy==1.6.3
- torch==1.8.1
- tqdm==4.60.0
- zr-obp==0.4.0

## Files

- [`main.py`](./main.py) implements the experimental workflows to evaluate the above off-policy learning methods using synthetic bandit feedback data.
- [`./conf/`](./conf/) specify experimental settings such as default hyperparameters to define deep-learning based policy and q-function estimator.

## Scripts

The experimental workflow is based on [Hydra](https://github.com/facebookresearch/hydra).
Below, we explain only important experimental configurations.


```bash
python main.py\
    setting.n_runs=$n_runs\
    setting.n_rounds_train=$n_rounds_train\
    setting.reward_function=$reward_function\
    setting.experiment=$experiment\
    setting.n_jobs=$n_jobs
```

- `$n_runs` specifies the number of simulation runs in the experiment. In the final project, `n_runs=500`.
- `$n_rounds_train` specifies the number of samples in the training set of synthetic bandit data. In the final project, `n_runs=1000`.
- `$reward_function` specifies the true reward model, and must be either `linear` or `quadratic`. In the final project, both cases were tested.
- `$experiment` specifies the experimental group, and must be one of `policy_optimizer`, `policy_architecture`, `bandwidth`, `q_func_optimizer`, `q_func_architecture`, and `cross-fitting`.
- `$n_jobs` is the maximum number of concurrently running jobs.

Other experimental configurations that can be modified from the command line include `n_rounds_test`, `dim_context`, `behavior_policy_function`, `action_noise`, `reward_noise`, `min_action`, and `max_action`. Please see [`./conf/`](./conf/) to confirm the default experimental configurations that are to be used when they are not overridden.

It is possible to run multiple experimental settings easily by using the `--multirun` option implemented in Hydra.
For example, the following script sweeps over all simulations including the six experimental groups ('policy_architecture', 'policy_optimizer', 'bandwidth', 'q_func_architecture', 'q_func_optimizer', 'cross_fitting') and two reward functions.

```bash
python main.py\
    setting.n_rounds_train=1000\
    setting.reward_function=quadratic,linear\
    setting.experiment=policy_architecture,policy_optimizer,bandwidth,q_func_architecture,q_func_optimizer,cross_fitting\
    --multirun
```
