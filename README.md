# **Adjoint Diff**
## The official Implementation of *High-dimensional Hyperparameter Optimization via Adjoint Differentiation*

+ Dependencies
+ Design loss function for imbalance data

```
CUDA_VISIBLE_DEVICES=1 python loss_func_design/main.py --config configs/cifar100/dyly_no_init/adjoint_diff_momentum.yaml

```
+ Selecting samples from noisy labels

```
CUDA_VISIBLE_DEVICES=1 python noisy_sample_reweight/main.py --config configs/cifar_noisy/adjoint_diff_momentum.yaml

```

