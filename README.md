# **Adjoint Diff**
## The official Implementation of *High-dimensional Hyperparameter Optimization via Adjoint Differentiation* (accepted by IEEE Transactions on Artificial Intelligence). [IEEE Xplore link](https://ieeexplore.ieee.org/document/10880096/authors#authors)

+ Dependencies

```
pip install -r requirements.txt
```

+ Design loss function for imbalance data

```
CUDA_VISIBLE_DEVICES=1 python loss_func_design/main.py --config configs/cifar100/dyly_no_init/adjoint_diff_momentum.yaml

```
+ Selecting samples from noisy labels

```
CUDA_VISIBLE_DEVICES=1 python noisy_sample_reweight/main.py --config configs/cifar_noisy/adjoint_diff_momentum.yaml

```

## Citation
If you find this work useful for your research, please consider citing:

```bibtex
@article{dou2025high,
  title={High-dimensional Hyperparameter Optimization via Adjoint Differentiation},
  author={Dou, Hongkun and Li, Hongjue and Du, Jinyang and Fang, Leyuan and Gao, Qing and Deng, Yue and Yao, Wen},
  journal={IEEE Transactions on Artificial Intelligence},
  year={2025},
  publisher={IEEE}
}
