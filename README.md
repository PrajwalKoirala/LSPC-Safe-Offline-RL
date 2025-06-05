# LSPC - Safe Offline RL [ICLR 2025]

**Paper: Latent Safety-Constrained Policy Approach for Safe Offline Reinforcement Learning**

**Arxiv:** [arxiv.org/abs/2412.08794](https://arxiv.org/abs/2412.08794)

**Abstract:** *In safe offline reinforcement learning, the objective is to develop a policy that maximizes cumulative rewards while strictly adhering to safety constraints, utilizing only offline data. Traditional methods often face difficulties in balancing these constraints, leading to either diminished performance or increased safety risks. We address these issues with a novel approach that begins by learning a conservatively safe policy through the use of Conditional Variational Autoencoders, which model the latent safety constraints. Subsequently, we frame this as a Constrained Reward-Return Maximization problem, wherein the policy aims to optimize rewards while complying with the inferred latent safety constraints. This is achieved by training an encoder with a reward-Advantage Weighted Regression objective within the latent constraint space. Our methodology is supported by theoretical analysis, including bounds on policy performance and sample complexity. Extensive empirical evaluation on benchmark datasets, including challenging autonomous driving scenarios, demonstrates that our approach not only maintains safety compliance but also excels in cumulative reward optimization, surpassing existing methods. Additional visualizations provide further insights into the effectiveness and underlying mechanisms of our approach.*


![LSPC Intro Diagram](https://github.com/PrajwalKoirala/LSPC-Safe-Offline-RL/blob/main/media/LSPC_Intro_Diagram.png?raw=true)

## Installation

Install [OSRL](https://github.com/liuzuxin/OSRL) package which provides all the datasets, environments, and utlities.
```bash
git clone https://github.com/liuzuxin/OSRL.git
cd osrl
pip install -e .
```
Next, clone this repository:
```bash
git https://github.com/PrajwalKoirala/LSPC-Safe-Offline-RL.git
cd LSPC-Safe-Offline-RL
```

## Usage
Run the training script using:
```bash
python train_lspc_iql.py --task OfflineHalfCheetahVelocityGymnasium-v1 --project [PROJECT_NAME] --logdir [LOG_DIR] 
```

Both LSPC-O and LSPC-S are trained with the same run. The distinction lies only in how inference is handled at evaluation time. 

LSPC-O evaluations results are logged as `eval/__` and `normalized_eval/__`.

LSPC-S evaluations results are logged as `eval_star/__` and `normalized_eval_star/__`.

Default hyperparameters set in `lspc_iql_configs.py` should work well for most task. See the [paper](https://arxiv.org/pdf/2412.08794) for discussion and ablation on hyperparameter and design choices.

## Citation
If you find this code useful, please consider citing:
```bibtex
@article{koirala2024latent,
  title={Latent Safety-Constrained Policy Approach for Safe Offline Reinforcement Learning},
  author={Koirala, Prajwal and Jiang, Zhanhong and Sarkar, Soumik and Fleming, Cody},
  journal={arXiv preprint arXiv:2412.08794},
  year={2024}
}
```
## Acknowledgements
Datasets: [DSRL](https://github.com/liuzuxin/DSRL)

Implementation style inspired by: [OSRL](https://github.com/liuzuxin/OSRL), [CORL](https://github.com/tinkoff-ai/CORL)

