from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

from pyrallis import field


@dataclass
class BCTrainConfig:
    # wandb params
    project: str = "LSPC-Metadrive"
    group: str = None
    name: Optional[str] = None
    prefix: Optional[str] = "IQL"
    suffix: Optional[str] = ""
    logdir: Optional[str] = "LSPC-Metadrive"
    verbose: bool = True
    # Dataset params (from DSRL)
    outliers_percent: float = None
    noise_scale: float = None
    inpaint_ranges: Tuple[Tuple[float, float, float, float], ...] = None
    epsilon: float = None
    density: float = 1.0
    max_action: float = 1.0
    num_workers: int = 8
    # Training params
    task: str = "OfflineMetadrive-easysparse-v0"
    seed: int = 0
    cost_limit: int = 10
    reward_scale: float = 1.0
    cost_scale: float = 1.0
    episode_len: int = 300
    device: str = "cuda"
    eval_episodes: int = 10 # Number of episodes to eval
    eval_every: int = 20_000  # Eval every n steps
    max_timesteps: int = int(1e6)  # Max timesteps to update
    # IQL params
    batch_size: int = 1024  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    tau: float = 0.005  # Target network update rate
    beta_reward: float = 2.0  # Inverse temperature. Small inv temp -> BC, big beta -> maximizing Q
    beta_cost: float = 2.0  # Inverse temperature. Small inv temp -> BC, big beta -> minimizing C
    iql_tau: float = 0.7  # Coefficient for asymmetric loss
    vf_lr: float = 3e-4  # V function learning rate
    qf_lr: float = 3e-4  # Critic learning rate
    actor_lr: float = 3e-4  # Actor learning rate
    exp_adv_max_cost: float = 200.0  # Max advantage for exp weight (cost)
    exp_adv_max_reward: float = 200.0  # Max advantage for exp weight (reward)
    # Safety params
    latent_dim: int = 32  # Latent dimension for VAE Safety Encodings
    vae_beta: float = 0.5  # VAE KL Divergence Coefficient
    vae_clamper: float = 0.25  # Clamper for VAE Safety Encodings
    max_latent_action: float = 0.25  # Search space for latent safety encodings (after tanh squashing)
    log_std_min: float = -10.0  # Min log std for policy safety encoder (before tanh squashing)
    log_std_max: float = 2.0    # Max log std for policy safety encoder (before tanh squashing)
    safe_qc_vc_threshold: float = 999.0  # Threshold for safe Q and V value; 0.02 for Metadrive, +999.0 for others


@dataclass
class BCCarCircleConfig(BCTrainConfig):
    vae_clamper: float = 0.60
    max_latent_action: float = 0.60
    pass


@dataclass
class BCAntRunConfig(BCTrainConfig):
    # training params
    task: str = "OfflineAntRun-v0"
    episode_len: int = 200


@dataclass
class BCDroneRunConfig(BCTrainConfig):
    # training params
    task: str = "OfflineDroneRun-v0"
    episode_len: int = 200
    vae_clamper: float = 0.60
    max_latent_action: float = 0.60


@dataclass
class BCDroneCircleConfig(BCTrainConfig):
    # training params
    task: str = "OfflineDroneCircle-v0"
    episode_len: int = 300
    vae_clamper: float = 0.60
    max_latent_action: float = 0.60


@dataclass
class BCCarRunConfig(BCTrainConfig):
    # training params
    task: str = "OfflineCarRun-v0"
    episode_len: int = 200
    vae_clamper: float = 0.60
    max_latent_action: float = 0.60


@dataclass
class BCAntCircleConfig(BCTrainConfig):
    # training params
    task: str = "OfflineAntCircle-v0"
    episode_len: int = 500
    vae_clamper: float = 0.60
    max_latent_action: float = 0.60


@dataclass
class BCBallRunConfig(BCTrainConfig):
    # training params
    task: str = "OfflineBallRun-v0"
    episode_len: int = 100
    vae_clamper: float = 0.60
    max_latent_action: float = 0.60


@dataclass
class BCBallCircleConfig(BCTrainConfig):
    # training params
    task: str = "OfflineBallCircle-v0"
    episode_len: int = 200
    vae_clamper: float = 0.60
    max_latent_action: float = 0.60


@dataclass
class BCCarButton1Config(BCTrainConfig):
    # training params
    task: str = "OfflineCarButton1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCCarButton2Config(BCTrainConfig):
    # training params
    task: str = "OfflineCarButton2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCCarCircle1Config(BCTrainConfig):
    # training params
    task: str = "OfflineCarCircle1Gymnasium-v0"
    episode_len: int = 500
    vae_clamper: float = 0.25
    max_latent_action: float = 0.25


@dataclass
class BCCarCircle2Config(BCTrainConfig):
    # training params
    task: str = "OfflineCarCircle2Gymnasium-v0"
    episode_len: int = 500


@dataclass
class BCCarGoal1Config(BCTrainConfig):
    # training params
    task: str = "OfflineCarGoal1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCCarGoal2Config(BCTrainConfig):
    # training params
    task: str = "OfflineCarGoal2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCCarPush1Config(BCTrainConfig):
    # training params
    task: str = "OfflineCarPush1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCCarPush2Config(BCTrainConfig):
    # training params
    task: str = "OfflineCarPush2Gymnasium-v0"
    episode_len: int = 1000
    buffer_size: int = 5_000_000


@dataclass
class BCPointButton1Config(BCTrainConfig):
    # training params
    task: str = "OfflinePointButton1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCPointButton2Config(BCTrainConfig):
    # training params
    task: str = "OfflinePointButton2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCPointCircle1Config(BCTrainConfig):
    # training params
    task: str = "OfflinePointCircle1Gymnasium-v0"
    episode_len: int = 500


@dataclass
class BCPointCircle2Config(BCTrainConfig):
    # training params
    task: str = "OfflinePointCircle2Gymnasium-v0"
    episode_len: int = 500


@dataclass
class BCPointGoal1Config(BCTrainConfig):
    # training params
    task: str = "OfflinePointGoal1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCPointGoal2Config(BCTrainConfig):
    # training params
    task: str = "OfflinePointGoal2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCPointPush1Config(BCTrainConfig):
    # training params
    task: str = "OfflinePointPush1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCPointPush2Config(BCTrainConfig):
    # training params
    task: str = "OfflinePointPush2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCAntVelocityConfig(BCTrainConfig):
    # training params
    task: str = "OfflineAntVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class BCHalfCheetahVelocityConfig(BCTrainConfig):
    # training params
    task: str = "OfflineHalfCheetahVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class BCHopperVelocityConfig(BCTrainConfig):
    # training params
    task: str = "OfflineHopperVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class BCSwimmerVelocityConfig(BCTrainConfig):
    # training params
    task: str = "OfflineSwimmerVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class BCWalker2dVelocityConfig(BCTrainConfig):
    # training params
    task: str = "OfflineWalker2dVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class BCEasySparseConfig(BCTrainConfig):
    # training params
    task: str = "OfflineMetadrive-easysparse-v0"
    episode_len: int = 1000
    safe_qc_vc_threshold: float = 0.02


@dataclass
class BCEasyMeanConfig(BCTrainConfig):
    # training params
    task: str = "OfflineMetadrive-easymean-v0"
    episode_len: int = 1000
    safe_qc_vc_threshold: float = 0.02


@dataclass
class BCEasyDenseConfig(BCTrainConfig):
    # training params
    task: str = "OfflineMetadrive-easydense-v0"
    episode_len: int = 1000
    safe_qc_vc_threshold: float = 0.02


@dataclass
class BCMediumSparseConfig(BCTrainConfig):
    # training params
    task: str = "OfflineMetadrive-mediumsparse-v0"
    episode_len: int = 1000
    safe_qc_vc_threshold: float = 0.02


@dataclass
class BCMediumMeanConfig(BCTrainConfig):
    # training params
    task: str = "OfflineMetadrive-mediummean-v0"
    episode_len: int = 1000
    safe_qc_vc_threshold: float = 0.02


@dataclass
class BCMediumDenseConfig(BCTrainConfig):
    # training params
    task: str = "OfflineMetadrive-mediumdense-v0"
    episode_len: int = 1000
    safe_qc_vc_threshold: float = 0.02


@dataclass
class BCHardSparseConfig(BCTrainConfig):
    # training params
    task: str = "OfflineMetadrive-hardsparse-v0"
    episode_len: int = 1000
    safe_qc_vc_threshold: float = 0.02


@dataclass
class BCHardMeanConfig(BCTrainConfig):
    # training params
    task: str = "OfflineMetadrive-hardmean-v0"
    episode_len: int = 1000
    safe_qc_vc_threshold: float = 0.02


@dataclass
class BCHardDenseConfig(BCTrainConfig):
    # training params
    task: str = "OfflineMetadrive-harddense-v0"
    episode_len: int = 1000
    safe_qc_vc_threshold: float = 0.02


BC_DEFAULT_CONFIG = {
    # bullet_safety_gym
    "OfflineCarCircle-v0": BCCarCircleConfig,
    "OfflineAntRun-v0": BCAntRunConfig,
    "OfflineDroneRun-v0": BCDroneRunConfig,
    "OfflineDroneCircle-v0": BCDroneCircleConfig,
    "OfflineCarRun-v0": BCCarRunConfig,
    "OfflineAntCircle-v0": BCAntCircleConfig,
    "OfflineBallCircle-v0": BCBallCircleConfig,
    "OfflineBallRun-v0": BCBallRunConfig,
    # safety_gymnasium: car
    "OfflineCarButton1Gymnasium-v0": BCCarButton1Config,
    "OfflineCarButton2Gymnasium-v0": BCCarButton2Config,
    "OfflineCarCircle1Gymnasium-v0": BCCarCircle1Config,
    "OfflineCarCircle2Gymnasium-v0": BCCarCircle2Config,
    "OfflineCarGoal1Gymnasium-v0": BCCarGoal1Config,
    "OfflineCarGoal2Gymnasium-v0": BCCarGoal2Config,
    "OfflineCarPush1Gymnasium-v0": BCCarPush1Config,
    "OfflineCarPush2Gymnasium-v0": BCCarPush2Config,
    # safety_gymnasium: point
    "OfflinePointButton1Gymnasium-v0": BCPointButton1Config,
    "OfflinePointButton2Gymnasium-v0": BCPointButton2Config,
    "OfflinePointCircle1Gymnasium-v0": BCPointCircle1Config,
    "OfflinePointCircle2Gymnasium-v0": BCPointCircle2Config,
    "OfflinePointGoal1Gymnasium-v0": BCPointGoal1Config,
    "OfflinePointGoal2Gymnasium-v0": BCPointGoal2Config,
    "OfflinePointPush1Gymnasium-v0": BCPointPush1Config,
    "OfflinePointPush2Gymnasium-v0": BCPointPush2Config,
    # safety_gymnasium: velocity
    "OfflineAntVelocityGymnasium-v1": BCAntVelocityConfig,
    "OfflineHalfCheetahVelocityGymnasium-v1": BCHalfCheetahVelocityConfig,
    "OfflineHopperVelocityGymnasium-v1": BCHopperVelocityConfig,
    "OfflineSwimmerVelocityGymnasium-v1": BCSwimmerVelocityConfig,
    "OfflineWalker2dVelocityGymnasium-v1": BCWalker2dVelocityConfig,
    # safe_metadrive
    "OfflineMetadrive-easysparse-v0": BCEasySparseConfig,
    "OfflineMetadrive-easymean-v0": BCEasyMeanConfig,
    "OfflineMetadrive-easydense-v0": BCEasyDenseConfig,
    "OfflineMetadrive-mediumsparse-v0": BCMediumSparseConfig,
    "OfflineMetadrive-mediummean-v0": BCMediumMeanConfig,
    "OfflineMetadrive-mediumdense-v0": BCMediumDenseConfig,
    "OfflineMetadrive-hardsparse-v0": BCHardSparseConfig,
    "OfflineMetadrive-hardmean-v0": BCHardMeanConfig,
    "OfflineMetadrive-harddense-v0": BCHardDenseConfig
}
