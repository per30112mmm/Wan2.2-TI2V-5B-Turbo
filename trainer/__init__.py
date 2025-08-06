from .diffusion import Trainer as DiffusionTrainer
from .gan import Trainer as GANTrainer
from .ode import Trainer as ODETrainer
from .distillation import Trainer as ScoreDistillationTrainer
from .wan22_distillation import Trainer as Wan22ScoreDistillationTrainer
__all__ = [
    "DiffusionTrainer",
    "GANTrainer",
    "ODETrainer",
    "ScoreDistillationTrainer",
    "Wan22ScoreDistillationTrainer"
]
