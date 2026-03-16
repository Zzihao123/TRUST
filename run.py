import random
import numpy as np
import torch

from main.run.Args import args
from main.run.train import run_train, run_test


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    set_seed(args.random_seed)
    if args.test:
        run_test(args)
    else:
        run_train(args)
