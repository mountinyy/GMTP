import hydra
import torch

from src.inference import main as inference
from src.utils.base import set_seed


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf):
    set_seed(conf.common.seed)
    inference(conf)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
