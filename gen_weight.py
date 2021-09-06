import hydra
import os
import torch

from model.cgd import LitCGD

@hydra.main(config_path='.', config_name='common')
def run(config):
    if config.model_name == 'cgd':
        pl_model = LitCGD.load_from_checkpoint(config.weight_path, config=config)
        
    os.makedirs(config.output_dir, exist_ok=True)

    model = pl_model.model
    torch.save(model.state_dict(), os.path.join(config.output_dir, os.path.join(config.weight_path.split("/")[-1])))
    print("saved!")


if __name__ == '__main__':
    run()