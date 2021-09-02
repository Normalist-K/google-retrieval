from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(config_path='.', config_name='common')
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    my_app()