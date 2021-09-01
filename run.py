from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import hydra

import pytorch_lightning as pl

# from model.delg import LitDELG

@hydra.main(config_path='.', config_name='common')
def run(config):
    if config.model_name == 'delg':
        # model = LitDELG(config)
        pass

    if config.wandb_logger:
        from pytorch_lightning.loggers import WandbLogger
        logger = WandbLogger(
            entity='monet-kaggle',
            project=config.project,
            name=config.backbone_name,
            config=config,
        )
    else:
        logger = pl.loggers.TestTubeLogger(
            'output', name='google-retrieval'
        )
        logger.log_hyperparams(config)

    if type(config.gpus) == str:
        config.gpus = [int(config.gpus.replace("cuda:", ""))]

    callbacks = []
    if config.es_patience is not None:
        callbacks.append(EarlyStopping(
            monitor='val_loss',
            min_delta=0.00,
            patience=config.es_patience,
            verbose=False,
            mode='min'
        ))
    callbacks.append(ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        filename=f"{config.backbone_name}-"+"{epoch:02d}"+
        f"fold{config.fold_no}-"+"{val_loss:4.f}",
        monitor='val_loss',
        mode='min'
    ))

    trainer = pl.Trainer(
        callbacks=callbacks,
        precision=config.precision,
        gradient_clip_val=0.5,
        deterministic=True,
        val_check_interval=8000,
        gpus=3,
        max_epochs=config.max_epochs,
        weights_summary='top',
        logger=logger,
    )

    if config.lr_finder:
        lr_finder = trainer.tuner.lr_find(model, min_lr=3e-4, max_lr=1e-2, num_training=100)
        model.hparams.lr = lr_finder.suggestion()
        print(model.hparams.lr)
    else:
        trainer.fit(model)
        trainer.test()

if __name__ == '__main__':
    run()

