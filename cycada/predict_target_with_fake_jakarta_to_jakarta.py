from cycada_jitter import *
from data import CycleGAN_datamodule

num_workers = 8

user = ''

datamodule = CycleGAN_datamodule(
    num_workers=num_workers, 
    target_root_dir='/cluster/scratch/jehrat/ai4good',
    target_folder_name="splits_jakarta", 
    target_satellite_name='satellite',
    tst_batch_sz=64,
)

model = Cycada.load_from_checkpoint(
    '/cluster/scratch/{user}/logs/lightning_logs/cycada_fake_jakarta_to_jakarta/last.ckpt',
    datamodule=datamodule, 
    filter_size=32, 
    num_classes=1,
    user=user,
    seg_system_checkpoint_path='/cluster/scratch/{user}/logs/lightning_logs/UNet_splits_new_buildings_jitter/last.ckpt'
    )

trainer = pl.Trainer(
    gpus = -1,  
    precision = 16, 
    benchmark=True,
    num_sanity_val_steps = 1, 
    limit_train_batches=0.02,
    #limit_val_batches=0.01,
    check_val_every_n_epoch=10
)

trainer.predict(model, datamodule.test_dataloader())
