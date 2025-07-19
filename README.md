
To reproduce our unconditional generation results on the QM9 dataset, run mol_gen_eval_torch_latent.py directly, and modify the code for importing weights.

To reproduce our conditional generation results on QM9. Run mol_gen_eval_conditional_qm9_latent.py with the following parameters:
datamodule=edm_qm9
model=qm9_mol_gen_ddpm_torch_latent
logger=csv
trainer.accelerator=gpu
trainer.devices=[0]
datamodule.dataloader_cfg.num_workers=1
model.diffusion_cfg.sample_during_training=false
generator_model_filepath="/root/check_val_GCLDM_from_git/ckpt/alpha/best_loss_mini_batch.pth"
classifier_model_dir="/root/check_val_GCLDM_from_git/ckpt/Property_Classifiers/exp_class_alpha"
property=alpha
iterations=100
batch_size=100
