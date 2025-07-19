
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



For the unconditional generation on QM9, the training strategy we adopted is as follows: first, pre-train for 1,000 epochs with a batch size of 256, and then continue training for another 1,000 epochs with a batch size of 64.

For conditional generation, it is sufficient to train directly for 1,700 to 2,000 epochs under the condition of a batch size of 64.

We provide pre-trained weights at https://drive.google.com/file/d/1QF-9UXZhEcGelECOwWw6U9tbHMyJbmyp/view?usp=sharing
