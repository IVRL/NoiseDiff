# ----------------- Diffusion Training ----------------------

# Train the diffusion model
python train_diffusion.py --use_tb_logger --save_epoch_freq 10 --generation_result noise --name train_diffusion --net_name NoiseDiffNet --beta_schedule sigmoid2 --positional_encoding --trainset SonyTrainDataset --dim 48 --crop_size 512 --with_camera_settings --batch_size 4 --max_iter 10000 --save_folder ./logs/noise_synthesis_newstart/weights

# ----------------- Diffusion Testing ----------------------

# Test the diffusion model for noise image generation
# You can generate noise data for different camera setting by changing --iso and --ratio
python test_diffusion.py --name ISO800Ratio250 --resume pretrained_ckpts/DiffusionNet_ckpt.pth --generation_result noise --testset NoiseImageGenerationDataset --save_npy --random_seed 0  --beta_schedule sigmoid2 --batch_size 4 --net_name NoiseDiffNet --positional_encoding --dim 48 --crop_size 512 --with_camera_settings --save_folder ./logs/generated_data/noise_imgs_SID_DDPM --iso 800 --ratio 250



# ----------------- Denoising Training ----------------------

# Train the denoising network with synthetic data
python train_denoising.py  --use_tb_logger --loss_l1  --save_epoch_freq 50 --crop_size 256 --sub_darkshading --use_sna --name train_denoising_noisediffstar --trainset SyntheticNoisDiffDenoisingDataset --batch_size 4 --max_iter 500 --save_folder ./logs/denoising_newstart/weights


# ----------------- Denoising Testing ----------------------

# Test the denoising network with darkshading correction on the SID testset
python test_denoising.py --resume pretrained_ckpts/NoiseDiffStar_ckpt.pth --correct_darkshading --correct_illum --ratio 100 --visualize_img --save_folder output/denoising/test_darkdiffusionstar_SID --test_dataset SID

# Test the denoising network with darkshading correction on the ELD testset
python test_denoising.py --resume pretrained_ckpts/NoiseDiffStar_ckpt.pth --correct_darkshading --correct_illum --ratio 100 --visualize_img --save_folder output/denoising/test_darkdiffusionstar_ELD --test_dataset ELD

# Test the denoising network without darkshading correction on the SID testset
python test_denoising.py --resume pretrained_ckpts/NoiseDiff_ckpt.pth --correct_illum --ratio 100 --visualize_img --save_folder output/denoising/test_darkdiffusion_SID  --test_dataset SID

# Test the denoising network without darkshading correction on the ELD testset
python test_denoising.py --resume pretrained_ckpts/NoiseDiff_ckpt.pth --correct_illum --ratio 100 --visualize_img --save_folder output/denoising/test_darkdiffusion_ELD --test_dataset ELD




# python train_denoising.py  --use_tb_logger --loss_l1  --save_epoch_freq 50 --crop_size 256 --name train_denoising_real --trainset RealSonyDenoisingDataset --batch_size 4 --max_iter 500 --save_folder ./logs/denoising_newstart/weights

# python train_denoising.py  --use_tb_logger --loss_l1  --save_epoch_freq 50 --crop_size 256 --name train_denoising_poissongaussian --trainset PossionGaussianDenoisingDataset --batch_size 4 --max_iter 500 --save_folder ./logs/denoising_newstart/weights

