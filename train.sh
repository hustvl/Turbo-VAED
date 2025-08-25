export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_LEVEL=NVL

EXP_NAME=Turbo_VAED_Hunyuan

torchrun \
    --nnodes=1 --nproc_per_node=1 \
    --master_addr=localhost \
    --master_port=47869 \
    train_vae/train_vae.py \
    --exp_name ${EXP_NAME} \
    --model_config configs/Turbo-VAED-Hunyuan.json \
    --batch_size 1 \
    --accumulation_steps 8 \
    --epochs 100 \
    --lr 1e-4 \
    --disc_lr 1e-4 \
    --betas_1 0.9 \
    --betas_2 0.95 \
    --weight_decay 1e-4 \
    --eps 1e-15 \
    --video_path "" \
    --eval_video_path "" \
    --num_frames 17 \
    --resolution 256 \
    --sample_rate 1 \
    --find_unused_parameters \
    --mix_precision fp32 \
    --disc_start 9999999 \
    --disc_weight 0.05 \
    --kl_weight 1e-7 \
    --perceptual_weight 1.0 \
    --distillation_weight 1.0 \
    --feature_indices 0 1 \
    --log_steps 500 \
    --save_ckpt_step 5000 \
    --eval_steps 5000 \
    --eval_num_frames 17 \
    --eval_resolution 256 \
    --eval_batch_size 1 \
    --eval_subset_size 490 \
    --eval_lpips \
    --ema \
    --ema_decay 0.999 \
    --freeze_encoder \
    --teacher_model_name "Hunyuan" \
    --teacher_pretrained_model_name_or_path "" 

