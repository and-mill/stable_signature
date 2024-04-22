python finetune_ldm_decoder.py --num_keys 10 --train_dir ../coco2017/train2017/ --val_dir ../coco2017/val2017/ --output_dir trained_decoders --ldm_config sd/stable-diffusion-2-1-base/v2-inference.yaml --ldm_ckpt sd/stable-diffusion-2-1-base/v2-1_512-ema-pruned.ckpt  --seed 42 --msg_decoder_path models/dec_48b_whit.torchscript.pt --steps 140