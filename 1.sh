python3 train.py --dataset lol_v1 --cropSize 400 --batchSize 6 --use_region_prior False
python3 train.py --dataset lol_v1 --cropSize 400 --batchSize 6 --use_region_prior True --prior_mode attn --prior_label_dir /home/zqh/code/sam2/notebooks/runs/automatic_mask_lolv1_final_packpkl_plus_labels_plus_npz/label_uint8
python3 train.py --dataset lol_v1 --cropSize 256 --batchSize 8 \
  --use_region_prior True --prior_mode attn --prior_label_dir /home/zqh/code/sam2/notebooks/runs/automatic_mask_lolv1_final_packpkl_plus_labels_plus_npz/label_uint8 \
  --max_regions 16 \
  --attn_alpha1_init -2.197225 --attn_alpha2_init -3.891 \
  --attn_mask_bias_scale1_init 1.0 --attn_mask_bias_scale2_init 0.3


# python3 train.py --dataset lolv2_real --cropSize 256 --batchSize 8 --use_region_prior True --prior_mode attn --prior_label_dir /home/zqh/code/sam2/notebooks/runs/automatic_mask_lolv2_real_pkl_labels_npz/label_unit8
