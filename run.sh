# fine-grained contrastive loss

CUDA_VISIBLE_DEVICES=0 nohup python train_ave_fgc.py --data MKG-W --num_epoch 1500 --hidden_dim 1024 --lr 5e-4 --dim 256 --max_txt_token 8 --num_head 4 --emb_dropout 0.9 --vis_dropout 0.4 --txt_dropout 0.1 --num_layer_dec 2 --mu 0.001 > log_MKG-W.txt &

CUDA_VISIBLE_DEVICES=0 nohup python train_ave_fgc.py --data DB15K --num_epoch 1500 --hidden_dim 1024 --lr 1e-3 --dim 256 --max_vis_token 8 --max_txt_token 4 --num_head 2 --emb_dropout 0.6 --vis_dropout 0.3 --txt_dropout 0.1 --num_layer_dec 1 --mu 0.01 > log_DB15K.txt &


# Ave: fine-grained representation + relation-role schema calibration
# + two-hop path calibration
# MKG-W
CUDA_VISIBLE_DEVICES=0 python train_ave.py \
  --data MKG-W --exp ave \
  --text_tokenizer bert --visual_tokenizer beit \
  --num_epoch 1500 --valid_epoch 50 --early_stop 0 \
  --seed 2024 --dim 256 --hidden_dim 1024 --num_head 4 \
  --num_layer_enc_ent 1 --num_layer_enc_rel 1 --num_layer_dec 2 \
  --dropout 0.01 --emb_dropout 0.9 --vis_dropout 0.4 --txt_dropout 0.1 \
  --max_vis_token 8 --max_txt_token 8 \
  --batch_size 2048 --eval_batch_size 256 \
  --lr 5e-4 --step_size 50 \
  --mu 0.001 \
  --lambda_role_ce 1.0 --lambda_role_reg 0.0001 \
  --role_direct_weight 0.5 --similar_roles 4 \
  --role_scales "0,0.25,0.5,0.75,1,1.5" \
  --min_rule_support 2 \
  --path_alphas "0,0.25,0.5,1,2,4,8" \
  --non_deterministic

# MKG-Y
# LLaMA-7B textual embeddings are loaded from tokens/textual_llama.pth.
# max_txt_token=12 and batch_size=512 are the 24 GiB GPU configuration.
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 CUDA_VISIBLE_DEVICES=0 python train_ave.py \
  --data MKG-Y --exp ave_llama_t12 \
  --text_tokenizer llama --visual_tokenizer beit \
  --num_epoch 1500 --valid_epoch 50 --early_stop 0 \
  --seed 2024 --dim 200 --hidden_dim 1024 --num_head 4 \
  --num_layer_enc_ent 1 --num_layer_enc_rel 1 --num_layer_dec 2 \
  --dropout 0.01 --emb_dropout 0.9 --vis_dropout 0.4 --txt_dropout 0.1 \
  --max_vis_token 6 --max_txt_token 12 \
  --batch_size 512 --eval_batch_size 64 \
  --lr 5e-4 --step_size 50 \
  --mu 0.001 \
  --lambda_role_ce 1.0 --lambda_role_reg 0.0001 \
  --role_direct_weight 0.5 --similar_roles 4 \
  --role_scales "0,0.25,0.5,0.75,1,1.5" \
  --min_rule_support 2 \
  --path_alphas "0,0.25,0.5,1,2,4,8" \
  --non_deterministic

# DB15K
CUDA_VISIBLE_DEVICES=0 python train_ave.py \
  --data DB15K --exp ave \
  --text_tokenizer bert --visual_tokenizer beit \
  --num_epoch 1500 --valid_epoch 50 --early_stop 0 \
  --seed 2024 --dim 256 --hidden_dim 1024 --num_head 2 \
  --num_layer_enc_ent 1 --num_layer_enc_rel 1 --num_layer_dec 1 \
  --dropout 0.01 --emb_dropout 0.6 --vis_dropout 0.3 --txt_dropout 0.1 \
  --max_vis_token 8 --max_txt_token 4 \
  --batch_size 2048 --eval_batch_size 256 \
  --lr 1e-3 --step_size 50 \
  --mu 0.01 \
  --lambda_role_ce 1.0 --lambda_role_reg 0.0001 \
  --role_direct_weight 0.5 --similar_roles 4 \
  --role_scales "0,0.25,0.5,0.75,1,1.5" \
  --min_rule_support 2 \
  --path_alphas "0,0.25,0.5,1,2,4,8" \
  --non_deterministic
