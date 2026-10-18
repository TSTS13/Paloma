model_name=Paloma

root_path_name=../../dataset/ETT-small
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1


seq_len=96
for pred_len in 96 192
do
  python -u run.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --cycle 24 \
    --d_model 256 \
    --dropout 0.5 \
    --train_epochs 30 \
    --batch_size 32 \
    --patience 5 \
    --gpu 1 \
    --learning_rate 0.0005 \
    --lossfun_alpha 0.5 \
    --loss_mode 'L1' \
    --itr 1
done


for pred_len in 336
do
  python -u run.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --cycle 24 \
    --d_model 256 \
    --dropout 0.5 \
    --train_epochs 30 \
    --batch_size 32 \
    --patience 5 \
    --gpu 1 \
    --learning_rate 0.0007 \
    --lossfun_alpha 0.5 \
    --loss_mode 'L1' \
    --itr 1
done


for pred_len in 720
do
  python -u run.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --cycle 24 \
    --d_model 256 \
    --dropout 0.5 \
    --train_epochs 30 \
    --batch_size 128 \
    --patience 5 \
    --gpu 1 \
    --learning_rate 0.0003 \
    --lossfun_alpha 0.5 \
    --loss_mode 'L1' \
    --itr 1
done




