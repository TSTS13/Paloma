model_name=Paloma

root_path_name=../../dataset/ETT-small
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2


seq_len=96
for pred_len in 96
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
    --d_model 128 \
    --dropout 0.5 \
    --train_epochs 30 \
    --batch_size 32 \
    --patience 5 \
    --learning_rate 0.001 \
    --lossfun_alpha 0.3 \
    --loss_mode 'L1' \
    --itr 1
done


for pred_len in 192
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
    --d_model 128 \
    --dropout 0.5 \
    --train_epochs 30 \
    --batch_size 8 \
    --patience 5 \
    --learning_rate 0.001 \
    --lossfun_alpha 0.8 \
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
    --d_model 128 \
    --dropout 0.5 \
    --train_epochs 30 \
    --batch_size 4 \
    --patience 5 \
    --learning_rate 0.001 \
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
    --dropout 0.4 \
    --train_epochs 30 \
    --batch_size 256 \
    --patience 5 \
    --learning_rate 0.01 \
    --lossfun_alpha 0.4 \
    --loss_mode 'L1' \
    --itr 1
done