model_name=Paloma

root_path_name=../../dataset/PEMS
data_path_name=PEMSD8.npz
model_id_name=PEMSD8
data_name=PEMS


seq_len=96
for pred_len in 12
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
    --enc_in 170 \
    --cycle 288 \
    --use_revin 0 \
    --d_model 512 \
    --dropout 0.1 \
    --train_epochs 30 \
    --batch_size 32 \
    --patience 5 \
    --gpu 1 \
    --learning_rate 0.003 \
    --lossfun_alpha 0.5 \
    --loss_mode 'L1' \
    --lradj 'type3' \
    --itr 1
done

for pred_len in 24
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
    --enc_in 170 \
    --cycle 288 \
    --use_revin 0 \
    --d_model 512 \
    --dropout 0.0 \
    --train_epochs 30 \
    --batch_size 64 \
    --patience 5 \
    --gpu 1 \
    --learning_rate 0.004 \
    --lossfun_alpha 0.5 \
    --loss_mode 'L1' \
    --lradj 'type3' \
    --itr 1
done


for pred_len in 48 96
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
    --enc_in 170 \
    --cycle 288 \
    --use_revin 0 \
    --d_model 512 \
    --dropout 0.0 \
    --train_epochs 30 \
    --batch_size 32 \
    --patience 5 \
    --gpu 1 \
    --learning_rate 0.003 \
    --lossfun_alpha 0.5 \
    --loss_mode 'L1' \
    --lradj 'type3' \
    --itr 1
done