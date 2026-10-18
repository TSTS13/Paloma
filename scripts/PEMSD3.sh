model_name=Paloma

root_path_name=../../dataset/PEMS
data_path_name=PEMSD3.npz
model_id_name=PEMSD3
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
    --enc_in 358 \
    --cycle 288 \
    --use_revin 0 \
    --d_model 512 \
    --dropout 0.1 \
    --train_epochs 30 \
    --batch_size 32 \
    --patience 5 \
    --gpu 1 \
    --learning_rate 0.005 \
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
    --enc_in 358 \
    --cycle 288 \
    --use_revin 0 \
    --d_model 512 \
    --dropout 0.1 \
    --train_epochs 30 \
    --batch_size 32 \
    --patience 5 \
    --gpu 1 \
    --learning_rate 0.006 \
    --lossfun_alpha 0.5 \
    --loss_mode 'L1' \
    --lradj 'type3' \
    --itr 1
done


for pred_len in 48
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
    --enc_in 358 \
    --cycle 288 \
    --use_revin 0 \
    --d_model 512 \
    --dropout 0.1 \
    --train_epochs 30 \
    --batch_size 8 \
    --patience 5 \
    --gpu 1 \
    --learning_rate 0.0009 \
    --lossfun_alpha 0.6 \
    --loss_mode 'L1L2' \
    --lradj 'type3' \
    --itr 1
done

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
    --enc_in 358 \
    --cycle 288 \
    --use_revin 0 \
    --d_model 512 \
    --dropout 0.1 \
    --train_epochs 30 \
    --batch_size 16 \
    --patience 5 \
    --gpu 1 \
    --learning_rate 0.002 \
    --lossfun_alpha 0.5 \
    --loss_mode 'L1L2' \
    --lradj 'type3' \
    --itr 1
done


