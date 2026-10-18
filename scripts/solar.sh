model_name=Paloma

root_path_name=../../dataset/Solar
data_path_name=solar_AL.txt
model_id_name=solar
data_name=Solar


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
    --enc_in 137 \
    --cycle 144 \
    --d_model 128 \
    --dropout 0.5 \
    --use_revin 0 \
    --train_epochs 30 \
    --batch_size 64 \
    --patience 5 \
    --gpu 1 \
    --learning_rate 0.003 \
    --lossfun_alpha 0.5 \
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
    --enc_in 137 \
    --cycle 144 \
    --d_model 64 \
    --dropout 0.4 \
    --use_revin 0 \
    --train_epochs 30 \
    --batch_size 64 \
    --patience 5 \
    --gpu 1 \
    --learning_rate 0.002 \
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
    --enc_in 137 \
    --cycle 144 \
    --d_model 128 \
    --dropout 0.5 \
    --use_revin 0 \
    --train_epochs 30 \
    --batch_size 64 \
    --patience 5 \
    --gpu 1 \
    --learning_rate 0.007 \
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
    --enc_in 137 \
    --cycle 144 \
    --d_model 128 \
    --dropout 0.5 \
    --use_revin 0 \
    --train_epochs 30 \
    --batch_size 64 \
    --patience 5 \
    --gpu 1 \
    --learning_rate 0.006 \
    --lossfun_alpha 0.5 \
    --loss_mode 'L1' \
    --itr 1
done

