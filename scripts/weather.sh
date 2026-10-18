model_name=Paloma

root_path_name=../../dataset/weather
data_path_name=weather.csv
model_id_name=weather
data_name=custom


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
    --enc_in 21 \
    --cycle 144 \
    --d_model 128 \
    --dropout 0.3 \
    --use_revin 1 \
    --train_epochs 30 \
    --batch_size 32 \
    --patience 5 \
    --gpu 1 \
    --learning_rate 0.004 \
    --lossfun_alpha 0.5 \
    --loss_mode 'L1' \
    --itr 1
done


for pred_len in 192 336
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
    --enc_in 21 \
    --cycle 144 \
    --d_model 128 \
    --dropout 0.5 \
    --use_revin 0 \
    --train_epochs 30 \
    --batch_size 256 \
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
    --enc_in 21 \
    --cycle 144 \
    --d_model 128 \
    --dropout 0.5 \
    --use_revin 0 \
    --train_epochs 30 \
    --batch_size 512 \
    --patience 5 \
    --gpu 1 \
    --learning_rate 0.01 \
    --lossfun_alpha 0.5 \
    --loss_mode 'L1' \
    --itr 1
done

