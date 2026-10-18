model_name=Paloma

root_path_name=../../dataset/electricity
data_path_name=electricity.csv
model_id_name=ECL
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
    --enc_in 321 \
    --cycle 168 \
    --d_model 512 \
    --dropout 0.0 \
    --train_epochs 30 \
    --batch_size 64 \
    --patience 5 \
    --gpu 1 \
    --learning_rate 0.005 \
    --lossfun_alpha 0.6 \
    --loss_mode 'L1L2' \
    --itr 1
done


for pred_len in 192 336 720
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
    --enc_in 321 \
    --cycle 168 \
    --d_model 512 \
    --dropout 0.1 \
    --train_epochs 30 \
    --batch_size 16 \
    --patience 5 \
    --gpu 1 \
    --learning_rate 0.003 \
    --lossfun_alpha 0.5 \
    --loss_mode 'L1' \
    --itr 1
done

