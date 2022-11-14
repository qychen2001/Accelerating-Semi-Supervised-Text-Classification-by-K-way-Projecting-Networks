
```shell
python run_uda.py \
--task=yahoo \
--gpu_num=0 \
--train_max_seq_length=128 \
--model_name_or_path="./pretrained/bert_base" \
--per_gpu_train_batch_size=8 \
--per_gpu_eval_batch_size=128 \
--learning_rate=1e-5 \
--linear_learning_rate=1e-3 \
--do_distri_train \
--num_train_epochs=20 \
--output_dir="./output" \
--weight_decay=0.01 \
--tsa=log_schedule \
--early_stop=2 \
--tsa=exp_schedule \
  $@
```#   A c c e l e r a t i n g - S e m i - S u p e r v i s e d - T e x t - C l a s s i f i c a t i o n - b y - K - w a y - P r o j e c t i n g - N e t w o r k s  
 #   A c c e l e r a t i n g - S e m i - S u p e r v i s e d - T e x t - C l a s s i f i c a t i o n - b y - K - w a y - P r o j e c t i n g - N e t w o r k s  
 # Accelerating-Semi-Supervised-Text-Classification-by-K-way-Projecting-Networks
