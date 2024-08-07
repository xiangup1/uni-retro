### gpu configs
n_gpu=8
MASTER_PORT=$1

###
data_path='./examples/mol_conformers'
exp_name='conf_gen'

###
lr=1e-4
wd=1e-4
batch_size=32
warmup_steps=10000
max_steps=1000000
update_freq=1
recycling=4
seed=1
global_batch_size=`expr $batch_size \* $n_gpu \* $update_freq`
save_dir="./weights/${exp_name}_recycling_${recycling}_dist_loss_${distance_loss}_coord_loss_${coord_loss}_lr_${lr}_bs_${global_batch_size}"

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
nohup python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path --user-dir ./unimol --train-subset train --valid-subset valid \
       --num-workers 8 --ddp-backend=c10d \
       --task mol_confG --loss mol_confG --arch mol_confG  \
       --optimizer adam --adam-betas "(0.9, 0.99)" --adam-eps 1e-6 --clip-norm 1.0 --weight-decay $wd \
       --lr-scheduler polynomial_decay --lr $lr --warmup-updates $warmup_steps --total-num-update $max_steps \
       --update-freq $update_freq --seed $seed --batch-size $batch_size \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --tensorboard-logdir $save_dir/tsb \
       --max-update $max_steps --log-interval 1000 --log-format simple \
       --save-interval-updates 10000 --validate-interval-updates 1000 --keep-interval-updates 10 --no-epoch-checkpoints  \
       --recycling $recycling --distance-loss $distance_loss --coord-loss $coord_loss \
       --save-dir $save_dir \
       --find-unused-parameters --all-gather-list-size 102400 \
> ${save_dir}.log &
