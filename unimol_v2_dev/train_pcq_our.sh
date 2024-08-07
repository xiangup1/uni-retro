[ -z "${MASTER_PORT}" ] && MASTER_PORT=10087
[ -z "${MASTER_IP}" ] && MASTER_IP=127.0.0.1
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)
[ -z "${update_freq}" ] && update_freq=1
[ -z "${total_step}" ] && total_step=1500000
[ -z "${warmup_step}" ] && warmup_step=150000
[ -z "${seed}" ] && seed=31
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1
[ -z "${OMPI_COMM_WORLD_RANK}" ] && OMPI_COMM_WORLD_RANK=0

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
echo "n_gpu per node" $n_gpu
echo "OMPI_COMM_WORLD_SIZE" $OMPI_COMM_WORLD_SIZE
echo "OMPI_COMM_WORLD_RANK" $OMPI_COMM_WORLD_RANK
echo "MASTER_IP" $MASTER_IP
echo "MASTER_PORT" $MASTER_PORT
echo "data" $1
echo "save_dir" $2
echo "warmup_step" $warmup_step
echo "total_step" $total_step
echo "update_freq" $update_freq
echo "seed" $seed
echo "data_folder:"

data_path=$1
save_dir=$2
lr=$3
batch_size=$4
num_recycle=$5
num_block=$6
coord_gen_prob=$7
src_noise=$8
tgt_noise=$9

save_dir=$save_dir"-lr"$lr"-bs"$batch_size"-nr"$num_recycle"-nb"$num_block"-cp"$coord_gen_prob"-sn"$src_noise"-tn"$tgt_noise-"nn"$OMPI_COMM_WORLD_SIZE

mkdir -p $save_dir

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port $MASTER_PORT --nnodes=$OMPI_COMM_WORLD_SIZE --node_rank=$OMPI_COMM_WORLD_RANK --master_addr=$MASTER_IP \
       $(which unicore-train) $data_path --user-dir ./unimol --train-subset train --valid-subset valid_all,valid_our \
       --num-workers 8 --ddp-backend=c10d \
       --task unimol_pcq --loss unimol_pcq --arch unimol_pcq_base  \
       --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.0 \
       --lr-scheduler polynomial_decay --lr $lr --warmup-updates $warmup_step --total-num-update $total_step \
       --update-freq $update_freq --seed $seed --batch-size $batch_size \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --tensorboard-logdir $save_dir/tsb \
       --max-update $total_step --log-interval 100 --log-format simple \
       --save-interval-updates 5000 --validate-interval-updates 5000 --keep-interval-updates 200 --no-epoch-checkpoints  \
       --save-dir $save_dir \
       --num-recycle $num_recycle --num-block $num_block --coord-gen-prob $coord_gen_prob \
       --best-checkpoint-metric pred_homo_lumo_loss  --src-noise $src_noise --tgt-noise $tgt_noise --data-buffer-size 32 --fixed-validation-seed 11 --batch-size-valid 256