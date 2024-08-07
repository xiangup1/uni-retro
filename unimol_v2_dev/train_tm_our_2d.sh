[ -z "${MASTER_PORT}" ] && MASTER_PORT=10087
[ -z "${MASTER_IP}" ] && MASTER_IP=127.0.0.1
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1
[ -z "${OMPI_COMM_WORLD_RANK}" ] && OMPI_COMM_WORLD_RANK=0

[ -z "${lr}" ] && lr=2e-4
[ -z "${end_lr}" ] && end_lr=1e-9
[ -z "${warmup_steps}" ] && warmup_steps=60000
[ -z "${total_steps}" ] && total_steps=1000000
[ -z "${layers}" ] && layers=12
[ -z "${hidden_size}" ] && hidden_size=768
[ -z "${ffn_size}" ] && ffn_size=768
[ -z "${num_head}" ] && num_head=32
[ -z "${batch_size}" ] && batch_size=128
[ -z "${update_freq}" ] && update_freq=1
[ -z "${seed}" ] && seed=1
[ -z "${clip_norm}" ] && clip_norm=5
[ -z "${data_path}" ] && data_path='./datasets/'
[ -z "${save_path}" ] && save_path='./logs/'
[ -z "${dropout}" ] && dropout=0.0
[ -z "${act_dropout}" ] && act_dropout=0.1
[ -z "${attn_dropout}" ] && attn_dropout=0.1
[ -z "${weight_decay}" ] && weight_decay=0.0
[ -z "${sandwich_ln}" ] && sandwich_ln="false"
[ -z "${droppath_prob}" ] && droppath_prob=0.1
[ -z "${noise_scale}" ] && noise_scale=0.2
[ -z "${mode_prob}" ] && mode_prob="0,1.0,0.0"

[ -z "${no_2d}" ] && no_2d="false"
[ -z "${num_3d_bias_kernel}" ] && num_3d_bias_kernel=128

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

save_dir=$save_dir"-lr"$lr"-bs"$batch_size


echo -e "\n\n"
echo "==================================ACTION ARGS==========================================="
if ( $sandwich_ln == "true")
then
  action_args="--sandwich-ln "
else
  action_args=""
fi
echo "action_args: ${action_args}"

if ( $add_3d == "true")
then
  add_3d_args="--add-3d"
else
  add_3d_args=""
fi
add_3d_args=""
echo "add_3d_args: ${add_3d_args}"

if ( $no_2d == "true")
then
  no_2d_args="--no-2d"
else
  no_2d_args=""
fi
echo "no_2d_args: ${no_2d_args}"

echo "========================================================================================"

mkdir -p $save_dir

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port $MASTER_PORT --nnodes=$OMPI_COMM_WORLD_SIZE --node_rank=$OMPI_COMM_WORLD_RANK --master_addr=$MASTER_IP \
      $(which unicore-train) $data_path --user-dir ./unimol --train-subset train --valid-subset valid_all,valid_our \
      --num-workers 8 --ddp-backend=no_c10d \
      --task Transformer-M --loss Transformer-M --arch transformer_m_base \
      --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --tensorboard-logdir $save_dir/tsb \
      --log-interval 100 --log-format simple \
      --save-interval-updates 5000 --validate-interval-updates 5000 --keep-interval-updates 200 --no-epoch-checkpoints  \
      --save-dir $save_dir \
      --batch-size $batch_size \
      --data-buffer-size 32 --fixed-validation-seed 11 --batch-size-valid 256 \
	    --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 $action_args --clip-norm $clip_norm \
      --lr $lr --end-learning-rate $end_lr --lr-scheduler polynomial_decay --power 1 \
      --warmup-updates $warmup_steps --total-num-update $total_steps --max-update $total_steps --update-freq $update_freq \
      --encoder-layers $layers --encoder-attention-heads $num_head $add_3d_args $no_2d_args --num-3d-bias-kernel $num_3d_bias_kernel \
      --encoder-embed-dim $hidden_size --encoder-ffn-embed-dim $ffn_size --droppath-prob $droppath_prob \
      --attention-dropout $attn_dropout --act-dropout $act_dropout --dropout $dropout --weight-decay $weight_decay \
      --noise-scale $noise_scale --mode-prob $mode_prob