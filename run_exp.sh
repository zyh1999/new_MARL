#!/bin/bash
trap 'onCtrlC' INT

function onCtrlC () {
  echo 'Ctrl+C is captured'
  for pid in $(jobs -p); do
    kill -9 $pid
  done

  kill -HUP $( ps -A -ostat,ppid | grep -e '^[Zz]' | awk '{print $2}')
  exit 1
}

configs=$1  # qmix
env=$2     # sc2
maps=$3    # MMM2,3s5z_vs_3s6z
args=$4    # use_cuda=True
threads=$5 # 2
gpus=$6    # 0,1
times=$7   # 5

configs=(${configs//,/ })
maps=(${maps//,/ })
gpus=(${gpus//,/ })
args=(${args//,/ })

if [ ! $configs ] || [ ! $env ] || [ ! $maps ]; then
    echo "Please enter the correct command."
    echo "bash run_fifo.sh config_name_list env_name map_name_list arg_list experinments_threads_num gpu_list experinments_num"
    exit 1
fi

if [ ! $threads ]; then
  threads=1
fi

if [ ! $gpus ]; then
  gpus=(0)
fi

if [ ! $times ]; then
  times=6
fi

echo "CONFIG LIST:" ${configs[@]}
echo "ENV:" env
echo "MAP LIST:" ${maps[@]}
echo "ARGS:"  ${args[@]}
echo "THREADS:" $threads
echo "GPU LIST:" ${gpus[@]}
echo "TIMES:" $times


# fifo
# https://www.cnblogs.com/maxgongzuo/p/6414376.html
FIFO_FILE=$(mktemp)
rm $FIFO_FILE
mkfifo $FIFO_FILE
trap "rm $FIFO_FILE" 3
trap "rm $FIFO_FILE" 15

exec 6<>$FIFO_FILE

for ((idx=0;idx<threads;idx++)); do
    echo
done >&6


# run parallel
count=0
for config in "${configs[@]}"; do
  for map in "${maps[@]}"; do
      for((i=0;i<times;i++)); do
          read -u6
          gpu=${gpus[$(($count % ${#gpus[@]}))]}
          {
            CUDA_VISIBLE_DEVICES="$gpu" python src/main.py --config="$config" --env-config="$env" with env_args.map_name="$map" "${args[@]}"
            echo >&6
          } &
          count=$(($count + 1))
          sleep $((RANDOM % 60 + 60))
      done
  done
done
wait

exec 6>&-   # 关闭fd6
rm $FIFO_FILE
