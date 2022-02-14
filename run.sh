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

config=$1  # qmix
env=$2     # sc2
maps=$3    # MMM2,3s5z_vs_3s6z
args=$4    # use_cuda=True
threads=$5 # 2
gpus=$6    # 0,1
times=$7   # 5

maps=(${maps//,/ })
gpus=(${gpus//,/ })
args=(${args//,/ })

if [ ! $config ] || [ ! $env ] || [ ! $maps ]; then
    echo "Please enter the correct command."
    echo "bash run.sh config_name env_name map_name_list arg_list experinments_threads_num gpu_list experinments_num"
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

echo "CONFIG:" $config
echo "ENV:" env
echo "MAP LIST:" ${maps[@]}
echo "ARGS:"  ${args[@]}
echo "THREADS:" $threads
echo "GPU LIST:" ${gpus[@]}
echo "TIMES:" $times


# run parallel
count=0
for map in "${maps[@]}"; do
    for((i=0;i<times;i++)); do
        gpu=${gpus[$(($count % ${#gpus[@]}))]}
        CUDA_VISIBLE_DEVICES="$gpu" python src/main.py --config="$config" --env-config="$env" with env_args.map_name="$map" "${args[@]}" &

        count=$(($count + 1))
        if [ $(($count % $threads)) -eq 0 ]; then
            wait
        fi
        # for random seeds
        sleep $((RANDOM % 60 + 60))
    done
done
wait