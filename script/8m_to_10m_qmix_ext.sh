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


threads=$1
gpus=$2
exc=$3

gpus=(${gpus//,/ })
exc=(${exc//,/ })

if [ ! $threads ]; then
  threads=1
fi

if [ ! $gpus ]; then
  gpus=(0)
fi

if [ ! $gpus ]; then
  exc=(0 1 2 3 4)
fi


echo "THREADS:" $threads
echo "GPU LIST:" ${gpus[@]}
echo "EXC LIST:" ${exc[@]}


commands=(
  "python src/main.py --config=qmix_ext --env-config=sc2 with t_max=4050000 seed=162570563 name=qmix_ext_8m_vs_9m env_args.map_name=10m_vs_11m checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/8m_vs_9m/qmix_ext/2021-09-12_19-12-38/models\""
  "python src/main.py --config=qmix_ext --env-config=sc2 with t_max=4050000 seed=708846337 name=qmix_ext_8m_vs_9m env_args.map_name=10m_vs_11m checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/8m_vs_9m/qmix_ext/2021-09-13_21-14-38/models\""
  "python src/main.py --config=qmix_ext --env-config=sc2 with t_max=4050000 seed=104240229 name=qmix_ext_8m_vs_9m env_args.map_name=10m_vs_11m checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/8m_vs_9m/qmix_ext/2021-09-14_10-20-05/models\""
  "python src/main.py --config=qmix_ext --env-config=sc2 with t_max=4050000 seed=727463145 name=qmix_ext_8m_vs_9m env_args.map_name=10m_vs_11m checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/8m_vs_9m/qmix_ext/2021-09-14_17-30-25/models\""
  "python src/main.py --config=qmix_ext --env-config=sc2 with t_max=4050000 seed=70127011 name=qmix_ext_8m_vs_9m env_args.map_name=10m_vs_11m checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/8m_vs_9m/qmix_ext/2021-09-16_20-23-28/models\""
)


# run parallel
count=0
for((i=0;i<${#exc[@]};i++)); do
    gpu=${gpus[$(($count % ${#gpus[@]}))]}
    CUDA_VISIBLE_DEVICES="$gpu" ${commands[${exc[i]}]} &

    count=$(($count + 1))
    if [ $(($count % $threads)) -eq 0 ]; then
        wait
    fi
    # for random seeds
    sleep $((RANDOM % 60 + 60))
done
wait