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
  "python src/main.py --config=qmix_ext --env-config=sc2 with t_max=4050000 name=qmix_ext_10m_vs_11m env_args.map_name=8m_vs_9m checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/10m_vs_11m/qmix_ext/2021-09-11_09-11-48/models\""
  "python src/main.py --config=qmix_ext --env-config=sc2 with t_max=4050000 name=qmix_ext_10m_vs_11m env_args.map_name=8m_vs_9m checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/10m_vs_11m/qmix_ext/2021-09-13_13-29-56/models\""
  "python src/main.py --config=qmix_ext --env-config=sc2 with t_max=4050000 name=qmix_ext_10m_vs_11m env_args.map_name=8m_vs_9m checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/10m_vs_11m/qmix_ext/2021-09-14_11-13-07/models\""
  "python src/main.py --config=qmix_ext --env-config=sc2 with t_max=4050000 name=qmix_ext_10m_vs_11m env_args.map_name=8m_vs_9m checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/10m_vs_11m/qmix_ext/2021-09-14_15-18-24/models\""
  "python src/main.py --config=qmix_ext --env-config=sc2 with t_max=4050000 name=qmix_ext_10m_vs_11m env_args.map_name=8m_vs_9m checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/10m_vs_11m/qmix_ext/2021-09-16_09-59-18/models\""
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