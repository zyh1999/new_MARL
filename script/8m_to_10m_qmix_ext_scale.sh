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
  "python src/main.py --config=qmix_ext_scale --env-config=sc2 with t_max=4050000 seed=729391332 name=qmix_ext_scale_8m_vs_9m env_args.map_name=10m_vs_11m checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/8m_vs_9m/qmix_ext_scale/2021-09-13_22-40-22/models\""
  "python src/main.py --config=qmix_ext_scale --env-config=sc2 with t_max=4050000 seed=741724017 name=qmix_ext_scale_8m_vs_9m env_args.map_name=10m_vs_11m checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/8m_vs_9m/qmix_ext_scale/2021-09-13_22-42-13/models\""
  "python src/main.py --config=qmix_ext_scale --env-config=sc2 with t_max=4050000 seed=928882717 name=qmix_ext_scale_8m_vs_9m env_args.map_name=10m_vs_11m checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/8m_vs_9m/qmix_ext_scale/2021-09-15_01-45-48/models\""
  "python src/main.py --config=qmix_ext_scale --env-config=sc2 with t_max=4050000 seed=886181106 name=qmix_ext_scale_8m_vs_9m env_args.map_name=10m_vs_11m checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/8m_vs_9m/qmix_ext_scale/2021-09-15_15-46-54/models\""
  "python src/main.py --config=qmix_ext_scale --env-config=sc2 with t_max=4050000 seed=880777079 name=qmix_ext_scale_8m_vs_9m env_args.map_name=10m_vs_11m checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/8m_vs_9m/qmix_ext_scale/2021-09-15_15-48-07/models\""
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