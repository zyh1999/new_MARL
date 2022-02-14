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
  "python src/main.py --config=qmix_latent --env-config=sc2 with t_max=4050000 seed=672155992 name=qmix_latent_10m_vs_11m env_args.map_name=3m checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/10m_vs_11m/qmix_latent/2021-09-22_18-14-54/models\""
  "python src/main.py --config=qmix_latent --env-config=sc2 with t_max=4050000 seed=183347219 name=qmix_latent_10m_vs_11m env_args.map_name=3m checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/10m_vs_11m/qmix_latent/2021-09-25_03-31-21/models\""
  "python src/main.py --config=qmix_latent --env-config=sc2 with t_max=4050000 seed=90384810 name=qmix_latent_10m_vs_11m env_args.map_name=3m checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/10m_vs_11m/qmix_latent/2021-09-26_10-39-46/models\""
  "python src/main.py --config=qmix_latent --env-config=sc2 with t_max=4050000 seed=629426021 name=qmix_latent_10m_vs_11m env_args.map_name=3m checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/10m_vs_11m/qmix_latent/2021-09-26_10-41-33/models\""
  "python src/main.py --config=qmix_latent --env-config=sc2 with t_max=4050000 seed=676417642 name=qmix_latent_10m_vs_11m env_args.map_name=3m checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/10m_vs_11m/qmix_latent/2021-09-27_16-38-15/models\""
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