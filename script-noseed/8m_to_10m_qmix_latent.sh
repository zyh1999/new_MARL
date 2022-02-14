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
  "python src/main.py --config=qmix_latent --env-config=sc2 with t_max=4050000 name=qmix_latent_8m_vs_9m env_args.map_name=10m_vs_11m checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/8m_vs_9m/qmix_latent/2021-09-22_18-18-23/models\""
  "python src/main.py --config=qmix_latent --env-config=sc2 with t_max=4050000 name=qmix_latent_8m_vs_9m env_args.map_name=10m_vs_11m checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/8m_vs_9m/qmix_latent/2021-09-24_00-03-45/models\""
  "python src/main.py --config=qmix_latent --env-config=sc2 with t_max=4050000 name=qmix_latent_8m_vs_9m env_args.map_name=10m_vs_11m checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/8m_vs_9m/qmix_latent/2021-09-24_00-04-46/models\""
  "python src/main.py --config=qmix_latent --env-config=sc2 with t_max=4050000 name=qmix_latent_8m_vs_9m env_args.map_name=10m_vs_11m checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/8m_vs_9m/qmix_latent/2021-09-25_07-24-59/models\""
  "python src/main.py --config=qmix_latent --env-config=sc2 with t_max=4050000 name=qmix_latent_8m_vs_9m env_args.map_name=10m_vs_11m checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/8m_vs_9m/qmix_latent/2021-09-26_10-34-51/models\""
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