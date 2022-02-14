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
  "python src/main.py --config=qmix_latent --env-config=sc2 with t_max=4050000 name=qmix_latent_5m_vs_6m env_args.map_name=3m checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/5m_vs_6m/qmix_latent/2021-09-19_08-22-06/models\""
  "python src/main.py --config=qmix_latent --env-config=sc2 with t_max=4050000 name=qmix_latent_5m_vs_6m env_args.map_name=3m checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/5m_vs_6m/qmix_latent/2021-09-19_10-54-14/models\""
  "python src/main.py --config=qmix_latent --env-config=sc2 with t_max=4050000 name=qmix_latent_5m_vs_6m env_args.map_name=3m checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/5m_vs_6m/qmix_latent/2021-09-20_09-54-28/models\""
  "python src/main.py --config=qmix_latent --env-config=sc2 with t_max=4050000 name=qmix_latent_5m_vs_6m env_args.map_name=3m checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/5m_vs_6m/qmix_latent/2021-09-26_10-30-48/models\""
  "python src/main.py --config=qmix_latent --env-config=sc2 with t_max=4050000 name=qmix_latent_5m_vs_6m env_args.map_name=3m checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/5m_vs_6m/qmix_latent/2021-09-26_10-31-57/models\""
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