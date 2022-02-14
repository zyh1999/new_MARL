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
  "python src/main.py --config=qmix_latent --env-config=sc2 with t_max=4050000 seed=137992755 name=qmix_latent_3s_vs_5z env_args.map_name=3s5z checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/3s_vs_5z/qmix_latent/2021-09-22_19-03-14/models\""
  "python src/main.py --config=qmix_latent --env-config=sc2 with t_max=4050000 seed=993835731 name=qmix_latent_3s_vs_5z env_args.map_name=3s5z checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/3s_vs_5z/qmix_latent/2021-09-22_19-04-26/models\""
  "python src/main.py --config=qmix_latent --env-config=sc2 with t_max=4050000 seed=143948270 name=qmix_latent_3s_vs_5z env_args.map_name=3s5z checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/3s_vs_5z/qmix_latent/2021-09-23_13-59-17/models\""
  "python src/main.py --config=qmix_latent --env-config=sc2 with t_max=4050000 seed=201021640 name=qmix_latent_3s_vs_5z env_args.map_name=3s5z checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/3s_vs_5z/qmix_latent/2021-09-23_14-00-37/models\""
  "python src/main.py --config=qmix_latent --env-config=sc2 with t_max=4050000 seed=703047415 name=qmix_latent_3s_vs_5z env_args.map_name=3s5z checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/3s_vs_5z/qmix_latent/2021-09-24_09-03-31/models\""
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