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
  "python src/main.py --config=qmix_latent_scale --env-config=sc2 with t_max=4050000 seed=360099969 name=qmix_latent_scale_3m env_args.map_name=10m_vs_11m checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/3m/qmix_latent_scale/2021-09-22_06-13-46/models\""
  "python src/main.py --config=qmix_latent_scale --env-config=sc2 with t_max=4050000 seed=927313682 name=qmix_latent_scale_3m env_args.map_name=10m_vs_11m checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/3m/qmix_latent_scale/2021-09-23_03-14-24/models\""
  "python src/main.py --config=qmix_latent_scale --env-config=sc2 with t_max=4050000 seed=785788412 name=qmix_latent_scale_3m env_args.map_name=10m_vs_11m checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/3m/qmix_latent_scale/2021-09-23_03-15-32/models\""
  "python src/main.py --config=qmix_latent_scale --env-config=sc2 with t_max=4050000 seed=807648481 name=qmix_latent_scale_3m env_args.map_name=10m_vs_11m checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/3m/qmix_latent_scale/2021-09-23_23-36-12/models\""
  "python src/main.py --config=qmix_latent_scale --env-config=sc2 with t_max=4050000 seed=843114148 name=qmix_latent_scale_3m env_args.map_name=10m_vs_11m checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/3m/qmix_latent_scale/2021-09-26_10-27-23/models\""
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