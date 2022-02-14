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
  "python src/main.py --config=qmix_latent_scale --env-config=sc2 with t_max=4050000 name=qmix_latent_scale_2s3z env_args.map_name=3s5z checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/2s3z/qmix_latent_scale/2021-09-22_18-25-40/models\""
  "python src/main.py --config=qmix_latent_scale --env-config=sc2 with t_max=4050000 name=qmix_latent_scale_2s3z env_args.map_name=3s5z checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/2s3z/qmix_latent_scale/2021-09-22_18-26-40/models\""
  "python src/main.py --config=qmix_latent_scale --env-config=sc2 with t_max=4050000 name=qmix_latent_scale_2s3z env_args.map_name=3s5z checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/2s3z/qmix_latent_scale/2021-09-23_20-11-27/models\""
  "python src/main.py --config=qmix_latent_scale --env-config=sc2 with t_max=4050000 name=qmix_latent_scale_2s3z env_args.map_name=3s5z checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/2s3z/qmix_latent_scale/2021-09-23_20-12-59/models\""
  "python src/main.py --config=qmix_latent_scale --env-config=sc2 with t_max=4050000 name=qmix_latent_scale_2s3z env_args.map_name=3s5z checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/2s3z/qmix_latent_scale/2021-09-24_21-42-01/models\""
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