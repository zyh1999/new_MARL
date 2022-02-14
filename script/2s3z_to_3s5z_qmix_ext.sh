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
  "python src/main.py --config=qmix_ext --env-config=sc2 with t_max=4050000 seed=939657615 name=qmix_ext_2s3z env_args.map_name=3s5z checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/2s3z/qmix_ext/2021-09-11_09-16-27/models\""
  "python src/main.py --config=qmix_ext --env-config=sc2 with t_max=4050000 seed=644220859 name=qmix_ext_2s3z env_args.map_name=3s5z checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/2s3z/qmix_ext/2021-09-12_11-49-48/models\""
  "python src/main.py --config=qmix_ext --env-config=sc2 with t_max=4050000 seed=543853909 name=qmix_ext_2s3z env_args.map_name=3s5z checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/2s3z/qmix_ext/2021-09-12_16-09-30/models\""
  "python src/main.py --config=qmix_ext --env-config=sc2 with t_max=4050000 seed=572698462 name=qmix_ext_2s3z env_args.map_name=3s5z checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/2s3z/qmix_ext/2021-09-13_08-52-12/models\""
  "python src/main.py --config=qmix_ext --env-config=sc2 with t_max=4050000 seed=279313002 name=qmix_ext_2s3z env_args.map_name=3s5z checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/2s3z/qmix_ext/2021-09-13_09-57-40/models\""
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