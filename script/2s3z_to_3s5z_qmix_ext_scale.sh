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
  "python src/main.py --config=qmix_ext_scale --env-config=sc2 with t_max=4050000 seed=506701265 name=qmix_ext_scale_2s3z env_args.map_name=3s5z checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/2s3z/qmix_ext_scale/2021-09-12_19-16-51/models\""
  "python src/main.py --config=qmix_ext_scale --env-config=sc2 with t_max=4050000 seed=951954444 name=qmix_ext_scale_2s3z env_args.map_name=3s5z checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/2s3z/qmix_ext_scale/2021-09-12_19-18-38/models\""
  "python src/main.py --config=qmix_ext_scale --env-config=sc2 with t_max=4050000 seed=993993570 name=qmix_ext_scale_2s3z env_args.map_name=3s5z checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/2s3z/qmix_ext_scale/2021-09-12_19-20-15/models\""
  "python src/main.py --config=qmix_ext_scale --env-config=sc2 with t_max=4050000 seed=82081060 name=qmix_ext_scale_2s3z env_args.map_name=3s5z checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/2s3z/qmix_ext_scale/2021-09-13_13-49-58/models\""
  "python src/main.py --config=qmix_ext_scale --env-config=sc2 with t_max=4050000 seed=122371991 name=qmix_ext_scale_2s3z env_args.map_name=3s5z checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/2s3z/qmix_ext_scale/2021-09-13_13-51-56/models\""
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