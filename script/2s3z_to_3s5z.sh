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
  "python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=557451342 name=vdn_updet_2s3z env_args.map_name=3s5z checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/2s3z/vdn_updet/2021-08-22_19-25-02/models\""
  "python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=605247556 name=vdn_updet_2s3z env_args.map_name=3s5z checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/2s3z/vdn_updet/2021-08-22_19-26-37/models\""
  "python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=990739171 name=vdn_updet_2s3z env_args.map_name=3s5z checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/2s3z/vdn_updet/2021-08-24_21-39-37/models\""
  "python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=550663257 name=vdn_updet_2s3z env_args.map_name=3s5z checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/2s3z/vdn_updet/2021-08-24_21-40-59/models\""
  "python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=814750593 name=vdn_updet_2s3z env_args.map_name=3s5z checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/2s3z/vdn_updet/2021-08-25_22-58-30/models\""
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