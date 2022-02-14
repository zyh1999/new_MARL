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
  "python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=763736170 name=vdn_updet_3s5z env_args.map_name=3s_vs_5z checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/3s5z/vdn_updet/2021-08-22_19-29-25/models\""
  "python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=690386827 name=vdn_updet_3s5z env_args.map_name=3s_vs_5z checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/3s5z/vdn_updet/2021-08-22_19-30-41/models\""
  "python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=80098928 name=vdn_updet_3s5z env_args.map_name=3s_vs_5z checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/3s5z/vdn_updet/2021-08-23_18-42-19/models\""
  "python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=383634973 name=vdn_updet_3s5z env_args.map_name=3s_vs_5z checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/3s5z/vdn_updet/2021-08-23_18-44-06/models\""
  "python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=187644705 name=vdn_updet_3s5z env_args.map_name=3s_vs_5z checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/3s5z/vdn_updet/2021-08-24_21-52-30/models\""
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