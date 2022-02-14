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
  "python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=884415179 name=vdn_updet_8m_vs_9m env_args.map_name=10m_vs_11m checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/8m_vs_9m/vdn_updet/2021-08-20_21-57-44/models\""
  "python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=83883218 name=vdn_updet_8m_vs_9m env_args.map_name=10m_vs_11m checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/8m_vs_9m/vdn_updet/2021-08-22_05-13-58/models\""
  "python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=21843385 name=vdn_updet_8m_vs_9m env_args.map_name=10m_vs_11m checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/8m_vs_9m/vdn_updet/2021-08-22_05-15-27/models\""
  "python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=532323478 name=vdn_updet_8m_vs_9m env_args.map_name=10m_vs_11m checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/8m_vs_9m/vdn_updet/2021-08-23_09-53-06/models\""
  "python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=879165945 name=vdn_updet_8m_vs_9m env_args.map_name=10m_vs_11m checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/8m_vs_9m/vdn_updet/2021-08-25_12-42-56/models\""
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