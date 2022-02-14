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
  "python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=549505994 name=vdn_updet_5m_vs_6m env_args.map_name=8m_vs_9m checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/5m_vs_6m/vdn_updet/2021-08-20_21-56-53/models\""
  "python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=662232093 name=vdn_updet_5m_vs_6m env_args.map_name=8m_vs_9m checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/5m_vs_6m/vdn_updet/2021-08-20_21-58-04/models\""
  "python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=609448473 name=vdn_updet_5m_vs_6m env_args.map_name=8m_vs_9m checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/5m_vs_6m/vdn_updet/2021-08-20_21-59-43/models\""
  "python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=360194117 name=vdn_updet_5m_vs_6m env_args.map_name=8m_vs_9m checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/5m_vs_6m/vdn_updet/2021-08-22_01-18-34/models\""
  "python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=810530385 name=vdn_updet_5m_vs_6m env_args.map_name=8m_vs_9m checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/5m_vs_6m/vdn_updet/2021-08-22_01-20-01/models\""
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