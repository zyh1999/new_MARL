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
  "python src/main.py --config=qmix_ext_scale --env-config=sc2 with t_max=4050000 seed=764856992 name=qmix_ext_scale_5m_vs_6m env_args.map_name=8m_vs_9m checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/5m_vs_6m/qmix_ext_scale/2021-09-12_19-14-44/models\""
  "python src/main.py --config=qmix_ext_scale --env-config=sc2 with t_max=4050000 seed=590675361 name=qmix_ext_scale_5m_vs_6m env_args.map_name=8m_vs_9m checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/5m_vs_6m/qmix_ext_scale/2021-09-12_19-15-41/models\""
  "python src/main.py --config=qmix_ext_scale --env-config=sc2 with t_max=4050000 seed=969101898 name=qmix_ext_scale_5m_vs_6m env_args.map_name=8m_vs_9m checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/5m_vs_6m/qmix_ext_scale/2021-09-13_11-33-33/models\""
  "python src/main.py --config=qmix_ext_scale --env-config=sc2 with t_max=4050000 seed=721623348 name=qmix_ext_scale_5m_vs_6m env_args.map_name=8m_vs_9m checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/5m_vs_6m/qmix_ext_scale/2021-09-13_20-41-33/models\""
  "python src/main.py --config=qmix_ext_scale --env-config=sc2 with t_max=4050000 seed=915198978 name=qmix_ext_scale_5m_vs_6m env_args.map_name=8m_vs_9m checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/5m_vs_6m/qmix_ext_scale/2021-09-14_03-07-08/models\""
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