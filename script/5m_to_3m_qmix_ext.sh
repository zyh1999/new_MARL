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
  "python src/main.py --config=qmix_ext --env-config=sc2 with t_max=4050000 seed=914162233 name=qmix_ext_5m_vs_6m env_args.map_name=3m checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/5m_vs_6m/qmix_ext/2021-09-11_09-13-06/models\""
  "python src/main.py --config=qmix_ext --env-config=sc2 with t_max=4050000 seed=94033454 name=qmix_ext_5m_vs_6m env_args.map_name=3m checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/5m_vs_6m/qmix_ext/2021-09-12_16-26-16/models\""
  "python src/main.py --config=qmix_ext --env-config=sc2 with t_max=4050000 seed=541755615 name=qmix_ext_5m_vs_6m env_args.map_name=3m checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/5m_vs_6m/qmix_ext/2021-09-12_16-27-53/models\""
  "python src/main.py --config=qmix_ext --env-config=sc2 with t_max=4050000 seed=37570131 name=qmix_ext_5m_vs_6m env_args.map_name=3m checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/5m_vs_6m/qmix_ext/2021-09-12_19-02-32/models\""
  "python src/main.py --config=qmix_ext --env-config=sc2 with t_max=4050000 seed=792968117 name=qmix_ext_5m_vs_6m env_args.map_name=3m checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/5m_vs_6m/qmix_ext/2021-09-12_19-03-38/models\""
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