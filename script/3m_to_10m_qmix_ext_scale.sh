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
  "python src/main.py --config=qmix_ext_scale --env-config=sc2 with t_max=4050000 seed=948538477 name=qmix_ext_scale_3m env_args.map_name=10m_vs_11m checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/3m/qmix_ext_scale/2021-09-12_19-04-08/models\""
  "python src/main.py --config=qmix_ext_scale --env-config=sc2 with t_max=4050000 seed=850212166 name=qmix_ext_scale_3m env_args.map_name=10m_vs_11m checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/3m/qmix_ext_scale/2021-09-12_19-05-31/models\""
  "python src/main.py --config=qmix_ext_scale --env-config=sc2 with t_max=4050000 seed=941328822 name=qmix_ext_scale_3m env_args.map_name=10m_vs_11m checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/3m/qmix_ext_scale/2021-09-12_19-07-24/models\""
  "python src/main.py --config=qmix_ext_scale --env-config=sc2 with t_max=4050000 seed=116711445 name=qmix_ext_scale_3m env_args.map_name=10m_vs_11m checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/3m/qmix_ext_scale/2021-09-12_19-09-11/models\""
  "python src/main.py --config=qmix_ext_scale --env-config=sc2 with t_max=4050000 seed=236949941 name=qmix_ext_scale_3m env_args.map_name=10m_vs_11m checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/3m/qmix_ext_scale/2021-09-12_19-11-05/models\""
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