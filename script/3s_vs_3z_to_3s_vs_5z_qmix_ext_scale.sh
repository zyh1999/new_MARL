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
  "python src/main.py --config=qmix_ext_scale --env-config=sc2 with t_max=4050000 seed=265902083 name=qmix_ext_scale_3s_vs_3z env_args.map_name=3s_vs_5z checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/3s_vs_3z/qmix_ext_scale/2021-09-13_14-36-19/models\""
  "python src/main.py --config=qmix_ext_scale --env-config=sc2 with t_max=4050000 seed=823188909 name=qmix_ext_scale_3s_vs_3z env_args.map_name=3s_vs_5z checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/3s_vs_3z/qmix_ext_scale/2021-09-13_14-38-06/models\""
  "python src/main.py --config=qmix_ext_scale --env-config=sc2 with t_max=4050000 seed=299747311 name=qmix_ext_scale_3s_vs_3z env_args.map_name=3s_vs_5z checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/3s_vs_3z/qmix_ext_scale/2021-09-13_14-39-46/models\""
  "python src/main.py --config=qmix_ext_scale --env-config=sc2 with t_max=4050000 seed=761706862 name=qmix_ext_scale_3s_vs_3z env_args.map_name=3s_vs_5z checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/3s_vs_3z/qmix_ext_scale/2021-09-14_08-49-20/models\""
  "python src/main.py --config=qmix_ext_scale --env-config=sc2 with t_max=4050000 seed=804112327 name=qmix_ext_scale_3s_vs_3z env_args.map_name=3s_vs_5z checkpoint_path=\"$NFS_HOME/code/pymarl/results/exp_v2/sc2/3s_vs_3z/qmix_ext_scale/2021-09-14_08-50-37/models\""
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