# render
python src/main.py --config=vdn --env-config=sc2 with env_args.map_name=5m_vs_6m checkpoint_path="/Users/liushunyu/Desktop/code/pymarl/results/exp_v2/sc2_grid/5m_vs_6m/vdn/2021-08-14_11-14-03/models" evaluate=True render=True

# 保存 replay
python src/main.py --config=vdn --env-config=sc2 with env_args.map_name=5m_vs_6m checkpoint_path="/Users/liushunyu/Desktop/code/pymarl/results/exp_v2/sc2_grid/5m_vs_6m/vdn/2021-08-14_11-14-03/models" save_replay=True

# 渲染 replay
# https://github.com/deepmind/pysc2/issues/288
python -m pysc2.bin.play --norender --rgb_minimap_size 0 --replay 5m_vs_6m_2021-08-18-05-51-14.SC2Replay


# qmix_ext_scale 5m_vs_6m to 10m_vs_11m
#CUDA_VISIBLE_DEVICES="0" python src/main.py --config=qmix_ext_scale --env-config=sc2 with name=qmix_ext_scale_5m_vs_6m env_args.map_name=10m_vs_11m checkpoint_path="/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/5m_vs_6m/qmix_ext_scale/2021-09-12_19-14-44/models" statistic=True

# qmix_ext 5m_vs_6m to 10m_vs_11m
#CUDA_VISIBLE_DEVICES="0" python src/main.py --config=qmix_ext --env-config=sc2 with name=qmix_ext_5m_vs_6m env_args.map_name=10m_vs_11m checkpoint_path="/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/5m_vs_6m/qmix_ext/2021-09-11_09-13-06/models" statistic=True

# vdn_updet 5m_vs_6m to 10m_vs_11m
CUDA_VISIBLE_DEVICES="0" python src/main.py --config=vdn_updet --env-config=sc2 with name=vdn_updet_5m_vs_6m env_args.map_name=10m_vs_11m checkpoint_path="/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/5m_vs_6m/vdn_updet/2021-08-20_21-56-53/models" statistic=True

