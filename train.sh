# 批量运行代码
# bash run.sh config_name env_name map_name_list arg_list experinments_threads_num gpu_list experinments_num
# bash run_fifo.sh config_name env_name map_name_list arg_list experinments_threads_num gpu_list experinments_num

# run.sh 例子：每跑完 experinments_threads_num 才跑下一批次的线程
bash run.sh token_qmix_attn sc2 5m_vs_6m use_cuda=True 2 1 6
bash run.sh token_qmix_attn sc2 3m,5m_vs_6m use_cuda=True 2 0,1 6

# run_fifo.sh 例子：每次保证有 experinments_threads_num 个线程在跑
bash run_fifo.sh token_qmix_attn sc2 5m_vs_6m use_cuda=True 2 1 6
bash run_fifo.sh token_qmix_attn sc2 3m,5m_vs_6m use_cuda=True 2 0,1 6


CUDA_VISIBLE_DEVICES="0" python src/main.py --config=vdn --env-config=sc2 with env_args.map_name=10m_vs_11m
CUDA_VISIBLE_DEVICES="0" python src/main.py --config=token_qmix_attn --env-config=sc2 with env_args.map_name=2s3z
CUDA_VISIBLE_DEVICES="0" python src/main.py --config=entity_qmix_attn --env-config=sc2mt with env_args.map_name=3-8sz_symmetric


# sc2
# Name            Agents  Enemies Limit
# 3m              3       3       60
# 8m              8       8       120
# 25m             25      25      150
# 5m_vs_6m        5       6       70
# 8m_vs_9m        8       9       120
# 10m_vs_11m      10      11      150
# 27m_vs_30m      27      30      180
# MMM             10      10      150
# MMM2            10      12      180
# 2s3z            5       5       120
# 3s5z            8       8       150
# 3s5z_vs_3s6z    8       9       170
# 3s_vs_3z        3       3       150
# 3s_vs_4z        3       4       200
# 3s_vs_5z        3       5       250
# 1c3s5z          9       9       180
# 2m_vs_1z        2       1       150
# corridor        6       24      400
# 6h_vs_8z        6       8       150
# 2s_vs_1sc       2       1       300
# so_many_baneling 7       32      100
# bane_vs_bane    24      24      200
# 2c_vs_64zg      2       64      400


# particle
# htag_8_4_2   # ally_num, enemy_num, num_landmarks
# tag_16_16_2  # ally_num, enemy_num, num_landmarks
# spread_8_8   # ally_num, enemy_num(num_landmarks)


# sc2mt
# 3-8m_symmetric
# 3-8sz_symmetric
# 3-8MMM_symmetric
# 3-8csz_symmetric
# 6-11m_asymmetric


bash run_fifo.sh entity_qmix_attn_scale sc2mt 3-8sz_symmetric use_cuda=True 3 2 6  # 8G
bash run_fifo.sh entity_qmix_trans_scale sc2mt 3-8sz_symmetric use_cuda=True 3 3 6  # 11G
bash run_fifo.sh entity_qmix_attn_scale sc2mt 3-8m_symmetric use_cuda=True 3 6 6  # 8G
bash run_fifo.sh entity_qmix_trans_scale sc2mt 3-8m_symmetric use_cuda=True 3 0 6  # 11G
bash run_fifo.sh entity_qmix_attn_scale sc2mt 3-8MMM_symmetric use_cuda=True 3 4 6  # 8G
bash run_fifo.sh entity_qmix_trans_scale sc2mt 3-8MMM_symmetric use_cuda=True 3 5 6  # 11G
bash run_fifo.sh entity_qmix_attn_scale sc2mt 3-8csz_symmetric use_cuda=True 3 7 6  # 8G
bash run_fifo.sh entity_qmix_trans_scale sc2mt 3-8csz_symmetric use_cuda=True 3 0 6  # 11G


bash run_fifo.sh entity_qmix_sparse_attn_scale sc2mt 3-8sz_symmetric use_cuda=True 3 0 6  # 8G
bash run_fifo.sh entity_qmix_sparse_trans_scale sc2mt 3-8sz_symmetric use_cuda=True 3 1 6  # 11G
bash run_fifo.sh entity_qmix_sparse_attn_scale sc2mt 3-8m_symmetric use_cuda=True 3 2 6  # 8G
bash run_fifo.sh entity_qmix_sparse_trans_scale sc2mt 3-8m_symmetric use_cuda=True 3 3 6  # 11G
bash run_fifo.sh entity_qmix_sparse_attn_scale sc2mt 3-8MMM_symmetric use_cuda=True 3 4 6  # 8G
bash run_fifo.sh entity_qmix_sparse_trans_scale sc2mt 3-8MMM_symmetric use_cuda=True 3 5 6  # 11G
bash run_fifo.sh entity_qmix_sparse_attn_scale sc2mt 3-8csz_symmetric use_cuda=True 3 2 6  # 8G
bash run_fifo.sh entity_qmix_sparse_trans_scale sc2mt 3-8csz_symmetric use_cuda=True 3 6 6  # 11G


bash run_fifo.sh entity_qmix_attn sc2mt 3-8m_symmetric use_cuda=True 3 0 6
bash run_fifo.sh entity_qmix_trans sc2mt 3-8m_symmetric use_cuda=True 3 1 6
bash run_fifo.sh entity_qmix_refil_attn sc2mt 3-8m_symmetric use_cuda=True 3 2 6
bash run_fifo.sh entity_qmix_refil_imagine sc2mt 3-8m_symmetric use_cuda=True 3 3 6
bash run_fifo.sh entity_vdn_attn sc2mt 3-8m_symmetric use_cuda=True 3 4 6
bash run_fifo.sh entity_vdn_trans sc2mt 3-8m_symmetric use_cuda=True 3 5 6




bash run_fifo.sh entity_qmix_attn sc2mt 3-8sz_symmetric use_cuda=True 3 0 6  # 8G
bash run_fifo.sh entity_qmix_trans sc2mt 3-8sz_symmetric use_cuda=True 3 1 6  # 11G
bash run_fifo.sh entity_qmix_refil_attn sc2mt 3-8sz_symmetric use_cuda=True 3 2 6  # 13G
bash run_fifo.sh entity_qmix_refil_imagine sc2mt 3-8sz_symmetric use_cuda=True 3 3 6  # 20G
bash run_fifo.sh entity_vdn_attn sc2mt 3-8sz_symmetric use_cuda=True 3 4 6 # 6G
bash run_fifo.sh entity_vdn_trans sc2mt 3-8sz_symmetric use_cuda=True 3 5 6 # 8G
bash run_fifo.sh entity_qmix_refil_imagine_parrallel sc2mt 3-8sz_symmetric use_cuda=True 3 6 6  # 20G
bash run_fifo.sh entity_vdn_refil_attn sc2mt 3-8sz_symmetric use_cuda=True 3 7 6

bash run_fifo.sh entity_qmix_multi_attn sc2mt 3-8sz_symmetric use_cuda=True 3 6 6

bash run_fifo.sh entity_qmix_attn sc2mt 3-8MMM_symmetric,3-8m_symmetric use_cuda=True 3 0 6  # 8G
bash run_fifo.sh entity_qmix_trans sc2mt 3-8MMM_symmetric,3-8m_symmetric use_cuda=True 3 1 6  # 11G
bash run_fifo.sh entity_qmix_refil_attn sc2mt 3-8MMM_symmetric use_cuda=True 3 2 6  # 13G
bash run_fifo.sh entity_qmix_refil_imagine sc2mt 3-8MMM_symmetric use_cuda=True 3 3 6  # 20G
bash run_fifo.sh entity_vdn_attn sc2mt 3-8MMM_symmetric,3-8m_symmetric use_cuda=True 3 4 6 # 6G
bash run_fifo.sh entity_vdn_trans sc2mt 3-8MMM_symmetric,3-8m_symmetric use_cuda=True 3 5 6 # 8G
bash run_fifo.sh entity_qmix_refil_imagine_parrallel sc2mt 3-8MMM_symmetric use_cuda=True 3 6 6  # 20G
bash run_fifo.sh entity_vdn_refil_attn sc2mt 3-8MMM_symmetric,3-8m_symmetric use_cuda=True 3 6 6

bash run_fifo.sh entity_qmix_refil_attn sc2mt 3-8m_symmetric use_cuda=True 3 2 6  # 13G
bash run_fifo.sh entity_qmix_refil_imagine sc2mt 3-8m_symmetric use_cuda=True 3 7 6  # 20G
bash run_fifo.sh entity_qmix_refil_imagine_parrallel sc2mt 3-8m_symmetric use_cuda=True 3 6 6  # 20G

bash run_fifo.sh entity_qmix_attn sc2mt 3-8csz_symmetric use_cuda=True 3 0 6  # 8G
bash run_fifo.sh entity_qmix_trans sc2mt 3-8csz_symmetric use_cuda=True 3 1 6  # 11G
bash run_fifo.sh entity_qmix_refil_attn sc2mt 3-8csz_symmetric use_cuda=True 3 2 6  # 13G
bash run_fifo.sh entity_qmix_refil_imagine sc2mt 3-8csz_symmetric use_cuda=True 3 3 6  # 20G
bash run_fifo.sh entity_vdn_attn sc2mt 3-8csz_symmetric use_cuda=True 3 4 6 # 6G
bash run_fifo.sh entity_vdn_trans sc2mt 3-8csz_symmetric use_cuda=True 3 5 6 # 8G
bash run_fifo.sh entity_qmix_refil_imagine_parrallel sc2mt 3-8csz_symmetric use_cuda=True 3 6 6  # 20G
bash run_fifo.sh entity_vdn_refil_attn sc2mt 3-8csz_symmetric use_cuda=True 3 7 6


bash run_fifo.sh token_qmix_wise_attn sc2 10m_vs_11m use_cuda=True 2 0 6  # 8G
bash run_fifo.sh token_vdn_wise_attn sc2 10m_vs_11m use_cuda=True 2 1 6  # 11G
bash run_fifo.sh token_qmix_wise_trans sc2 10m_vs_11m use_cuda=True 2 2 6  # 13G
bash run_fifo.sh token_vdn_wise_trans sc2 10m_vs_11m use_cuda=True 2 3 6  # 20G
bash run_fifo.sh token_qmix_updet sc2 10m_vs_11m use_cuda=True 2 3 6
bash run_fifo.sh token_vdn_updet sc2 10m_vs_11m use_cuda=True 2 2,3 6

bash run_exp.sh token_qmix_wise_trans sc2 10m_vs_11m use_cuda=True 4 0,1,2,3 6
bash run_exp.sh token_qmix_wise_trans sc2 8m_vs_9m use_cuda=True 4 0,1,2,3 6
bash run_exp.sh token_vdn_wise_attn,token_qmix_wise_attn sc2 3s_vs_5z use_cuda=True 4 0,1,2,3 4

# douzaipaole

bash run_exp.sh token_qmix_wise_trans,token_vdn_wise_attn sc2 5m_vs_6m use_cuda=True 2 2,3 1
bash run_exp.sh token_qmix_wise_trans,token_vdn_wise_trans,token_qmix_updet sc2 2s3z use_cuda=True 6 0,1 2
bash run_exp.sh token_vdn_wise_attn sc2 2s3z use_cuda=True 1 2 1


bash run_exp.sh qplex sc2 5m_vs_6m,3s_vs_5z,3s_vs_3z,3m use_cuda=True 6 0,1 6
bash run_exp.sh qplex sc2 3s5z,10m_vs_11m,8m_vs_9m,2s3z use_cuda=True 8 0,1,2,3 6

bash run_exp.sh iql,qmix,vdn,qtran,coma sc2 5m_vs_6m,3s_vs_5z,3s_vs_3z,3m use_cuda=True 4 0,1 2
bash run_exp.sh iql,qmix,vdn,qtran,coma sc2 3s5z,10m_vs_11m,8m_vs_9m,2s3z use_cuda=True 4 0 2

bash run_fifo.sh token_qmix_wise_attn sc2 5m_vs_6m use_cuda=True 2 0 6  # 8G
bash run_fifo.sh token_vdn_wise_attn sc2 5m_vs_6m use_cuda=True 2 1 6  # 11G
bash run_fifo.sh token_qmix_wise_trans sc2 5m_vs_6m use_cuda=True 2 0 6  # 13G
bash run_fifo.sh token_vdn_wise_trans sc2 5m_vs_6m use_cuda=True 2 0 6  # 20G
bash run_fifo.sh token_qmix_updet sc2 5m_vs_6m use_cuda=True 2 1 6
bash run_fifo.sh token_vdn_updet sc2 5m_vs_6m use_cuda=True 2 3 6 # 6G
bash run_fifo.sh token_vdn_dyan sc2 5m_vs_6m use_cuda=True 2 2 6 # 6G
bash run_fifo.sh token_qmix_wise_branch_attn sc2 5m_vs_6m use_cuda=True 2 1 6  # 8G


bash run_fifo.sh token_qmix_wise_attn sc2 3s5z,3s_vs_5z use_cuda=True 2 2 6  # 8G
bash run_fifo.sh token_vdn_wise_attn sc2 3s5z,3s_vs_5z use_cuda=True 2 3 6  # 11G
bash run_fifo.sh token_qmix_wise_trans sc2 3s5z,3s_vs_5z use_cuda=True 2 0 6  # 13G
bash run_fifo.sh token_vdn_wise_trans sc2 3s5z,3s_vs_5z use_cuda=True 2 2 6  # 20G
bash run_fifo.sh token_qmix_updet sc2 3s5z,3s_vs_5z use_cuda=True 2 1 6 # 6G
bash run_fifo.sh token_vdn_updet sc2 3s5z,3s_vs_5z use_cuda=True 2 1 6 # 6G

bash run_fifo.sh token_qmix_wise_attn sc2 2s3z,3s_vs_3z use_cuda=True 2 0 6  # 8G
bash run_fifo.sh token_vdn_wise_attn sc2 2s3z,3s_vs_3z use_cuda=True 2 2 6  # 11G
bash run_fifo.sh token_qmix_wise_trans sc2 2s3z,3s_vs_3z use_cuda=True 2 1 6  # 13G
bash run_fifo.sh token_vdn_wise_trans sc2 2s3z,3s_vs_3z use_cuda=True 2 0 6  # 20G
bash run_fifo.sh token_qmix_updet sc2 2s3z,3s_vs_3z use_cuda=True 2 1 6 # 6G
bash run_fifo.sh token_vdn_updet sc2 2s3z,3s_vs_3z use_cuda=True 2 1 6 # 6G

bash run_fifo.sh token_qmix_wise_attn,token_vdn_wise_attn,token_vdn_wise_trans sc2 3s_vs_3z use_cuda=True 4 2,3 3
bash run_fifo.sh token_qmix_wise_trans sc2 3s_vs_3z use_cuda=True 1 0 1
bash run_fifo.sh token_qmix_updet sc2 3s_vs_3z use_cuda=True 2 1 4



bash run_fifo.sh token_qmix_wise_attn sc2 8m_vs_9m,3m use_cuda=True 3 2 6  # 8G
bash run_fifo.sh token_vdn_wise_attn sc2 8m_vs_9m,3m use_cuda=True 2 0,1 6  # 11G
bash run_fifo.sh token_qmix_wise_trans sc2 8m_vs_9m,3m use_cuda=True 2 0 6  # 13G
bash run_fifo.sh token_vdn_wise_trans sc2 8m_vs_9m,3m use_cuda=True 2 1 6  # 20G
bash run_fifo.sh token_qmix_updet sc2 8m_vs_9m,3m use_cuda=True 2 3 6 # 6G
bash run_fifo.sh token_vdn_updet sc2 8m_vs_9m,3m use_cuda=True 2 0,2 6 # 6G


bash run_exp.sh iql,qmix,vdn,qtran,coma sc2 MMM2,MMM use_cuda=True 9 1,2,3 6

bash run_fifo.sh token_qmix_wise_attn sc2 MMM2,MMM use_cuda=True 3 2 6  # 8G
bash run_fifo.sh token_vdn_wise_attn sc2 MMM2,MMM use_cuda=True 2 0,1 6  # 11G
bash run_fifo.sh token_qmix_wise_trans sc2 MMM2,MMM use_cuda=True 2 0 6  # 13G
bash run_fifo.sh token_vdn_wise_trans sc2 MMM2,MMM use_cuda=True 2 1 6  # 20G
bash run_fifo.sh token_qmix_updet sc2 MMM2,MMM use_cuda=True 2 3 6 # 6G
bash run_fifo.sh token_vdn_updet sc2 MMM2,MMM use_cuda=True 2 0,2 6 # 6G



# 20211122
# 这一堆跑了超久都没跑完
#bash run_exp.sh token_qmix_wise_trans sc2 3m use_cuda=True 6 0 6
#bash run_exp.sh token_vdn_wise_trans sc2 3m use_cuda=True 6 1 6
#bash run_exp.sh token_qmix_updet sc2 3m use_cuda=True 6 2 6
#bash run_exp.sh token_vdn_updet sc2 3m use_cuda=True 6 3 6
#bash run_exp.sh token_qmix_wise_trans sc2 8m_vs_9m use_cuda=True 1 2 1
#bash run_exp.sh token_qmix_wise_attn,token_vdn_updet sc2 3s_vs_5z use_cuda=True  2 0,1 2
#bash run_exp.sh token_qmix_wise_attn,token_vdn_updet sc2 3s_vs_5z use_cuda=True  2 0,1 2
#bash run_exp.sh token_qmix_wise_trans,token_vdn_wise_trans sc2 8m_vs_9m use_cuda=True 3 0,1,2 3
#bash run_exp.sh token_vdn_wise_trans sc2 3s_vs_3z use_cuda=True 3 1,0 3
#bash run_exp.sh token_qmix_wise_attn,token_vdn_wise_attn sc2 3s_vs_3z use_cuda=True 2 1 1
#bash run_exp.sh token_qmix_wise_trans,token_vdn_wise_trans sc2 3s_vs_5z use_cuda=True 4 0,1 2
#bash run_exp.sh token_qmix_wise_trans sc2 10m_vs_11m use_cuda=True 1 0 1
#bash run_exp.sh token_qmix_updet,token_vdn_updet sc2 8m_vs_9m use_cuda=True 4 1,2 2
#bash run_exp.sh token_qmix_wise_trans,token_vdn_wise_trans sc2 3s_vs_5z use_cuda=True 2 3 2

# 20211125
# 24
bash run_fifo.sh entity_qmix_sparse_trans_scale sc2mt 3-8sz_symmetric use_cuda=True 3 0 6
bash run_fifo.sh entity_qmix_trans_scale sc2mt 3-8sz_symmetric use_cuda=True 3 1 6
bash run_fifo.sh entity_qmix_trans sc2mt 3-8sz_symmetric use_cuda=True 3 2 6
bash run_fifo.sh entity_vdn_trans sc2mt 3-8sz_symmetric use_cuda=True 3 3 6
bash run_fifo.sh entity_qmix_sparse_trans_scale sc2mt 3-8MMM_symmetric use_cuda=True 3 4 6
bash run_fifo.sh entity_qmix_trans_scale sc2mt 3-8MMM_symmetric use_cuda=True 3 5 6
bash run_fifo.sh entity_qmix_trans sc2mt 3-8MMM_symmetric use_cuda=True 3 6 6
bash run_fifo.sh entity_vdn_trans sc2mt 3-8MMM_symmetric use_cuda=True 3 7 6
# 22
bash run_fifo.sh entity_qmix_sparse_trans_scale sc2mt 3-8m_symmetric use_cuda=True 3 0 6
bash run_fifo.sh entity_qmix_trans_scale sc2mt 3-8m_symmetric use_cuda=True 3 1 6
bash run_fifo.sh entity_qmix_trans sc2mt 3-8m_symmetric use_cuda=True 3 2 6
bash run_fifo.sh entity_vdn_trans sc2mt 3-8m_symmetric use_cuda=True 3 3 6
bash run_fifo.sh entity_qmix_sparse_trans_scale sc2mt 3-8csz_symmetric use_cuda=True 3 7 6
bash run_fifo.sh entity_qmix_trans_scale sc2mt 3-8csz_symmetric use_cuda=True 3 4 6
bash run_fifo.sh entity_qmix_trans sc2mt 3-8csz_symmetric use_cuda=True 3 5 6
bash run_fifo.sh entity_vdn_trans sc2mt 3-8csz_symmetric use_cuda=True 3 6 6

#vipa
bash run_exp.sh token_qmix_wise_attn sc2 3m use_cuda=True 4 0,1 6
bash run_exp.sh token_qmix_wise_attn sc2 3s_vs_3z use_cuda=True 4 0,1 6
bash run_exp.sh token_qmix_wise_attn sc2 3s_vs_5z use_cuda=True 4 0,1 4

# dai
#bash run_exp.sh token_qmix_wise_attn sc2 10m_vs_11m use_cuda=True 4 0,1,2,3 6
#bash run_exp.sh token_qmix_wise_attn sc2 5m_vs_6m use_cuda=True 3 3 6
#bash run_exp.sh token_qmix_wise_attn sc2 3s5z use_cuda=True 3 0,1,2 6
#bash run_exp.sh token_qmix_wise_attn sc2 8m_vs_9m use_cuda=True 3 0,1,2 6
#bash run_exp.sh token_qmix_wise_attn sc2 2s3z use_cuda=True 4 0,1 6



# 2021209
# zj24
bash run_fifo.sh entity_qmix_spotlight_contrastive_trans_scale sc2mt 3-8sz_symmetric use_cuda=True 3 0 6
bash run_fifo.sh entity_qmix_spotlight_contrastive_trans_scale sc2mt 3-8csz_symmetric use_cuda=True 3 1 6
bash run_fifo.sh entity_qmix_spotlight_contrastive_trans_scale sc2mt 3-8MMM_symmetric use_cuda=True 3 2 6
bash run_fifo.sh entity_qmix_spotlight_consistent_trans_scale sc2mt 3-8sz_symmetric use_cuda=True 3 3 6
bash run_fifo.sh entity_qmix_spotlight_consistent_trans_scale sc2mt 3-8csz_symmetric use_cuda=True 3 4 6
bash run_fifo.sh entity_qmix_spotlight_consistent_trans_scale sc2mt 3-8MMM_symmetric use_cuda=True 3 5 6
#bash run_fifo.sh entity_qmix_pattern_contrastive_trans_scale sc2mt 3-8sz_symmetric use_cuda=True 3 6 6  # 跑完3个停了
#bash run_fifo.sh entity_qmix_pattern_contrastive_trans_scale sc2mt 3-8csz_symmetric use_cuda=True 3 7 6  # 跑完3个停了
# zj23   # 跑完3个停了
#bash run_fifo.sh entity_qmix_pattern_contrastive_trans_scale sc2mt 3-8MMM_symmetric use_cuda=True 3 7 6
#bash run_fifo.sh entity_qmix_pattern_consistent_trans_scale sc2mt 3-8sz_symmetric use_cuda=True 3 6 6
#bash run_fifo.sh entity_qmix_pattern_consistent_trans_scale sc2mt 3-8csz_symmetric use_cuda=True 3 5 6
#bash run_fifo.sh entity_qmix_pattern_consistent_trans_scale sc2mt 3-8MMM_symmetric use_cuda=True 3 4 6

# zj
bash run_exp.sh owqmix sc2 10m_vs_11m use_cuda=True 2 0 6
bash run_exp.sh cwqmix sc2 10m_vs_11m use_cuda=True 2 2 6
bash run_exp.sh owqmix sc2 3s_vs_5z,3s_vs_3z,3m use_cuda=True 4 2 6
bash run_exp.sh owqmix sc2 5m_vs_6m,2s3z use_cuda=True 4 1 6
bash run_exp.sh owqmix sc2 3s5z,8m_vs_9m use_cuda=True 4 0 6
bash run_exp.sh cwqmix sc2 3s_vs_5z,3s_vs_3z,3m use_cuda=True 4 2 6
bash run_exp.sh cwqmix sc2 5m_vs_6m,2s3z use_cuda=True 4 3 6
bash run_exp.sh cwqmix sc2 3s5z,8m_vs_9m use_cuda=True 4 1 6

# vipa
bash run_exp.sh qmix sc2 MMM2,MMM use_cuda=True 2 0,1 6
bash run_exp.sh vdn sc2 MMM2,MMM use_cuda=True 4 1 6
bash run_exp.sh qplex sc2 MMM2,MMM use_cuda=True 2 0,1 6
bash run_exp.sh cwqmix sc2 MMM2,MMM use_cuda=True 3 1 6
bash run_exp.sh qtran sc2 MMM2,MMM use_cuda=True 3 0 6
bash run_exp.sh owqmix sc2 MMM2,MMM use_cuda=True 2 2,3 6

# dai
bash run_exp.sh iql sc2 MMM2,MMM use_cuda=True 3 0 6
bash run_exp.sh coma sc2 MMM2,MMM use_cuda=True 6 1 6

# zj24
bash run_fifo.sh entity_qmix_q_pattern_contrastive_trans_scale sc2mt 3-8sz_symmetric use_cuda=True 3 6 6
bash run_fifo.sh entity_qmix_q_pattern_contrastive_trans_scale sc2mt 3-8csz_symmetric use_cuda=True 3 7 6
# zj23
bash run_fifo.sh entity_qmix_q_pattern_contrastive_trans_scale sc2mt 3-8MMM_symmetric use_cuda=True 3 7 6
bash run_fifo.sh entity_qmix_q_spotlight_contrastive_trans_scale sc2mt 3-8sz_symmetric use_cuda=True 3 6 6
bash run_fifo.sh entity_qmix_q_spotlight_contrastive_trans_scale sc2mt 3-8csz_symmetric use_cuda=True 3 5 6
bash run_fifo.sh entity_qmix_q_spotlight_contrastive_trans_scale sc2mt 3-8MMM_symmetric use_cuda=True 3 4 6

bash run_fifo.sh entity_qmix_attn sc2mt_grid 3-8sz_symmetric use_cuda=True 3 4 6
bash run_fifo.sh entity_qmix_trans sc2mt_grid 3-8sz_symmetric use_cuda=True 3 3 6
bash run_fifo.sh entity_qmix_refil_imagine sc2mt_grid 3-8sz_symmetric use_cuda=True 3 2 6
bash run_fifo.sh entity_qmix_q_pattern_contrastive_trans_scale sc2mt_grid 3-8sz_symmetric use_cuda=True 3 1 6
bash run_fifo.sh entity_qmix_q_spotlight_contrastive_trans_scale sc2mt_grid 3-8sz_symmetric use_cuda=True 3 0 6


# 1220
# zj24
bash run_fifo.sh entity_qmix_q_latent_inner_pattern_contrastive_trans_scale sc2mt 3-8MMM_symmetric use_cuda=True 6 0,1 6
bash run_fifo.sh entity_qmix_q_latent_inner_pattern_contrastive_trans_scale sc2mt 3-8sz_symmetric use_cuda=True 6 2,3 6
bash run_fifo.sh entity_qmix_q_latent_inner_pattern_contrastive_trans_scale sc2mt 3-8csz_symmetric use_cuda=True 6 4,5 6

# zj
bash run_exp.sh token_qmix_q_latent_pattern_contrastive_wise_trans_scale sc2 10m_vs_11m use_cuda=True 3 1,2,3 6
bash run_exp.sh token_qmix_q_latent_pattern_contrastive_wise_trans_scale sc2 3s_vs_5z,3s_vs_3z,3m use_cuda=True 4 0 6
bash run_exp.sh token_qmix_q_latent_pattern_contrastive_wise_trans_scale sc2 5m_vs_6m,2s3z use_cuda=True 4 1 6
bash run_exp.sh token_qmix_q_latent_pattern_contrastive_wise_trans_scale sc2 3s5z,8m_vs_9m use_cuda=True 6 0,1 6
bash run_exp.sh token_qmix_q_latent_pattern_contrastive_wise_trans_scale sc2 MMM2,MMM use_cuda=True 4 2,3 6
bash run_exp.sh token_qmix_q_latent_pattern_contrastive_wise_trans_scale sc2 8m_vs_9m use_cuda=True 2 2,3 6
bash run_exp.sh token_qmix_q_latent_pattern_contrastive_wise_trans_scale sc2 MMM2 use_cuda=True 6 1,2,3 6

# vipa
bash run_exp.sh vdn sc2 MMM2,MMM use_cuda=True 4 0,1 2
bash run_exp.sh owqmix,cwqmix sc2 MMM use_cuda=True 2 2 2
bash run_exp.sh owqmix,cwqmix sc2 MMM2 use_cuda=True 4 2,3 6
bash run_exp.sh token_qmix_q_latent_pattern_contrastive_wise_trans_scale sc2 5m_vs_6m use_cuda=True 2 0 2
bash run_exp.sh token_qmix_q_latent_pattern_contrastive_wise_trans_scale sc2 3s5z use_cuda=True 2 1,2 6
bash run_exp.sh token_qmix_q_latent_pattern_contrastive_wise_trans_scale sc2 3s_vs_5z use_cuda=True 1 2 4

# dai
#bash run_exp.sh qmix sc2 MMM2 use_cuda=True 1 0 1
#bash run_exp.sh qtran sc2 MMM use_cuda=True 2 0 5
#bash run_exp.sh qtran sc2 MMM2 use_cuda=True 2 1 5
#bash run_exp.sh cwqmix sc2 MMM2 use_cuda=True 2 0,1 4
bash run_exp.sh token_qmix_q_latent_pattern_contrastive_wise_trans_scale sc2 6h_vs_8z use_cuda=True 2 0,1 6




# 20211231
#zj24
bash run_fifo.sh entity_qmix_q_latent_inner_pattern_contrastive_trans_scale sc2mt 5-11MMM_symmetric use_cuda=True 6 2,6,7 6
#zj23
bash run_fifo.sh entity_qmix_attn sc2mt 5-11MMM_symmetric use_cuda=True 6 0,1 6
bash run_fifo.sh entity_vdn_attn sc2mt 5-11MMM_symmetric use_cuda=True 6 2,3 6
bash run_fifo.sh entity_qmix_refil_imagine sc2mt 5-11MMM_symmetric use_cuda=True 6 4,5,6 6



#zj
bash run_exp.sh token_qmix_q_latent_pattern_contrastive_wise_trans_scale sc2 2c_vs_64zg use_cuda=True 4 0,1,2,3 8
bash run_exp.sh qplex sc2 2c_vs_64zg use_cuda=True 3 1 6
bash run_exp.sh cwqmix sc2 2c_vs_64zg use_cuda=True 3 2 6
bash run_exp.sh owqmix sc2 2c_vs_64zg use_cuda=True 3 3 6


# 20220103
# zj23
bash run_fifo.sh entity_qmix_attn sc2mt 5-11csz_symmetric use_cuda=True 6 0,1 6
bash run_fifo.sh entity_vdn_attn sc2mt 5-11csz_symmetric use_cuda=True 6 2,3 6
bash run_fifo.sh entity_qmix_refil_imagine sc2mt 5-11csz_symmetric use_cuda=True 6 4,5 6
bash run_fifo.sh entity_qmix_q_latent_inner_pattern_contrastive_trans_scale sc2mt 5-11csz_symmetric use_cuda=True 6 6,7 6

# zj24
bash run_fifo.sh entity_qmix_q_latent_inner_pattern_contrastive_trans_scale sc2mt 5-11sz_symmetric use_cuda=True 6 0,5 6
# zj22
bash run_fifo.sh entity_qmix_attn sc2mt 5-11sz_symmetric use_cuda=True 6 2,7 6
bash run_fifo.sh entity_qmix_refil_imagine sc2mt 5-11sz_symmetric use_cuda=True 6 0,1 6

# zj
bash run_fifo.sh entity_vdn_attn sc2mt 5-11sz_symmetric use_cuda=True 6 0,1 6
bash run_exp.sh token_qmix_q_latent_pattern_contrastive_wise_trans_scale sc2 3s_vs_5z use_cuda=True 6 2,3 6
bash run_exp.sh token_qmix_q_latent_pattern_contrastive_wise_trans_scale sc2 MMM2 use_cuda=True 6 1,2,3 6


# 20220105
# zj
bash run_fifo.sh entity_vdn_refil_imagine sc2mt 3-8MMM_symmetric use_cuda=True,t_max=4050000 6 2,3 6
bash run_fifo.sh entity_vdn_refil_imagine sc2mt 3-8sz_symmetric use_cuda=True,t_max=4050000 6 2,3 6
bash run_fifo.sh entity_vdn_refil_imagine sc2mt 3-8csz_symmetric use_cuda=True,t_max=4050000 6 0,1 6
bash run_fifo.sh entity_vdn_refil_imagine sc2mt 5-11MMM_symmetric use_cuda=True,t_max=4050000 6 1,2,3 6
bash run_fifo.sh entity_vdn_refil_imagine sc2mt 5-11sz_symmetric use_cuda=True,t_max=4050000 6 1,2,3 6
bash run_fifo.sh entity_vdn_refil_imagine sc2mt 5-11csz_symmetric use_cuda=True,t_max=4050000 2 0,1 2
bash run_fifo.sh entity_vdn_refil_imagine sc2mt 5-11csz_symmetric use_cuda=True,t_max=4050000 4 0,1,2,3 4

# vipa
bash run_fifo.sh token_vdn_updet sc2 MMM use_cuda=True 2 0,1 6 # 6G
bash run_fifo.sh token_vdn_updet sc2 MMM2 use_cuda=True 2 0,1 6 # 6G

# ----------------------------------------------------------------------------------------------------

# 齐全：3m、5m_vs_6m、3s_vs_5z、3s5z、2s3z、3s_vs_3z、8m_vs_9m、10m_vs_11m
# 3s_vs_3z qmix (6个)
# 2s3z vdn_updet (6个，1不好)

# 25m：iql、vdn、coma、qtran、qmix
# vdn_updet 安排不了 20G

python src/main.py --config=entity_qmix_refil_attn --env-config=sc2mt_test with env_args.map_name=3-8m_symmetric
python src/main.py --config=entity_qmix_refil_imagine --env-config=sc2mt_test with env_args.map_name=3-8m_symmetric
python src/main.py --config=entity_vdn_refil_attn --env-config=sc2mt_test with env_args.map_name=3-8m_symmetric
python src/main.py --config=entity_vdn_refil_imagine --env-config=sc2mt_test with env_args.map_name=3-8m_symmetric
python src/main.py --config=entity_qmix_attn --env-config=sc2mt_test with env_args.map_name=3-8m_symmetric
CUDA_VISIBLE_DEVICES="2" python src/main.py --config=entity_qmix_trans --env-config=sc2mt_grid with env_args.map_name=3-8MMM_symmetric

python src/main.py --config=qmix --env-config=sc2_test with env_args.map_name=3m
python src/main.py --config=vdn --env-config=sc2_test with env_args.map_name=3m
python src/main.py --config=iql --env-config=sc2_test with env_args.map_name=3m
python src/main.py --config=coma --env-config=sc2_test with env_args.map_name=3m
python src/main.py --config=qtran --env-config=sc2_test with env_args.map_name=3m

python src/main.py --config=token_qmix_attn --env-config=sc2_test with env_args.map_name=3s_vs_5z
CUDA_VISIBLE_DEVICES="2" python src/main.py --config=token_qmix_trans --env-config=sc2_test with env_args.map_name=MMM
python src/main.py --config=token_qmix_wise_trans_latent_scale --env-config=particle_test with env_args.map_name=tag_16_16_2
python src/main.py --config=token_vdn_attn --env-config=sc2_test with env_args.map_name=2s3z
python src/main.py --config=token_vdn_trans --env-config=sc2_test with env_args.map_name=3m


bash run_fifo.sh qmix_latent_scale 2s3z 3 use_cuda=True,name=qmix_sparse 0 6
bash run_fifo.sh qmix_latent_scale 3s_vs_5z 3 use_cuda=True,name=qmix_sparse 1 6
bash run_fifo.sh qmix_latent_scale 5m_vs_6m 3 use_cuda=True,name=qmix_sparse 2 6
bash run_fifo.sh qmix_latent_scale 8m_vs_9m 3 use_cuda=True,name=qmix_sparse 3 6

bash run_fifo.sh iql 2c_vs_64zg 3 use_cuda=True 0 6 # pao zj
bash run_fifo.sh qmix 2c_vs_64zg 3 use_cuda=True 1 6 # pao zj
bash run_fifo.sh qtran 2c_vs_64zg 3 use_cuda=True 2 6 # pao zj
bash run_fifo.sh vdn 2c_vs_64zg 3 use_cuda=True 3 6 # pao zj
bash run_fifo.sh coma 2c_vs_64zg 3 use_cuda=True 1,2,3 6 # pao wan
bash run_fifo.sh vdn_updet 2c_vs_64zg 2 use_cuda=True 0,3 4 # pao zj  20 G
bash run_fifo.sh qmix_ext 2c_vs_64zg 4 use_cuda=True 2,3 6 # pao zj
bash run_fifo.sh qmix_ext_scale 2c_vs_64zg 4 use_cuda=True 0,1 6 # pao zj
bash run_fifo.sh qmix_latent 2c_vs_64zg 3 use_cuda=True 0 6 # pao
bash run_fifo.sh qmix_latent_scale 2c_vs_64zg 1 use_cuda=True 1 6 # pao 38 G


bash run.sh vdn 3s_vs_5z 4 use_cuda=True 1,2 8 # zj fang
bash run.sh qmix 3s5z 4 use_cuda=True 0,1,2,3 8 # zj fang

bash run_fifo.sh qmix_latent_scale 3s_vs_5z 3 use_cuda=True 0 6 # pao
bash run_fifo.sh qmix_latent_scale 3s5z 4 use_cuda=True 2,3 8 # pao



bash script-noseed/5m_to_3m_qmix_latent_scale.sh 1 2 0 # pao wan

bash script-noseed/10m_to_8m_qmix_latent_scale.sh 1 2 3 # pao wan

bash script-noseed/3m_to_8m_qmix_latent_scale.sh 1 1 3 # pao wan
bash script-noseed/3m_to_8m_qmix_latent_scale.sh 1 0 3 # pao wan

bash script-noseed/2s3z_to_3s5z_qmix_latent_scale.sh 1 0 4 # pao wan
bash script-noseed/2s3z_to_3s5z_qmix_latent_scale.sh 2 0 4,2 # pao wan
bash script-noseed/2s3z_to_3s5z_qmix_latent_scale.sh 1 1 4 # pao wan

bash script-noseed/3s_vs_3z_to_3s_vs_5z_qmix_latent_scale.sh 5 0,1,2,3 0,1,2,3,4  # pao wan zj fang

bash 3s5z_to_3s_vs_5z_qmix_latent.sh 1 1 1 # pao wan
bash 3m_to_10m_qmix_latent_scale.sh 2 1,2 1,3 # pao wan zj fang
bash 3s5z_to_3s_vs_5z_qmix_latent_scale.sh 5 0,1,2,3 0,1,2,3,4 # pao wan

bash 5m_to_8m_qmix_latent_scale.sh 1 2 4 # pao wan


 ########
bash run.sh qmix_latent_scale 3s_vs_5z 3 use_cuda=True 0 5 # zj fanghuiqu
bash run.sh qmix 3s_vs_5z 3 use_cuda=True 1 3 # zj fanghuiqu
bash run.sh vdn 3s_vs_5z 3 use_cuda=True 1 3 # zj fanghuiqu

bash run.sh qmix 3s5z 3 use_cuda=True 2 3 #  zj fanghuiqu
bash run.sh qmix_latent_scale 3s5z 3 use_cuda=True 3 3 #  zj fanghuiqu



bash 3m_to_10m_qmix_ext_scale.sh 1 1 0  # pao
CUDA_VISIBLE_DEVICES="2" python src/main.py --config=qmix_ext_scale --env-config=sc2 with t_max=4050000 seed=948538477 name=qmix_ext_scale_3m env_args.map_name=10m_vs_11m checkpoint_path="$NFS_HOME/code/pymarl/results/exp_v2/sc2/3m/qmix_ext_scale/2021-09-12_19-04-08/models"


bash 2s3z_to_3s5z_qmix_latent.sh 2 0,2 0,1,2,3,4 # pao

bash 3s5z_to_3s_vs_5z_qmix_latent.sh 3 0 0,1,2,3,4 # pao







bash 2s3z_to_3s5z_qmix_latent_scale.sh 2 0,1 0,1,2,3,4 # pao

bash 3m_to_10m_qmix_latent.sh 2 0,1 0,1,2,3,4 # pao


bash 8m_to_10m_qmix_latent.sh 1 1 0,1,2,3,4  # pao dai fang
bash 8m_to_10m_qmix_latent_scale.sh 2 0,1 0,1,2,3,4  # pao dai fang

bash 3m_to_5m_qmix_latent.sh 2 0,1 0,1,2,3,4 # pao
bash 3m_to_5m_qmix_latent_scale.sh 2 3 0,1,2,3,4 # pao





bash 3s_vs_3z_to_3s_vs_5z_qmix_latent.sh 3 0 0,1,2,3,4 # pao zj fang
bash 3s_vs_3z_to_3s_vs_5z_qmix_latent_scale.sh 3 0 0,1,2,3,4 # pao

bash 3s_vs_5z_to_3s5z_qmix_latent.sh 3 1 0,1,2,3,4 # pao zj fang
bash 3s_vs_5z_to_3s5z_qmix_latent_scale.sh 3 2 0,1,2,3,4 # pao zj fang

bash 5m_to_3m_qmix_latent.sh 3 3 0,1,2,3,4 # pao zj fang
bash 5m_to_3m_qmix_latent_scale.sh 3 3 0,1,2,3,4 # pao zj fang

bash 10m_to_3m_qmix_latent.sh 3 1 0,1,2,3,4  # pao zj fang
bash 10m_to_3m_qmix_latent_scale.sh 3 0 0,1,2,3,4  # pao zj fang

bash 10m_to_8m_qmix_latent.sh 3 3 0,1,2,3,4 # pao zj fang
bash 10m_to_8m_qmix_latent_scale.sh 3 2 0,1,2,3,4 # pao zj fang

bash 3m_to_8m.sh 3 0 0,1,2,3,4 # pao
bash 3m_to_8m_qmix_ext.sh 3 1 0,1,2,3,4 # pao
bash 3m_to_8m_qmix_ext_scale.sh 3 3 0,1,2,3,4 # pao
bash 3m_to_8m_qmix_latent.sh 3 2 0,1,2,3,4 # pao
bash 3m_to_8m_qmix_latent_scale.sh 4 0,1 0,1,2,3,4 # pao


bash 5m_to_8m.sh 3 2 0,1,2,3,4 # pao
bash 5m_to_8m_qmix_ext.sh 3 3 0,1,2,3,4 # pao
bash 5m_to_8m_qmix_ext_scale.sh 3 3 0,1,2,3,4 # pao
bash 5m_to_8m_qmix_latent.sh 3 1 0,1,2,3,4 # pao
bash 5m_to_8m_qmix_latent_scale.sh 3 0 0,1,2,3,4 # pao




###############

bash run.sh iql 3m 1 use_cuda=True 0 5
bash run.sh qmix 3m 1 use_cuda=True 0 5
bash run.sh qtran 3m 1 use_cuda=True 0 5
bash run.sh vdn 3m 1 use_cuda=True 0 5
bash run.sh coma 3m 1 use_cuda=True 0 5
bash run.sh vdn_updet 3m 1 use_cuda=True 0,1 5

bash run.sh vdn_updet 25m 1 use_cuda=True,batch_size=16 0 5
bash run.sh qmix_latent_scale 25m 1 use_cuda=True,batch_size=16 1 5

CUDA_VISIBLE_DEVICES="0" python src/main.py --config=qmix --env-config=particle with env_args.map_name=spread_8 buffer_size=1000

# 全跑 zjlab
bash run_particle.sh vdn htag_16_8_2 3 use_cuda=True 0 6
bash run_particle.sh qmix htag_16_8_2 3 use_cuda=True 0 6
bash run_particle.sh vdn_updet htag_16_8_2 3 use_cuda=True 1 6
bash run_particle.sh qmix_ext htag_16_8_2 3 use_cuda=True 2 6
bash run_particle.sh qmix_ext_scale htag_16_8_2 3 use_cuda=True 3 6
bash run_particle.sh qmix_latent htag_16_8_2 3 use_cuda=True 2 6
bash run_particle.sh qmix_latent_scale htag_16_8_2 3 use_cuda=True 3 6

bash run_particle.sh vdn htag_8_4_2 3 use_cuda=True 0 6
bash run_particle.sh qmix htag_8_4_2 3 use_cuda=True 0 6
bash run_particle.sh vdn_updet htag_8_4_2 3 use_cuda=True 1 6
bash run_particle.sh qmix_ext htag_8_4_2 3 use_cuda=True 2 6
bash run_particle.sh qmix_ext_scale htag_8_4_2 3 use_cuda=True 3 6
bash run_particle.sh qmix_latent htag_8_4_2 3 use_cuda=True 2 6 # 跑
bash run_particle.sh qmix_latent_scale htag_8_4_2 3 use_cuda=True 3 6 # 跑

bash run_particle.sh vdn tag_16_16_2 2 use_cuda=True 0 5
bash run_particle.sh qmix tag_16_16_2 2 use_cuda=True 0,1 5
bash run_particle.sh vdn_updet tag_16_16_2 3 use_cuda=True 1 5
bash run_particle.sh qmix_ext tag_16_16_2 3 use_cuda=True 1 5
bash run_particle.sh qmix_ext_scale tag_16_16_2 3 use_cuda=True 2 5
bash run_particle.sh qmix_latent tag_16_16_2 3 use_cuda=True 2 5
bash run_particle.sh qmix_latent_scale tag_16_16_2 3 use_cuda=True 3 5

bash run_particle.sh vdn tag_8_8_2 3 use_cuda=True 2 6 # 跑
bash run_particle.sh qmix tag_8_8_2 3 use_cuda=True 2 6 # 跑
bash run_particle.sh vdn_updet tag_8_8_2 3 use_cuda=True 1 6 # 跑
bash run_particle.sh qmix_ext tag_8_8_2 3 use_cuda=True 2 6 # 跑
bash run_particle.sh qmix_ext_scale tag_8_8_2 3 use_cuda=True 2 6  # 跑
bash run_particle.sh qmix_latent tag_8_8_2 3 use_cuda=True 0 6 # 跑dai放回去了
bash run_particle.sh qmix_latent_scale tag_8_8_2 3 use_cuda=True 1 6 # 跑dai放回去了

bash run_particle.sh vdn tag_4_4_2 3 use_cuda=True 2 6 # 跑
bash run_particle.sh qmix tag_4_4_2 3 use_cuda=True 2 6 # 跑
bash run_particle.sh vdn_updet tag_4_4_2 3 use_cuda=True 1 6 # 跑
bash run_particle.sh qmix_ext tag_4_4_2 3 use_cuda=True 0 6 # 跑
bash run_particle.sh qmix_ext_scale tag_4_4_2 3 use_cuda=True 1 6 # 跑
bash run_particle.sh qmix_latent tag_4_4_2 3 use_cuda=True 0 6 # 跑
bash run_particle.sh qmix_latent_scale tag_4_4_2 3 use_cuda=True 0,1 6  # 跑

bash run.sh qmix_latent 5m_vs_6m 2 use_cuda=True 0 5 # 跑完
bash run.sh qmix_latent 5m_vs_6m 2 use_cuda=True 0 2 # 跑完

bash run.sh qmix_latent_scale 5m_vs_6m 2 use_cuda=True 1 2 # 跑完


bash run.sh qmix_latent 3m 2 use_cuda=True 0 1 # 跑
bash run.sh qmix_latent_scale 3m 1 use_cuda=True 0 1 # 跑

bash run.sh qmix_latent 8m_vs_9m 1 use_cuda=True 1 1 # 跑
bash run.sh qmix_latent_scale 8m_vs_9m 2 use_cuda=True 0 5 # 跑

bash run.sh qmix_latent 10m_vs_11m 2 use_cuda=True 0,2 4 # 跑
bash run.sh qmix_latent_scale 10m_vs_11m 2 use_cuda=True 0 5 # 跑

bash run.sh qmix_latent 3s5z 2 use_cuda=True 2 4 # 跑
bash run.sh qmix_latent_scale 3s5z 2 use_cuda=True 0,2 2 # 跑

bash run.sh qmix_latent 2s3z 2 use_cuda=True 0 2 # 跑
bash run.sh qmix_latent_scale 2s3z 2 use_cuda=True 0 5 # 跑


bash run.sh qmix_latent 3s_vs_3z 2 use_cuda=True 2 5 # 跑
bash run.sh qmix_latent_scale 3s_vs_3z 2 use_cuda=True 0,1 5 # 跑

bash run.sh qmix_latent 3s_vs_5z 2 use_cuda=True 0 5 # 跑
bash run.sh qmix_latent_scale 3s_vs_5z 2 use_cuda=True 1 4 # 跑


bash run.sh qmix_ext_scale 3m 5 use_cuda=True 3 5  # 跑完
bash run.sh qmix_ext_scale 5m_vs_6m 2 use_cuda=True 1 5  # 跑完
bash run.sh qmix_ext_scale 2s3z 3 use_cuda=True 1 5  # 跑完
bash run.sh qmix_ext_scale 3s5z 2 use_cuda=True 1 5  # 跑完
bash run.sh qmix_ext_scale 3s_vs_3z 3 use_cuda=True 3 5   # 跑完
bash run.sh qmix_ext_scale 3s_vs_5z 3 use_cuda=True 2 5  # 跑完

# dai
bash run.sh qmix_ext_scale 8m_vs_9m 2 use_cuda=True 0 5 # (9个 1257不好) 放回去了
bash run.sh qmix_ext_scale 10m_vs_11m 1 use_cuda=True 1 2 # (6个 4效果不好) 放回去了
bash 10m_to_3m_qmix_ext_scale.sh 5 0 0,1,2,3,4 # 放回去了
bash 10m_to_8m_qmix_ext_scale.sh 3 1 0,1,2,3,4 # 放回去了


# 跑完
bash run.sh qmix_ext_single 5m_vs_6m 2 use_cuda=True 1,3 5


# 跑完
bash run.sh qmix_ext 3m 5 use_cuda=True 1 5  # 跑完

# 跑完9个（234效果不好）
bash run.sh qmix_ext 10m_vs_11m 1 use_cuda=True 2 3  # 110
bash run.sh qmix_ext 10m_vs_11m 1 use_cuda=True 0 1


bash 10m_to_3m_qmix_ext.sh 5 0 0,1,2,3,4
bash 10m_to_8m_qmix_ext.sh 2 0 0,1 # pao
bash 10m_to_8m_qmix_ext.sh 2 0 2,3 # pao
bash 10m_to_8m_qmix_ext.sh 1 1 4 # pao


# 跑完6个（1效果不好）
bash run.sh qmix_ext 8m_vs_9m 2 use_cuda=True 0 5
bash run.sh qmix_ext 8m_vs_9m 1 use_cuda=True 1 1

# 跑完7个 (2效果不好)
bash run.sh qmix_ext 5m_vs_6m 2 use_cuda=True 1 5
bash run.sh qmix_ext 5m_vs_6m 2 use_cuda=True 3 2

# 跑完
bash run.sh qmix_ext 3s_vs_3z 2 use_cuda=True 0 5

# 跑完
bash run.sh qmix_ext 3s_vs_5z 2 use_cuda=True 0,1 5
bash run.sh qmix_ext 3s_vs_5z 2 use_cuda=True 0 2

# 跑完（4效果不好，但没跑)
bash run.sh qmix_ext 2s3z 2 use_cuda=True 0,1 5
bash run.sh qmix_ext 2s3z 1 use_cuda=True 1 1

# 跑完 8个 (2 4效果不好)
bash run.sh qmix_ext 3s5z 2 use_cuda=True 0 5
bash run.sh qmix_ext 3s5z 2 use_cuda=True 2 2


# 跑完
bash 3m_to_5m_qmix_ext.sh 2 0 0,1,2,3,4
bash 3m_to_5m_qmix_ext_scale.sh 2 1 0,1,2,3,4

# 跑完
bash 5m_to_3m_qmix_ext.sh 3 0 0,1,2,3,4

# 跑完
bash 5m_to_3m_qmix_ext_scale.sh 2 1 3,4
bash 5m_to_3m_qmix_ext_scale.sh 3 0 0,1,2

# 5个跑完
bash 2s3z_to_3s5z_qmix_ext.sh 2 2 0,1,2,3,4

# 5个跑完
bash 2s3z_to_3s5z_qmix_ext_scale.sh 2 3 0,1,2,3,4
bash 2s3z_to_3s5z_qmix_ext_scale.sh 1 3 4

# 5个跑完
bash 3s5z_to_3s_vs_5z_qmix_ext_scale.sh 3 0 0,1,2,3,4
bash 3s5z_to_3s_vs_5z_qmix_ext_scale.sh 1 0 0

# 5个跑完
bash 3s5z_to_3s_vs_5z_qmix_ext.sh 3 2 0,1,2,3,4

# zjlab
bash 3m_to_10m_qmix_ext.sh 2 0 0,1,2,3,4
bash 3m_to_10m_qmix_ext_scale.sh 2 1 0,1,2,3,4

# 跑完 放回去了
bash 3s_vs_3z_to_3s_vs_5z_qmix_ext.sh 5 2 0,1,2,3,4
bash 3s_vs_3z_to_3s_vs_5z_qmix_ext_scale.sh 5 3 0,1,2,3,4

# 跑完 放回去了
bash 3s_vs_5z_to_3s5z_qmix_ext.sh 3 0 0,1,2,3,4
bash 3s_vs_5z_to_3s5z_qmix_ext_scale.sh 3 1 0,1,2,3,4

bash 8m_to_10m_qmix_ext.sh 2 0 0,1,2,3
bash 8m_to_10m_qmix_ext.sh 1 2 4 # 210跑

bash 8m_to_10m_qmix_ext_scale.sh 2 1 0,1,2,3,4




CUDA_VISIBLE_DEVICES="0" python src/main.py --config=qmix_ext --env-config=sc2 with env_args.map_name=5m_vs_6m

CUDA_VISIBLE_DEVICES="0" python src/main.py --config=qmix --env-config=particle with env_args.map_name=spread_5_5
CUDA_VISIBLE_DEVICES="0" python src/main.py --config=qmix --env-config=particle with env_args.map_name=tag_5_5_2
python src/main.py --config=qmix --env-config=particle with env_args.map_name=spread_5_5 checkpoint_path="/nfs4-p1/lsy/code/pymarl/results/exp_v2/particle/spread_5_5/qmix/2021-09-17_13-37-27/models" evaluate=True render=True
python src/main.py --config=qmix_ext_scale --env-config=particle with env_args.map_name=tag_5_5_2 checkpoint_path="/nfs4-p1/lsy/code/pymarl/results/exp_v2/particle/tag_5_5_2/qmix_ext_scale/2021-09-17_21-25-35/models" evaluate=True render=True


# 齐全：3m、5m_vs_6m、3s_vs_5z、3s5z、2s3z、3s_vs_3z、8m_vs_9m、10m_vs_11m
# 2s3z updet 第一个不好用
# 3s_vs_3z qmix 有六个


# 25m：iql、vdn、coma、qtran、qmix
# 安排不了 20G
bash run.sh vdn_updet 25m 2 use_cuda=True 0 5



# 3m to 5m 跑完
bash 3m_to_5m.sh 3 0 0,1,2,3,4
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=28726148 name=vdn_updet_3m env_args.map_name=5m_vs_6m checkpoint_path="/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/3m/vdn_updet/2021-08-20_23-44-42/models"
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=447599472 name=vdn_updet_3m env_args.map_name=5m_vs_6m checkpoint_path="/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/3m/vdn_updet/2021-08-20_23-46-41/models"
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=858219515 name=vdn_updet_3m env_args.map_name=5m_vs_6m checkpoint_path="/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/3m/vdn_updet/2021-08-20_23-48-38/models"
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=245961003 name=vdn_updet_3m env_args.map_name=5m_vs_6m checkpoint_path="/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/3m/vdn_updet/2021-08-22_04-43-39/models"
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=277897023 name=vdn_updet_3m env_args.map_name=5m_vs_6m checkpoint_path="/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/3m/vdn_updet/2021-08-22_04-45-02/models"

# 5m to 3m  跑完
bash 5m_to_3m.sh 3 1,0 0,1,2,3,4
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=549505994 name=vdn_updet_5m_vs_6m env_args.map_name=3m checkpoint_path="/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/5m_vs_6m/vdn_updet/2021-08-20_21-56-53/models"
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=662232093 name=vdn_updet_5m_vs_6m env_args.map_name=3m checkpoint_path="/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/5m_vs_6m/vdn_updet/2021-08-20_21-58-04/models"
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=609448473 name=vdn_updet_5m_vs_6m env_args.map_name=3m checkpoint_path="/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/5m_vs_6m/vdn_updet/2021-08-20_21-59-43/models"
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=360194117 name=vdn_updet_5m_vs_6m env_args.map_name=3m checkpoint_path="/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/5m_vs_6m/vdn_updet/2021-08-22_01-18-34/models"
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=810530385 name=vdn_updet_5m_vs_6m env_args.map_name=3m checkpoint_path="/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/5m_vs_6m/vdn_updet/2021-08-22_01-20-01/models"

# 8m to 10m 跑完
bash 8m_to_10m.sh 1 0 0,1,2
bash 8m_to_10m.sh 1 0 3,4
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=884415179 name=vdn_updet_8m_vs_9m env_args.map_name=10m_vs_11m checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/8m_vs_9m/vdn_updet/2021-08-20_21-57-44/models\"
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=83883218 name=vdn_updet_8m_vs_9m env_args.map_name=10m_vs_11m checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/8m_vs_9m/vdn_updet/2021-08-22_05-13-58/models\"
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=21843385 name=vdn_updet_8m_vs_9m env_args.map_name=10m_vs_11m checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/8m_vs_9m/vdn_updet/2021-08-22_05-15-27/models\"
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=532323478 name=vdn_updet_8m_vs_9m env_args.map_name=10m_vs_11m checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/8m_vs_9m/vdn_updet/2021-08-23_09-53-06/models\"
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=879165945 name=vdn_updet_8m_vs_9m env_args.map_name=10m_vs_11m checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/8m_vs_9m/vdn_updet/2021-08-25_12-42-56/models\"

# 10m to 8m 跑完
bash 10m_to_8m.sh 2 1 0,1,2,3,4
bash 10m_to_8m.sh 1 1 1
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=1872341 name=vdn_updet_10m_vs_11m env_args.map_name=8m_vs_9m checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/10m_vs_11m/vdn_updet/2021-08-24_14-23-38/models\"
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=35015916 name=vdn_updet_10m_vs_11m env_args.map_name=8m_vs_9m checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/10m_vs_11m/vdn_updet/2021-08-24_14-24-14/models\"
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=966424768 name=vdn_updet_10m_vs_11m env_args.map_name=8m_vs_9m checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/10m_vs_11m/vdn_updet/2021-08-24_14-24-59/models\"
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=435750059 name=vdn_updet_10m_vs_11m env_args.map_name=8m_vs_9m checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/10m_vs_11m/vdn_updet/2021-08-24_14-25-24/models\"
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=936524720 name=vdn_updet_10m_vs_11m env_args.map_name=8m_vs_9m checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/10m_vs_11m/vdn_updet/2021-08-24_18-47-56/models\"

# 3m to 10m  跑完
bash 3m_to_10m.sh 1 2 0,1,2,3,4
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=28726148 name=vdn_updet_3m env_args.map_name=10m_vs_11m checkpoint_path="/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/3m/vdn_updet/2021-08-20_23-44-42/models"
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=447599472 name=vdn_updet_3m env_args.map_name=10m_vs_11m checkpoint_path="/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/3m/vdn_updet/2021-08-20_23-46-41/models"
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=858219515 name=vdn_updet_3m env_args.map_name=10m_vs_11m checkpoint_path="/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/3m/vdn_updet/2021-08-20_23-48-38/models"
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=245961003 name=vdn_updet_3m env_args.map_name=10m_vs_11m checkpoint_path="/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/3m/vdn_updet/2021-08-22_04-43-39/models"
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=277897023 name=vdn_updet_3m env_args.map_name=10m_vs_11m checkpoint_path="/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/3m/vdn_updet/2021-08-22_04-45-02/models"


# 10m to 3m 跑完
bash 10m_to_3m.sh 3 0 0,1,2,3,4
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=1872341 name=vdn_updet_10m_vs_11m env_args.map_name=3m checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/10m_vs_11m/vdn_updet/2021-08-24_14-23-38/models\"
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=35015916 name=vdn_updet_10m_vs_11m env_args.map_name=3m checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/10m_vs_11m/vdn_updet/2021-08-24_14-24-14/models\"
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=966424768 name=vdn_updet_10m_vs_11m env_args.map_name=3m checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/10m_vs_11m/vdn_updet/2021-08-24_14-24-59/models\"
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=435750059 name=vdn_updet_10m_vs_11m env_args.map_name=3m checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/10m_vs_11m/vdn_updet/2021-08-24_14-25-24/models\"
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=936524720 name=vdn_updet_10m_vs_11m env_args.map_name=3m checkpoint_path=\"/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/10m_vs_11m/vdn_updet/2021-08-24_18-47-56/models\"


# 2s3z to 3s5z 跑完
bash 2s3z_to_3s5z.sh 3 1 0,1,2,3,4
bash 2s3z_to_3s5z.sh 1 0 2
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=557451342 name=vdn_updet_2s3z env_args.map_name=3s5z checkpoint_path="/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/2s3z/vdn_updet/2021-08-22_19-25-02/models"
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=605247556 name=vdn_updet_2s3z env_args.map_name=3s5z checkpoint_path="/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/2s3z/vdn_updet/2021-08-22_19-26-37/models"
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=990739171 name=vdn_updet_2s3z env_args.map_name=3s5z checkpoint_path="/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/2s3z/vdn_updet/2021-08-24_21-39-37/models"
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=550663257 name=vdn_updet_2s3z env_args.map_name=3s5z checkpoint_path="/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/2s3z/vdn_updet/2021-08-24_21-40-59/models"
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=814750593 name=vdn_updet_2s3z env_args.map_name=3s5z checkpoint_path="/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/2s3z/vdn_updet/2021-08-25_22-58-30/models"

# 3s_vs_3z to 3s_vs_5z  跑完
bash 3s_vs_3z_to_3s_vs_5z.sh 3 0,1 0,1,2,3,4
bash 3s_vs_3z_to_3s_vs_5z.sh 1 0 0
bash 3s_vs_3z_to_3s_vs_5z.sh 1 0 4
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=143651867 name=vdn_updet_3s_vs_3z env_args.map_name=3s_vs_5z checkpoint_path="/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/3s_vs_3z/vdn_updet/2021-08-23_10-56-14/models"
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=980048948 name=vdn_updet_3s_vs_3z env_args.map_name=3s_vs_5z checkpoint_path="/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/3s_vs_3z/vdn_updet/2021-08-23_10-57-42/models"
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=261189610 name=vdn_updet_3s_vs_3z env_args.map_name=3s_vs_5z checkpoint_path="/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/3s_vs_3z/vdn_updet/2021-08-23_10-59-35/models"
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=562405198 name=vdn_updet_3s_vs_3z env_args.map_name=3s_vs_5z checkpoint_path="/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/3s_vs_3z/vdn_updet/2021-08-24_12-38-13/models"
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=193987767 name=vdn_updet_3s_vs_3z env_args.map_name=3s_vs_5z checkpoint_path="/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/3s_vs_3z/vdn_updet/2021-08-24_12-39-30/models"

# 3s5z to 3s_vs_5z 跑完
bash 3s5z_to_3s_vs_5z.sh 3 0,1 0,1,2,3,4
bash 3s5z_to_3s_vs_5z.sh 4 0,1 0,1,2,3
bash 3s5z_to_3s_vs_5z.sh 1 0 2
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=763736170 name=vdn_updet_3s5z env_args.map_name=3s_vs_5z checkpoint_path="/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/3s5z/vdn_updet/2021-08-22_19-29-25/models"
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=690386827 name=vdn_updet_3s5z env_args.map_name=3s_vs_5z checkpoint_path="/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/3s5z/vdn_updet/2021-08-22_19-30-41/models"
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=80098928 name=vdn_updet_3s5z env_args.map_name=3s_vs_5z checkpoint_path="/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/3s5z/vdn_updet/2021-08-23_18-42-19/models"
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=383634973 name=vdn_updet_3s5z env_args.map_name=3s_vs_5z checkpoint_path="/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/3s5z/vdn_updet/2021-08-23_18-44-06/models"
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=187644705 name=vdn_updet_3s5z env_args.map_name=3s_vs_5z checkpoint_path="/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/3s5z/vdn_updet/2021-08-24_21-52-30/models"

# 3s_vs_5z to 3s5z  跑完
bash 3s_vs_5z_to_3s5z.sh 3 0 0,1,2,3,4
bash 3s_vs_5z_to_3s5z.sh 1 0 2
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=333127656 name=vdn_updet_3s_vs_5z env_args.map_name=3s5z checkpoint_path="/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/3s_vs_5z/vdn_updet/2021-08-23_10-52-41/models"
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=367990614 name=vdn_updet_3s_vs_5z env_args.map_name=3s5z checkpoint_path="/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/3s_vs_5z/vdn_updet/2021-08-23_10-54-25/models"
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=136427816 name=vdn_updet_3s_vs_5z env_args.map_name=3s5z checkpoint_path="/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/3s_vs_5z/vdn_updet/2021-08-23_10-56-15/models"
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=176528899 name=vdn_updet_3s_vs_5z env_args.map_name=3s5z checkpoint_path="/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/3s_vs_5z/vdn_updet/2021-08-24_03-44-54/models"
python src/main.py --config=vdn_updet --env-config=sc2 with t_max=4050000 seed=952454574 name=vdn_updet_3s_vs_5z env_args.map_name=3s5z checkpoint_path="/nfs4-p1/lsy/code/pymarl/results/exp_v2/sc2/3s_vs_5z/vdn_updet/2021-08-24_03-46-06/models"



python src/main.py --config=iql --env-config=sc2 with env_args.map_name=5m_vs_6m
python src/main.py --config=qmix --env-config=sc2 with env_args.map_name=5m_vs_6m
python src/main.py --config=qtran --env-config=sc2 with env_args.map_name=5m_vs_6m
python src/main.py --config=vdn --env-config=sc2 with env_args.map_name=5m_vs_6m
python src/main.py --config=coma --env-config=sc2 with env_args.map_name=5m_vs_6m
python src/main.py --config=qmix_updet --env-config=sc2 with env_args.map_name=5m_vs_6m ally_num=5 enemy_num=6
python src/main.py --config=qmix_dueling --env-config=sc2 with env_args.map_name=5m_vs_6m

python src/main.py --config=iql --env-config=sc2 with env_args.map_name=3m
python src/main.py --config=qmix --env-config=sc2 with env_args.map_name=3m
python src/main.py --config=qtran --env-config=sc2 with env_args.map_name=3m
python src/main.py --config=vdn --env-config=sc2 with env_args.map_name=3m
python src/main.py --config=coma --env-config=sc2 with env_args.map_name=3m
python src/main.py --config=qmix_updet --env-config=sc2 with env_args.map_name=3m ally_num=3 enemy_num=3
python src/main.py --config=qmix_dueling --env-config=sc2 with env_args.map_name=3m

python src/main.py --config=iql --env-config=sc2 with env_args.map_name=8m_vs_9m
python src/main.py --config=qmix --env-config=sc2 with env_args.map_name=8m_vs_9m
python src/main.py --config=qtran --env-config=sc2 with env_args.map_name=8m_vs_9m
python src/main.py --config=vdn --env-config=sc2 with env_args.map_name=8m_vs_9m
python src/main.py --config=coma --env-config=sc2 with env_args.map_name=8m_vs_9m
python src/main.py --config=qmix_updet --env-config=sc2 with env_args.map_name=8m_vs_9m ally_num=8 enemy_num=9
python src/main.py --config=qmix_dueling --env-config=sc2 with env_args.map_name=8m_vs_9m

python src/main.py --config=iql --env-config=sc2 with env_args.map_name=10m_vs_11m
python src/main.py --config=qmix --env-config=sc2 with env_args.map_name=10m_vs_11m
python src/main.py --config=qtran --env-config=sc2 with env_args.map_name=10m_vs_11m
python src/main.py --config=vdn --env-config=sc2 with env_args.map_name=10m_vs_11m
python src/main.py --config=coma --env-config=sc2 with env_args.map_name=10m_vs_11m
python src/main.py --config=qmix_updet --env-config=sc2 with env_args.map_name=10m_vs_11m ally_num=10 enemy_num=11
python src/main.py --config=qmix_dueling --env-config=sc2 with env_args.map_name=10m_vs_11m

python src/main.py --config=iql --env-config=sc2_grid with env_args.map_name=5m_vs_6m
python src/main.py --config=qmix --env-config=sc2_grid with env_args.map_name=5m_vs_6m
python src/main.py --config=qtran --env-config=sc2_grid with env_args.map_name=5m_vs_6m
python src/main.py --config=vdn --env-config=sc2_grid with env_args.map_name=5m_vs_6m
python src/main.py --config=coma --env-config=sc2_grid with env_args.map_name=5m_vs_6m
python src/main.py --config=qmix_updet --env-config=sc2_grid with env_args.map_name=5m_vs_6m
python src/main.py --config=qmix_dueling --env-config=sc2_grid with env_args.map_name=5m_vs_6m

python src/main.py --config=iql --env-config=sc2_grid with env_args.map_name=8m_vs_9m
python src/main.py --config=qmix --env-config=sc2_grid with env_args.map_name=8m_vs_9m
python src/main.py --config=qtran --env-config=sc2_grid with env_args.map_name=8m_vs_9m
python src/main.py --config=vdn --env-config=sc2_grid with env_args.map_name=8m_vs_9m
python src/main.py --config=coma --env-config=sc2_grid with env_args.map_name=8m_vs_9m
python src/main.py --config=qmix_updet --env-config=sc2_grid with env_args.map_name=8m_vs_9m
python src/main.py --config=qmix_dueling --env-config=sc2_grid with env_args.map_name=8m_vs_9m

python src/main.py --config=iql --env-config=sc2_grid with env_args.map_name=10m_vs_11m
python src/main.py --config=qmix --env-config=sc2_grid with env_args.map_name=10m_vs_11m
python src/main.py --config=qtran --env-config=sc2_grid with env_args.map_name=10m_vs_11m
python src/main.py --config=vdn --env-config=sc2_grid with env_args.map_name=10m_vs_11m
python src/main.py --config=coma --env-config=sc2_grid with env_args.map_name=10m_vs_11m
python src/main.py --config=qmix_updet --env-config=sc2_grid with env_args.map_name=10m_vs_11m
python src/main.py --config=qmix_dueling --env-config=sc2_grid with env_args.map_name=10m_vs_11m

python src/main.py --config=iql --env-config=sc2_grid with env_args.map_name=2s3z
python src/main.py --config=qmix --env-config=sc2_grid with env_args.map_name=2s3z
python src/main.py --config=qtran --env-config=sc2_grid with env_args.map_name=2s3z
python src/main.py --config=vdn --env-config=sc2_grid with env_args.map_name=2s3z
python src/main.py --config=coma --env-config=sc2_grid with env_args.map_name=2s3z
python src/main.py --config=qmix_updet --env-config=sc2_grid with env_args.map_name=2s3z
python src/main.py --config=qmix_dueling --env-config=sc2_grid with env_args.map_name=2s3z

python src/main.py --config=iql --env-config=sc2_grid with env_args.map_name=3s5z
python src/main.py --config=qmix --env-config=sc2_grid with env_args.map_name=3s5z
python src/main.py --config=qtran --env-config=sc2_grid with env_args.map_name=3s5z
python src/main.py --config=vdn --env-config=sc2_grid with env_args.map_name=3s5z
python src/main.py --config=coma --env-config=sc2_grid with env_args.map_name=3s5z
python src/main.py --config=qmix_updet --env-config=sc2_grid with env_args.map_name=3s5z
python src/main.py --config=qmix_dueling --env-config=sc2_grid with env_args.map_name=3s5z

python src/main.py --config=iql --env-config=sc2_grid with env_args.map_name=3m
python src/main.py --config=qmix --env-config=sc2_grid with env_args.map_name=3m
python src/main.py --config=qtran --env-config=sc2_grid with env_args.map_name=3m
python src/main.py --config=vdn --env-config=sc2_grid with env_args.map_name=3m
python src/main.py --config=coma --env-config=sc2_grid with env_args.map_name=3m
python src/main.py --config=qmix_updet --env-config=sc2_grid with env_args.map_name=3m
python src/main.py --config=qmix_dueling --env-config=sc2_grid with env_args.map_name=3m



python src/main.py --config=qmix --env-config=sc2_grid with env_args.map_name=5m_vs_6m checkpoint_path="/Users/liushunyu/Desktop/code/pymarl/results/exp_v2/sc2_grid/5m_vs_6m/qmix/2021-08-14_11-13-10/models" save_replay=True
python src/main.py --config=vdn --env-config=sc2_grid with env_args.map_name=5m_vs_6m checkpoint_path="/Users/liushunyu/Desktop/code/pymarl/results/exp_v2/sc2_grid/5m_vs_6m/vdn/2021-08-14_11-14-03/models" save_replay=True
python src/main.py --config=vdn --env-config=sc2_grid with env_args.map_name=5m_vs_6m checkpoint_path="/Users/liushunyu/Desktop/code/pymarl/results/exp_v2/sc2_grid/5m_vs_6m/vdn/2021-08-14_11-14-03/models" evaluate=True render=True

python src/main.py --config=qmix --env-config=particle with env_args.map_name=spread_5_5 checkpoint_path="/nfs4-p1/lsy/code/pymarl/results/exp_v2/particle/spread_5_5/qmix/2021-09-17_13-37-27/models" evaluate=True render=True

# https://github.com/deepmind/pysc2/issues/288
python -m pysc2.bin.play --norender --rgb_minimap_size 0 --replay 5m_vs_6m_2021-08-18-05-51-14.SC2Replay

Name            Agents  Enemies Limit
3m              3       3       60
8m              8       8       120
25m             25      25      150
5m_vs_6m        5       6       70
8m_vs_9m        8       9       120
10m_vs_11m      10      11      150
27m_vs_30m      27      30      180
MMM             10      10      150
MMM2            10      12      180
2s3z            5       5       120
3s5z            8       8       150
3s5z_vs_3s6z    8       9       170
3s_vs_3z        3       3       150
3s_vs_4z        3       4       200
3s_vs_5z        3       5       250
1c3s5z          9       9       180
2m_vs_1z        2       1       150
corridor        6       24      400
6h_vs_8z        6       8       150
2s_vs_1sc       2       1       300
so_many_baneling 7       32      100
bane_vs_bane    24      24      200
2c_vs_64zg      2       64      400



python3 src/main.py --config=qmix --env-config=particle with env_args.map_name=simple_tag_coop

