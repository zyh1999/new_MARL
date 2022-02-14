import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os
import json
import numpy as np
import pandas as pd
sns.set()
sns.set_style("darkgrid", {"axes.facecolor": "#f0f0f7"})
linestyle = ['-', '--', ':', '-.']
fontsize = 20

EXP_PATH = os.path.join(os.environ['NFS_HOME'], 'code/pymarl')

total_timesteps = {'sc2': 2000000, 'sc2mt': 10000000, 'particle': 1000000}


def smooth(data, window=20):
    y = np.ones(window)
    for idx in range(len(data)):
        x = np.asarray(data[idx])
        z = np.ones(len(x))
        smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
        data[idx] = smoothed_x

    return data


def json_to_list(data_json):
    data_list = []
    for data in data_json:
        data_list.append(data['value'])
    return data_list


def check_original_data(env_name, map_list, algo_list, seed_idx_list):
    original_data = dict()
    result_dir = os.path.join(EXP_PATH, 'results', 'exp_v2', env_name)

    error_algo_path = []
    error_state_path = []
    error_data_path = []
    error_timestep_path = []

    for map_name in map_list:
        original_data[map_name] = dict()
        for algo_id in algo_list[map_name]:
            map_dir = map_name
            if 'to' in map_name:
                map_dir = map_name.split('to')[1][1:]

            algo_path = os.path.join(result_dir, map_dir, algo_id)

            if not os.path.exists(algo_path):
                error_algo_path.append(algo_path)
                continue

            original_data[map_name][algo_id] = dict()

            seed_list = os.listdir(algo_path)
            if ".DS_Store" in seed_list:
                seed_list.remove(".DS_Store")

            seed_list.sort()

            tmp_seed_list = []
            # if algo_id != 'qmix_latent_scale':
            if algo_id not in seed_idx_list[map_name]:
                for i in range(len(seed_list)):
                    tmp_seed_list.append(seed_list[i])
            else:
                for i in seed_idx_list[map_name][algo_id]:
                    tmp_seed_list.append(seed_list[i])
            seed_list = tmp_seed_list

            for seed_id, seed_path in enumerate(seed_list):
                state_path = os.path.join(algo_path, seed_path, '1', 'run.json')

                if not os.path.exists(state_path) or os.path.getsize(state_path) == 0:
                    error_state_path.append(state_path)
                    continue

                with open(state_path) as json_file:
                    state = json.load(json_file)
                    if state['status'] != "RUNNING":
                        error_state_path.append(state_path)
                        continue

                original_data[map_name][algo_id][seed_id] = None
                data_path = os.path.join(algo_path, seed_path, '1', 'info.json')

                if not os.path.exists(data_path) or os.path.getsize(data_path) == 0:
                    error_data_path.append(data_path)
                    del original_data[map_name][algo_id][seed_id]
                    continue

                with open(data_path) as json_file:
                    data = json.load(json_file)

                    if env_name == 'sc2' or env_name == 'sc2mt':
                        data_y = data['test/battle_won_mean']
                        data_x = np.array(data['test/battle_won_mean_T'])
                    elif env_name == 'particle':
                        data_y = json_to_list(data['test/return_mean'])
                        data_x = np.array(data['test/return_mean_T'])

                    if 'to' in map_name:
                        data_x = data_x - total_timesteps[env_name]

                    if len(data_y) != len(data_x) or data_x[-1] < total_timesteps[env_name]:
                        error_timestep_path.append(data_path)
                        del original_data[map_name][algo_id][seed_id]
                        continue

                    original_data[map_name][algo_id][seed_id] = pd.DataFrame({'y': data_y, 'x': data_x})

            if original_data[map_name][algo_id] == dict():
                del original_data[map_name][algo_id]
                continue

            original_data[map_name][algo_id]['x'] = pd.concat([
                original_data[map_name][algo_id][seed]['x'] for seed in original_data[map_name][algo_id]
            ], axis=0, ignore_index=True).drop_duplicates().sort_values(ignore_index=True)

            original_data[map_name][algo_id]['y'] = []
            for seed in original_data[map_name][algo_id]:
                if seed == 'x' or seed == 'y':
                    continue
                original_data[map_name][algo_id]['y'].append(pd.merge(original_data[map_name][algo_id]['x'],
                                                                      original_data[map_name][algo_id][seed].loc[:, ['x', 'y']],
                                                                      how='left', left_on='x', right_on='x').interpolate(method='linear').fillna(0)['y'])

            original_data[map_name][algo_id]['x'] = np.array(original_data[map_name][algo_id]['x'])

            original_data[map_name][algo_id]['y'] = np.array(original_data[map_name][algo_id]['y'])
            # original_data[map_name][algo_id]['y'] = smooth(original_data[map_name][algo_id]['y'])

    print(error_state_path)
    print(error_data_path)
    print(error_timestep_path)

    return original_data


def get_original_data(env_name, map_list, algo_list, seed_idx_list):
    original_data = dict()
    result_dir = os.path.join(EXP_PATH, 'results', 'exp_v2', env_name)

    for map_name in map_list:
        original_data[map_name] = dict()
        for algo_id in algo_list[map_name]:
            original_data[map_name][algo_id] = dict()

            map_dir = map_name
            if 'to' in map_name:
                map_dir = map_name.split('to')[1][1:]

            algo_path = os.path.join(result_dir, map_dir, algo_id)
            seed_list = os.listdir(algo_path)
            seed_list.sort()

            tmp_seed_list = []
            # if algo_id != 'qmix_latent_scale':
            if algo_id not in seed_idx_list[map_name]:
                for i in range(len(seed_list)):
                    tmp_seed_list.append(seed_list[i])
            else:
                for i in seed_idx_list[map_name][algo_id]:
                    tmp_seed_list.append(seed_list[i])
            seed_list = tmp_seed_list

            if ".DS_Store" in seed_list:
                seed_list.remove(".DS_Store")

            # assert len(seed_list) >= 5, "Not enough seeds"
            # if len(seed_list) > 5:
                # del seed_list[5:]
                # del seed_list[:-5]

            for seed_id, seed_path in enumerate(seed_list):
                original_data[map_name][algo_id][seed_id] = dict()
                data_path = os.path.join(algo_path, seed_path, '1', 'info.json')
                with open(data_path) as json_file:
                    data = json.load(json_file)

                    if env_name == 'sc2':
                        data_y = data['test/battle_won_mean']
                        data_x = np.array(data['test/battle_won_mean_T'])
                    elif env_name == 'particle':
                        data_y = json_to_list(data['test/return_mean'])
                        data_x = np.array(data['test/return_mean_T'])

                    # original_data[map_name][algo_id][seed_id] = pd.DataFrame({'y': data['test/battle_won_mean'], 'x': data['test/battle_won_mean_T']})
                    if 'to' in map_name:
                        original_data[map_name][algo_id][seed_id] = pd.DataFrame({'y': data_y, 'x': data_x - 2000000})
                    else:
                        original_data[map_name][algo_id][seed_id] = pd.DataFrame({'y': data_y, 'x': data_x})

            original_data[map_name][algo_id]['x'] = pd.concat([
                original_data[map_name][algo_id][seed_id]['x'] for seed_id in range(len(seed_list))
            ], axis=0, ignore_index=True).drop_duplicates().sort_values(ignore_index=True)

            original_data[map_name][algo_id]['y'] = []
            for seed_id in range(len(seed_list)):
                original_data[map_name][algo_id]['y'].append(pd.merge(original_data[map_name][algo_id]['x'],
                                                                    original_data[map_name][algo_id][seed_id].loc[:, ['x', 'y']],
                                                                    how='left', left_on='x', right_on='x').interpolate(method='linear').fillna(0)['y'])

            original_data[map_name][algo_id]['x'] = np.array(original_data[map_name][algo_id]['x'])

            original_data[map_name][algo_id]['y'] = np.array(original_data[map_name][algo_id]['y'])
            # original_data[map_name][algo_id]['y'] = smooth(original_data[map_name][algo_id]['y'])

    return original_data


def changex(temp, position):
    return int(temp/100000)


def plot_reward_results(original_data, algo_list, map_name, env_name, type):
    filename = env_name + '_' + type + '_' + map_name + '.pdf'

    plt.figure(figsize=(10, 7))
    # plt.figure(figsize=(10, 4))

    plt.gca().xaxis.set_major_formatter(FuncFormatter(changex))

    gap = 200

    for idx, algo_id in enumerate(original_data[map_name]):
        sns.tsplot(time=original_data[map_name][algo_id]['x'][0::gap], data=original_data[map_name][algo_id]['y'][:, 0::gap],
                   linestyle=linestyle[0], condition=algo_id, color=sns.color_palette(n_colors=12)[idx])


    # for idx, algo_id in enumerate(algo_list[map_name]):
    #     sns.tsplot(time=original_data[map_name][algo_id]['x'][0::gap], data=original_data[map_name][algo_id]['y'][:, 0::gap],
    #                linestyle=linestyle[0], condition=algo_id[4:] if 'updet' in algo_id else algo_id, color=sns.color_palette()[idx])

    # plt.legend(loc='upper left', ncol=1, fontsize=14)
    # plt.legend(loc='upper center', ncol=3, mode="expand", fontsize=14)
    plt.legend(loc='upper center', ncol=2, handlelength=2,
               mode="expand", borderaxespad=0.1, prop={'size': 14})
    plt.title(map_name, fontsize=fontsize)

    plt.xlabel(r'Total timesteps ($\times 10^5$)', fontsize=fontsize)

    if env_name == 'sc2':
        plt.xlim((-10000, 2000000 + 20000))
        plt.ylim((-0.1, 1.6))
        plt.xticks([0, 500000, 1000000, 1500000, 2000000], fontsize=fontsize)
        plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=fontsize)
        plt.ylabel('Median Test Win %', fontsize=fontsize, labelpad=10)
    elif env_name == 'sc2mt':
        plt.xlim((-10000, 10000000 + 20000))
        plt.ylim((-0.1, 1.6))
        plt.xticks([0, 2000000, 4000000, 6000000, 8000000, 10000000], fontsize=fontsize)
        plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=fontsize)
        plt.ylabel('Median Test Win %', fontsize=fontsize, labelpad=10)
    elif env_name == 'particle':
        plt.xlim((-10000, 1000000 + 20000))
        plt.ylim((-0.1, 20))
        plt.xticks([0, 200000, 400000, 600000, 800000, 1000000], fontsize=fontsize)
        plt.yticks([0.0, 2.0, 4.0, 6.0, 8.0, 10.0], fontsize=fontsize)
        plt.ylabel('Average Return', fontsize=fontsize, labelpad=10)

    # plt.xticks([0, 500000, 1000000, 1500000, 2000000, 2500000, 3000000, 3500000, 4000000], fontsize=fontsize)

    plt.savefig(os.path.join(EXP_PATH, 'results', 'fig', filename), format='pdf', bbox_inches='tight')
    plt.show()


def plot_attention_map():
    attn_mix_data = []  # n_blocks * timesteps * n_heads * n_units * n_units
    attn_mix_data.append(np.load(os.path.join(EXP_PATH, 'results', 'fig', 'attn_mix', 'block_0', 'attn.npy')))
    attn_mix_data.append(np.load(os.path.join(EXP_PATH, 'results', 'fig', 'attn_mix', 'block_1', 'attn.npy')))
    attn_mix_data = np.array(attn_mix_data)
    attn_mix_data = attn_mix_data.reshape(2, -1, 3, attn_mix_data.shape[3], attn_mix_data.shape[4])

    attn_agent_data = []  # n_blocks * timesteps * n_agents * n_heads * n_units * n_units
    attn_agent_data.append(np.load(os.path.join(EXP_PATH, 'results', 'fig', 'attn_agent', 'block_0', 'attn.npy')))
    attn_agent_data.append(np.load(os.path.join(EXP_PATH, 'results', 'fig', 'attn_agent', 'block_1', 'attn.npy')))
    attn_agent_data = np.array(attn_agent_data)[:, -71:-1, :, :-1, :-1]
    attn_agent_data = attn_agent_data.reshape(2, attn_agent_data.shape[1], -1, 3, attn_agent_data.shape[3], attn_agent_data.shape[4])

    # 一行是一个 attention，每行的列元素求和为 0

    filename = 'attn_mix.pdf'
    plt.figure(figsize=(10, 3))
    n_heads = attn_mix_data.shape[2]
    for i in range(1, n_heads + 1):
        plt.subplot(1, n_heads, i)
        # sns.heatmap(1 - attn_mix_data[0, 0, i - 1], vmin=0.5, vmax=1, cmap='rocket', linewidths=.5)
        sns.heatmap(attn_mix_data[0, 0, i - 1], cmap=sns.cubehelix_palette(as_cmap=True, gamma=0.8), linewidths=.5)
        plt.xticks([])
        # plt.yticks([])
        # plt.axis('off')
        plt.tight_layout()
    plt.savefig(os.path.join(EXP_PATH, 'results', 'fig', filename), format='pdf', bbox_inches='tight')
    plt.show()

    filename = 'attn_agent.pdf'
    plt.figure(figsize=(10, 15))
    n_heads = attn_agent_data.shape[3]
    n_agents = attn_agent_data.shape[2]
    for i in range(1, n_heads + 1):
        for j in range(1, n_agents + 1):
            plt.subplot(n_agents, n_heads, (j - 1) * n_heads + i)
            # sns.heatmap(1 - attn_agent_data[0, 0, 0, i - 1], vmin=0.5, vmax=1, cmap='rocket', linewidths=.5)
            sns.heatmap(attn_agent_data[0, 0, j - 1, i - 1], cmap=sns.cubehelix_palette(as_cmap=True, gamma=0.8), linewidths=.5)

            plt.xticks([])
            # plt.yticks([])
            # plt.axis('off')
            plt.tight_layout()
    plt.savefig(os.path.join(EXP_PATH, 'results', 'fig', filename), format='pdf', bbox_inches='tight')
    plt.show()


# seed_idx_list = {
#     '3m': {'vdn_updet': [0, 1, 2, 3, 4], 'qmix_ext': [0, 1, 2, 3, 4], 'qmix_ext_scale': [0, 1, 2, 3, 4], 'qmix_latent': [0, 1, 2, 3, 4], 'qmix_latent_scale': [1, 2, 3, 4, 5]},
#     '5m_vs_6m': {'vdn_updet': [0, 1, 2, 3, 4], 'qmix_ext': [0, 2, 3, 4, 5], 'qmix_ext_scale': [0, 1, 2, 3, 4], 'qmix_latent': [1, 2, 4, 5, 6], 'qmix_latent_scale': [0, 1, 2, 4, 5]},
#     '8m_vs_9m': {'vdn_updet': [0, 1, 2, 3, 4], 'qmix_ext': [1, 2, 3, 4, 5], 'qmix_ext_scale': [2, 3, 5, 7, 8], 'qmix_latent': [1, 2, 3, 4, 5], 'qmix_latent_scale': [0, 1, 2, 3, 4]},
#     '10m_vs_11m': {'vdn_updet': [0, 1, 2, 3, 4], 'qmix_ext': [0, 4, 5, 6, 7], 'qmix_ext_scale': [0, 1, 2, 4, 5], 'qmix_latent': [1, 3, 4, 5, 6], 'qmix_latent_scale': [0, 1, 2, 3, 4]},
#     '2s3z': {'vdn_updet': [1, 2, 3, 4, 5], 'qmix_ext': [0, 1, 2, 3, 4], 'qmix_ext_scale': [0, 1, 2, 3, 4], 'qmix_latent': [0, 1, 3, 4, 5], 'qmix_latent_scale': [0, 1, 2, 3, 4]},
#     '3s5z': {'vdn_updet': [0, 1, 2, 3, 4], 'qmix_ext': [0, 2, 4, 5, 6], 'qmix_ext_scale': [0, 1, 2, 3, 4], 'qmix_latent': [0, 2, 4, 5, 6], 'qmix_latent_scale': [1, 7, 9, 10, 11], 'qmix': [0, 1, 2, 4, 6]},
#     '3s_vs_3z': {'vdn_updet': [0, 1, 2, 3, 4], 'qmix_ext': [0, 1, 2, 3, 4], 'qmix_ext_scale': [0, 1, 2, 3, 4], 'qmix_latent': [0, 1, 2, 3, 4], 'qmix_latent_scale': [0, 1, 2, 3, 4]},
#     '3s_vs_5z': {'vdn_updet': [0, 1, 2, 3, 4], 'qmix_ext': [0, 1, 2, 3, 4], 'qmix_ext_scale': [0, 1, 2, 3, 4], 'qmix_latent': [0, 1, 2, 3, 4], 'qmix_latent_scale': [0, 3, 4, 9, 13], 'qmix': [1, 2, 4, 5, 6], 'vdn': [0, 1, 3, 4, 10]},
#     '3m_to_5m_vs_6m': {},
#     '5m_vs_6m_to_3m': {'qmix_latent_scale_5m_vs_6m': [1, 2, 3, 4, 5]},
#     '8m_vs_9m_to_10m_vs_11m': {},
#     '10m_vs_11m_to_8m_vs_9m': {'qmix_latent_scale_10m_vs_11m': [0, 1, 2, 4, 5]},
#     '3m_to_10m_vs_11m': {'qmix_latent_scale_3m': [0, 1, 2, 4, 5]},
#     '10m_vs_11m_to_3m': {},
#     '2s3z_to_3s5z': {'qmix_latent_scale_2s3z': [0, 1, 3, 7, 8]},
#     '3s_vs_3z_to_3s_vs_5z': {'qmix_latent_scale_3s_vs_3z': [4, 0, 3, 5, 8]},
#     '3s5z_to_3s_vs_5z': {},
#     '3s_vs_5z_to_3s5z': {},
#     '3m_to_8m_vs_9m': {'qmix_latent_scale_3m': [0, 1, 2, 4, 5]},
#     '5m_vs_6m_to_8m_vs_9m': {},
#     'tag_4_4_2': {},
#     'tag_8_8_2': {},
#     'tag_16_16_2': {},
#     'htag_8_4_2': {},
#     'htag_16_8_2': {},
#     '25m': {},
# }

seed_idx_list = {
    '3m': {},
    '5m_vs_6m': {},
    '8m_vs_9m': {},
    '10m_vs_11m': {},
    '2s3z': {},
    '3s5z': {},
    '3s_vs_3z': {},
    '3s_vs_5z': {},
    '3m_to_5m_vs_6m': {},
    '5m_vs_6m_to_3m': {},
    '8m_vs_9m_to_10m_vs_11m': {},
    '10m_vs_11m_to_8m_vs_9m': {},
    '3m_to_10m_vs_11m': {},
    '10m_vs_11m_to_3m': {},
    '2s3z_to_3s5z': {},
    '3s_vs_3z_to_3s_vs_5z': {},
    '3s5z_to_3s_vs_5z': {},
    '3s_vs_5z_to_3s5z': {},
    '3m_to_8m_vs_9m': {},
    '5m_vs_6m_to_8m_vs_9m': {},
    'tag_4_4_2': {},
    'tag_8_8_2': {},
    'tag_16_16_2': {},
    'htag_8_4_2': {},
    'htag_16_8_2': {},
    '25m': {},
    '3-8m_symmetric': {},
    '3-8sz_symmetric': {},
    '3-8MMM_symmetric': {},
    '3-8csz_symmetric': {},
}


def plot_sc2_normal_all():
    map_list = ['3m', '5m_vs_6m', '8m_vs_9m', '10m_vs_11m', '2s3z', '3s5z', '3s_vs_3z', '3s_vs_5z']
    algo_list = {
        '3m': ['coma', 'iql', 'vdn', 'qmix', 'qtran', 'vdn_updet', 'qmix_ext', 'qmix_ext_scale', 'qmix_latent', 'qmix_latent_scale'],
        '5m_vs_6m': ['coma', 'iql', 'vdn', 'qmix', 'qtran', 'vdn_updet', 'qmix_ext', 'qmix_ext_scale', 'qmix_latent', 'qmix_latent_scale'],
        '8m_vs_9m': ['coma', 'iql', 'vdn', 'qmix', 'qtran', 'vdn_updet', 'qmix_ext', 'qmix_ext_scale', 'qmix_latent', 'qmix_latent_scale'],
        '10m_vs_11m': ['coma', 'iql', 'vdn', 'qmix', 'qtran', 'vdn_updet', 'qmix_ext', 'qmix_ext_scale', 'qmix_latent', 'qmix_latent_scale'],
        '2s3z': ['coma', 'iql', 'vdn', 'qmix', 'qtran', 'vdn_updet', 'qmix_ext', 'qmix_ext_scale', 'qmix_latent', 'qmix_latent_scale'],
        '3s5z': ['coma', 'iql', 'vdn', 'qmix', 'qtran', 'vdn_updet', 'qmix_ext', 'qmix_ext_scale', 'qmix_latent', 'qmix_latent_scale'],
        '3s_vs_3z': ['coma', 'iql', 'vdn', 'qmix', 'qtran', 'vdn_updet', 'qmix_ext', 'qmix_ext_scale', 'qmix_latent', 'qmix_latent_scale'],
        '3s_vs_5z': ['coma', 'iql', 'vdn', 'qmix', 'qtran', 'vdn_updet', 'qmix_ext', 'qmix_ext_scale', 'qmix_latent', 'qmix_latent_scale']
    }

    original_data = get_original_data('sc2', map_list, algo_list, seed_idx_list)
    for map_name in map_list:
        plot_reward_results(original_data, algo_list, map_name, 'sc2', 'normal_all')


def plot_sc2_normal_sota():
    map_list = ['3m', '5m_vs_6m', '8m_vs_9m', '10m_vs_11m', '2s3z', '3s5z', '3s_vs_3z', '3s_vs_5z']
    # map_list = ['3m', '5m_vs_6m', '8m_vs_9m', '10m_vs_11m', '2s3z', '3s5z', '3s_vs_3z', '3s_vs_5z', '25m']
    algo_list = {
        '3m': ['coma', 'iql', 'vdn', 'qmix', 'qtran', 'qplex', 'token_qmix_wise_attn', 'token_qmix_wise_trans', 'token_vdn_wise_attn', 'token_vdn_wise_trans', 'token_qmix_updet', 'token_vdn_updet'],
        '5m_vs_6m': ['coma', 'iql', 'vdn', 'qmix', 'qtran', 'qplex', 'token_qmix_wise_attn', 'token_qmix_wise_trans', 'token_vdn_wise_attn', 'token_vdn_wise_trans', 'token_qmix_updet', 'token_vdn_updet'],
        '8m_vs_9m': ['coma', 'iql', 'vdn', 'qmix', 'qtran', 'qplex', 'token_qmix_wise_attn', 'token_qmix_wise_trans', 'token_vdn_wise_attn', 'token_vdn_wise_trans', 'token_qmix_updet', 'token_vdn_updet'],
        '10m_vs_11m': ['coma', 'iql', 'vdn', 'qmix', 'qtran', 'qplex', 'token_qmix_wise_attn', 'token_qmix_wise_trans', 'token_vdn_wise_attn', 'token_vdn_wise_trans', 'token_qmix_updet', 'token_vdn_updet'],
        '2s3z': ['coma', 'iql', 'vdn', 'qmix', 'qtran', 'qplex', 'token_qmix_wise_attn', 'token_qmix_wise_trans', 'token_vdn_wise_attn', 'token_vdn_wise_trans', 'token_qmix_updet', 'token_vdn_updet'],
        '3s5z': ['coma', 'iql', 'vdn', 'qmix', 'qtran', 'qplex', 'token_qmix_wise_attn', 'token_qmix_wise_trans', 'token_vdn_wise_attn', 'token_vdn_wise_trans', 'token_qmix_updet', 'token_vdn_updet'],
        '3s_vs_3z': ['coma', 'iql', 'vdn', 'qmix', 'qtran', 'qplex', 'token_qmix_wise_attn', 'token_qmix_wise_trans', 'token_vdn_wise_attn', 'token_vdn_wise_trans', 'token_qmix_updet', 'token_vdn_updet'],
        '3s_vs_5z': ['coma', 'iql', 'vdn', 'qmix', 'qtran', 'qplex', 'token_qmix_wise_attn', 'token_qmix_wise_trans', 'token_vdn_wise_attn', 'token_vdn_wise_trans', 'token_qmix_updet', 'token_vdn_updet'],
        '25m': ['coma', 'iql', 'vdn', 'qmix', 'qtran', 'qplex', 'token_qmix_wise_attn', 'token_qmix_wise_trans', 'token_vdn_wise_attn', 'token_vdn_wise_trans', 'token_qmix_updet', 'token_vdn_updet']
    }

    original_data = check_original_data('sc2', map_list, algo_list, seed_idx_list)
    for map_name in map_list:
        plot_reward_results(original_data, algo_list, map_name, 'sc2', 'normal_sota')


def plot_sc2mt_normal_sota():
    map_list = ['3-8m_symmetric', '3-8sz_symmetric', '3-8MMM_symmetric', '3-8csz_symmetric']
    algo_list = {
        '3-8m_symmetric': ['entity_qmix_attn', 'entity_qmix_trans', 'entity_vdn_attn', 'entity_vdn_trans', 'entity_qmix_refil_attn', 'entity_vdn_refil_attn', 'entity_qmix_refil_imagine', 'entity_qmix_refil_imagine_parallel'],
        '3-8sz_symmetric': ['entity_qmix_attn', 'entity_qmix_trans', 'entity_vdn_attn', 'entity_vdn_trans', 'entity_qmix_refil_attn', 'entity_vdn_refil_attn', 'entity_qmix_refil_imagine', 'entity_qmix_refil_imagine_parallel'],
        '3-8MMM_symmetric': ['entity_qmix_attn', 'entity_qmix_trans', 'entity_vdn_attn', 'entity_vdn_trans', 'entity_qmix_refil_attn', 'entity_vdn_refil_attn', 'entity_qmix_refil_imagine', 'entity_qmix_refil_imagine_parallel'],
        '3-8csz_symmetric': ['entity_qmix_attn', 'entity_qmix_trans', 'entity_vdn_attn', 'entity_vdn_trans', 'entity_qmix_refil_attn', 'entity_vdn_refil_attn', 'entity_qmix_refil_imagine', 'entity_qmix_refil_imagine_parallel'],
    }

    original_data = check_original_data('sc2mt', map_list, algo_list, seed_idx_list)
    for map_name in map_list:
        plot_reward_results(original_data, algo_list, map_name, 'sc2mt', 'normal_sota')


def plot_sc2_normal_baseline():
    map_list = ['5m_vs_6m', '8m_vs_9m', '3s_vs_5z', '2s3z']
    # map_list = ['3m', '5m_vs_6m', '8m_vs_9m', '10m_vs_11m', '2s3z', '3s5z', '3s_vs_3z', '3s_vs_5z', '25m']
    algo_list = {
        '3m': ['coma', 'iql', 'vdn', 'qmix', 'qtran', 'vdn_updet', 'qmix_latent_scale'],
        '5m_vs_6m': ['coma', 'iql', 'vdn', 'qmix', 'qtran', 'vdn_updet', 'qmix_latent_scale', 'qmix_sparse'],
        '8m_vs_9m': ['coma', 'iql', 'vdn', 'qmix', 'qtran', 'vdn_updet', 'qmix_latent_scale', 'qmix_sparse'],
        '10m_vs_11m': ['coma', 'iql', 'vdn', 'qmix', 'qtran', 'vdn_updet', 'qmix_latent_scale'],
        '2s3z': ['coma', 'iql', 'vdn', 'qmix', 'qtran', 'vdn_updet', 'qmix_latent_scale', 'qmix_sparse'],
        '3s5z': ['coma', 'iql', 'vdn', 'qmix', 'qtran', 'vdn_updet', 'qmix_latent_scale'],
        '3s_vs_3z': ['coma', 'iql', 'vdn', 'qmix', 'qtran', 'vdn_updet', 'qmix_latent_scale'],
        '3s_vs_5z': ['coma', 'iql', 'vdn', 'qmix', 'qtran', 'vdn_updet', 'qmix_latent_scale', 'qmix_sparse'],
        '25m': ['coma', 'iql', 'vdn', 'qmix', 'qtran', 'vdn_updet', 'qmix_latent_scale']
    }

    original_data = get_original_data('sc2', map_list, algo_list, seed_idx_list)
    for map_name in map_list:
        plot_reward_results(original_data, algo_list, map_name, 'sc2', 'normal_baseline')


def plot_sc2_normal_ablation():
    map_list = ['3m', '5m_vs_6m', '8m_vs_9m', '10m_vs_11m', '2s3z', '3s5z', '3s_vs_3z', '3s_vs_5z']
    algo_list = {
        '3m': ['vdn_updet', 'qmix_ext', 'qmix_ext_scale', 'qmix_latent', 'qmix_latent_scale'],
        '5m_vs_6m': ['vdn_updet', 'qmix_ext', 'qmix_ext_scale', 'qmix_latent', 'qmix_latent_scale'],
        '8m_vs_9m': ['vdn_updet', 'qmix_ext', 'qmix_ext_scale', 'qmix_latent', 'qmix_latent_scale'],
        '10m_vs_11m': ['vdn_updet', 'qmix_ext', 'qmix_ext_scale', 'qmix_latent', 'qmix_latent_scale'],
        '2s3z': ['vdn_updet', 'qmix_ext', 'qmix_ext_scale', 'qmix_latent', 'qmix_latent_scale'],
        '3s5z': ['vdn_updet', 'qmix_ext', 'qmix_ext_scale', 'qmix_latent', 'qmix_latent_scale'],
        '3s_vs_3z': ['vdn_updet', 'qmix_ext', 'qmix_ext_scale', 'qmix_latent', 'qmix_latent_scale'],
        '3s_vs_5z': ['vdn_updet', 'qmix_ext', 'qmix_ext_scale', 'qmix_latent', 'qmix_latent_scale']
    }

    original_data = get_original_data('sc2', map_list, algo_list, seed_idx_list)
    for map_name in map_list:
        plot_reward_results(original_data, algo_list, map_name, 'sc2', 'normal_ablation')


def plot_sc2_transfer_all():
    # map_list = ['3s_vs_3z_to_3s_vs_5z']
    map_list = ['3m_to_5m_vs_6m', '5m_vs_6m_to_3m', '8m_vs_9m_to_10m_vs_11m', '10m_vs_11m_to_8m_vs_9m',
                '3m_to_10m_vs_11m', '10m_vs_11m_to_3m', '2s3z_to_3s5z', '3s_vs_3z_to_3s_vs_5z', '3s5z_to_3s_vs_5z', '3s_vs_5z_to_3s5z', '3m_to_8m_vs_9m', '5m_vs_6m_to_8m_vs_9m']
    algo_list = {
        '3m_to_5m_vs_6m': ['vdn_updet_3m', 'qmix_ext_3m', 'qmix_ext_scale_3m', 'qmix_latent_3m', 'qmix_latent_scale_3m'],
        '5m_vs_6m_to_3m': ['vdn_updet_5m_vs_6m', 'qmix_ext_5m_vs_6m', 'qmix_ext_scale_5m_vs_6m', 'qmix_latent_5m_vs_6m', 'qmix_latent_scale_5m_vs_6m'],
        '8m_vs_9m_to_10m_vs_11m': ['vdn_updet_8m_vs_9m', 'qmix_ext_8m_vs_9m', 'qmix_ext_scale_8m_vs_9m', 'qmix_latent_8m_vs_9m', 'qmix_latent_scale_8m_vs_9m'],
        '10m_vs_11m_to_8m_vs_9m': ['vdn_updet_10m_vs_11m', 'qmix_ext_10m_vs_11m', 'qmix_ext_scale_10m_vs_11m', 'qmix_latent_10m_vs_11m', 'qmix_latent_scale_10m_vs_11m'],
        '3m_to_10m_vs_11m': ['vdn_updet_3m', 'qmix_ext_3m', 'qmix_ext_scale_3m', 'qmix_latent_3m', 'qmix_latent_scale_3m'],
        '10m_vs_11m_to_3m': ['vdn_updet_10m_vs_11m', 'qmix_ext_10m_vs_11m', 'qmix_ext_scale_10m_vs_11m', 'qmix_latent_10m_vs_11m', 'qmix_latent_scale_10m_vs_11m'],
        '2s3z_to_3s5z': ['vdn_updet_2s3z', 'qmix_ext_2s3z', 'qmix_ext_scale_2s3z', 'qmix_latent_2s3z', 'qmix_latent_scale_2s3z'],
        '3s_vs_3z_to_3s_vs_5z': ['vdn_updet_3s_vs_3z', 'qmix_ext_3s_vs_3z', 'qmix_ext_scale_3s_vs_3z', 'qmix_latent_3s_vs_3z', 'qmix_latent_scale_3s_vs_3z'],
        '3s5z_to_3s_vs_5z': ['vdn_updet_3s5z', 'qmix_ext_3s5z', 'qmix_ext_scale_3s5z', 'qmix_latent_3s5z', 'qmix_latent_scale_3s5z'],
        '3s_vs_5z_to_3s5z': ['vdn_updet_3s_vs_5z', 'qmix_ext_3s_vs_5z', 'qmix_ext_scale_3s_vs_5z', 'qmix_latent_3s_vs_5z', 'qmix_latent_scale_3s_vs_5z'],
        '3m_to_8m_vs_9m': ['vdn_updet_3m', 'qmix_ext_3m', 'qmix_ext_scale_3m', 'qmix_latent_3m', 'qmix_latent_scale_3m'],
        '5m_vs_6m_to_8m_vs_9m': ['vdn_updet_5m_vs_6m', 'qmix_ext_5m_vs_6m', 'qmix_ext_scale_5m_vs_6m', 'qmix_latent_5m_vs_6m', 'qmix_latent_scale_5m_vs_6m'],
    }

    original_data = get_original_data('sc2', map_list, algo_list, seed_idx_list)
    for map_name in map_list:
        plot_reward_results(original_data, algo_list, map_name, 'sc2', 'transfer_all')


def plot_sc2_transfer_baseline():
    # map_list = ['3s_vs_3z_to_3s_vs_5z']
    map_list = ['3m_to_5m_vs_6m', '5m_vs_6m_to_3m', '8m_vs_9m_to_10m_vs_11m', '10m_vs_11m_to_8m_vs_9m',
                '3m_to_10m_vs_11m', '10m_vs_11m_to_3m', '2s3z_to_3s5z', '3s_vs_3z_to_3s_vs_5z', '3s5z_to_3s_vs_5z', '3s_vs_5z_to_3s5z', '3m_to_8m_vs_9m', '5m_vs_6m_to_8m_vs_9m']
    algo_list = {
        '3m_to_5m_vs_6m': ['vdn_updet_3m', 'qmix_latent_scale_3m'],
        '5m_vs_6m_to_3m': ['vdn_updet_5m_vs_6m', 'qmix_latent_scale_5m_vs_6m'],
        '8m_vs_9m_to_10m_vs_11m': ['vdn_updet_8m_vs_9m', 'qmix_latent_scale_8m_vs_9m'],
        '10m_vs_11m_to_8m_vs_9m': ['vdn_updet_10m_vs_11m', 'qmix_latent_scale_10m_vs_11m'],
        '3m_to_10m_vs_11m': ['vdn_updet_3m', 'qmix_latent_scale_3m'],
        '10m_vs_11m_to_3m': ['vdn_updet_10m_vs_11m', 'qmix_latent_scale_10m_vs_11m'],
        '2s3z_to_3s5z': ['vdn_updet_2s3z', 'qmix_latent_scale_2s3z'],
        '3s_vs_3z_to_3s_vs_5z': ['vdn_updet_3s_vs_3z', 'qmix_latent_scale_3s_vs_3z'],
        '3s5z_to_3s_vs_5z': ['vdn_updet_3s5z', 'qmix_latent_scale_3s5z'],
        '3s_vs_5z_to_3s5z': ['vdn_updet_3s_vs_5z', 'qmix_latent_scale_3s_vs_5z'],
        '3m_to_8m_vs_9m': ['vdn_updet_3m', 'qmix_latent_scale_3m'],
        '5m_vs_6m_to_8m_vs_9m': ['vdn_updet_5m_vs_6m', 'qmix_latent_scale_5m_vs_6m'],
    }

    original_data = get_original_data('sc2', map_list, algo_list, seed_idx_list)
    for map_name in map_list:
        plot_reward_results(original_data, algo_list, map_name, 'sc2', 'transfer_baseline')


def plot_particle_normal_all():
    map_list = ['tag_4_4_2', 'tag_8_8_2', 'tag_16_16_2', 'htag_8_4_2', 'htag_16_8_2']
    algo_list = {
        'tag_4_4_2': ['vdn', 'qmix', 'vdn_updet', 'qmix_ext', 'qmix_ext_scale', 'qmix_latent', 'qmix_latent_scale'],
        'tag_8_8_2': ['vdn', 'qmix', 'vdn_updet', 'qmix_ext', 'qmix_ext_scale', 'qmix_latent', 'qmix_latent_scale'],
        'tag_16_16_2': ['vdn', 'qmix', 'vdn_updet', 'qmix_ext', 'qmix_ext_scale', 'qmix_latent', 'qmix_latent_scale'],
        'htag_8_4_2': ['vdn', 'qmix', 'vdn_updet', 'qmix_ext', 'qmix_ext_scale', 'qmix_latent', 'qmix_latent_scale'],
        'htag_16_8_2': ['vdn', 'qmix', 'vdn_updet', 'qmix_ext', 'qmix_ext_scale', 'qmix_latent', 'qmix_latent_scale']
    }
    seed_idx_list = {
        'tag_4_4_2': {},
        'tag_8_8_2': {},
        'tag_16_16_2': {},
        'htag_8_4_2': {},
        'htag_16_8_2': {},
    }
    original_data = get_original_data('particle', map_list, algo_list, seed_idx_list)
    for map_name in map_list:
        plot_reward_results(original_data, algo_list, map_name, 'particle', 'normal_all')


if __name__ == '__main__':
    # plot_sc2_normal_all()
    # plot_sc2_normal_baseline()
    # plot_sc2_normal_ablation()
    # plot_sc2_transfer_all()
    # plot_sc2_transfer_baseline()
    # plot_particle_normal_all()
    # plot_attention_map()
    plot_sc2_normal_sota()
    # plot_sc2mt_normal_sota()




