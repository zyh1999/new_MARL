import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os
import json
import numpy as np
import shutil
import pandas as pd
sns.set()
sns.set_style("darkgrid", {"axes.facecolor": "#f0f0f7"})
linestyle = ['-', '--', ':', '-.']
fontsize = 20

EXP_PATH = os.path.join(os.environ['NFS_HOME'], 'code/pymarl')

total_timesteps = {'sc2': 2000000, 'sc2mt': 2000000, 'particle': 1000000}


def mv_rm(path_list):
    # for path in path_list:
    #     to_path = os.path.join(os.environ['NFS_HOME'], 'dump', os.path.dirname(path)[1:])
    #     if not os.path.exists(to_path):
    #         os.makedirs(to_path)
    #     shutil.move(path, to_path)
    pass


def print_error(map_list, algo_list, error_original_data):
    for map_name in map_list:
        for algo_id in algo_list:
            if error_original_data[map_name][algo_id]['exist']:
                del error_original_data[map_name][algo_id]['exist']
                flag = 0

                if not error_original_data[map_name][algo_id]['state_file_error']:
                    flag += 1
                    del error_original_data[map_name][algo_id]['state_file_error']

                if not error_original_data[map_name][algo_id]['state_error']:
                    flag += 1
                    del error_original_data[map_name][algo_id]['state_error']
                else:
                    mv_rm(error_original_data[map_name][algo_id]['state_error'])

                if not error_original_data[map_name][algo_id]['config_error']:
                    flag += 1
                    del error_original_data[map_name][algo_id]['config_error']

                if not error_original_data[map_name][algo_id]['data_file_error']:
                    flag += 1
                    del error_original_data[map_name][algo_id]['data_file_error']
                else:
                    mv_rm(error_original_data[map_name][algo_id]['data_file_error'])

                if not error_original_data[map_name][algo_id]['data_json_error']:
                    flag += 1
                    del error_original_data[map_name][algo_id]['data_json_error']

                if not error_original_data[map_name][algo_id]['timestep_error']:
                    flag += 1
                    del error_original_data[map_name][algo_id]['timestep_error']
                else:
                    mv_rm(error_original_data[map_name][algo_id]['timestep_error'])

                if flag == 6:
                    del error_original_data[map_name][algo_id]

    print("error_original_data:")
    print(json.dumps(error_original_data, indent=2))


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


def get_original_data(env_name, map_list, algo_list):
    original_data = dict()
    error_original_data = dict()
    result_dir = os.path.join(EXP_PATH, 'results', 'exp_v2', env_name)

    for map_name in map_list:
        original_data[map_name] = dict()
        error_original_data[map_name] = dict()
        for algo_id in algo_list:
            error_original_data[map_name][algo_id] = dict()

            algo_path = os.path.join(result_dir, map_name, algo_id)

            if not os.path.exists(algo_path):
                error_original_data[map_name][algo_id]['exist'] = False
                continue

            original_data[map_name][algo_id] = dict()
            error_original_data[map_name][algo_id]['exist'] = True
            error_original_data[map_name][algo_id]['state_file_error'] = []
            error_original_data[map_name][algo_id]['state_error'] = []
            error_original_data[map_name][algo_id]['config_error'] = []
            error_original_data[map_name][algo_id]['data_file_error'] = []
            error_original_data[map_name][algo_id]['data_json_error'] = []
            error_original_data[map_name][algo_id]['timestep_error'] = []

            seed_list = os.listdir(algo_path)
            if ".DS_Store" in seed_list:
                seed_list.remove(".DS_Store")

            seed_list.sort()

            if algo_id in seed_idx_list[map_name]:
                tmp_seed_list = []
                for i in seed_idx_list[map_name][algo_id]:
                    tmp_seed_list.append(seed_list[i])
                seed_list = tmp_seed_list

            for seed_id, seed_path in enumerate(seed_list):
                if len(original_data[map_name][algo_id]) == 5:
                    continue

                error_path = os.path.join(algo_path, seed_path)
                state_path = os.path.join(algo_path, seed_path, '1', 'run.json')

                if not os.path.exists(state_path) or os.path.getsize(state_path) == 0:
                    error_original_data[map_name][algo_id]['state_file_error'].append(error_path)
                    continue

                with open(state_path) as json_file:
                    state = json.load(json_file)
                    if state['status'] != "RUNNING":
                        error_original_data[map_name][algo_id]['state_error'].append(error_path)
                        continue

                config_path = os.path.join(algo_path, seed_path, '1', 'config.json')
                with open(config_path) as json_file:
                    config = json.load(json_file)
                    if 'epsilon500k' in config['name'] and config['name'] in algo_id:
                        if config['epsilon_anneal_time'] != 500000:
                            error_original_data[map_name][algo_id]['config_error'].append(error_path)
                    elif 'epsilon50k' in config['name'] and config['name'] in algo_id:
                        if config['epsilon_anneal_time'] != 50000:
                            error_original_data[map_name][algo_id]['config_error'].append(error_path)
                    elif config['name'] != algo_id:
                        error_original_data[map_name][algo_id]['config_error'].append(error_path)
                    elif env_name == 'sc2mt' and config['epsilon_anneal_time'] != 500000:
                        error_original_data[map_name][algo_id]['config_error'].append(error_path)
                    elif env_name == 'sc2' and config['epsilon_anneal_time'] != 50000:
                        error_original_data[map_name][algo_id]['config_error'].append(error_path)

                original_data[map_name][algo_id][seed_id] = None
                data_path = os.path.join(algo_path, seed_path, '1', 'info.json')

                if not os.path.exists(data_path) or os.path.getsize(data_path) == 0:
                    error_original_data[map_name][algo_id]['data_file_error'].append(error_path)
                    del original_data[map_name][algo_id][seed_id]
                    continue

                with open(data_path) as json_file:
                    try:
                        data = json.load(json_file)
                        if env_name == 'sc2' or env_name == 'sc2mt':
                            data_y = data['test/battle_won_mean']
                            data_x = np.array(data['test/battle_won_mean_T'])
                        elif env_name == 'particle':
                            data_y = json_to_list(data['test/return_mean'])
                            data_x = np.array(data['test/return_mean_T'])
                    except:
                        error_original_data[map_name][algo_id]['data_json_error'].append(error_path)
                        del original_data[map_name][algo_id][seed_id]
                        continue

                    if len(data_y) != len(data_x) or data_x[-1] < total_timesteps[env_name]:
                        error_original_data[map_name][algo_id]['timestep_error'].append(error_path)
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
            # original_data[map_name][algo_id]['y'] = np.array(smooth(original_data[map_name][algo_id]['y']))

    print_error(map_list, algo_list, error_original_data)

    original_data_cnt = dict()
    for map_name in map_list:
        original_data_cnt[map_name] = dict()
        for algo_id in algo_list:
            original_data_cnt[map_name][algo_id] = 0
            if algo_id in original_data[map_name]:
                original_data_cnt[map_name][algo_id] = len(original_data[map_name][algo_id]) - 2
    print("original_data_cnt:")
    print(json.dumps(original_data_cnt, indent=2))

    return original_data


def changex(temp, position):
    return float(temp/1000000)


def plot_reward_results(original_data, map_name, env_name, info):
    filename = env_name + '_' + info + '_' + map_name + '.pdf'

    if env_name == 'sc2mt':
        plt.figure(figsize=(12, 6))
    elif env_name == 'sc2':
        plt.figure(figsize=(10, 6))

    plt.gca().xaxis.set_major_formatter(FuncFormatter(changex))

    if env_name == 'sc2mt':
        gap = 40
        name = {'entity_qmix_attn': 'QMIX',
                'entity_vdn_attn': 'VDN',
                'entity_qmix_refil_imagine': 'REFIL',
                'entity_qmix_trans': 'Trans(QMIX)',
                'entity_vdn_trans': 'UPDeT',
                'entity_qmix_sparse_trans_scale': 'STrans(QMIX)',
                'entity_qmix_spotlight_contrastive_trans_scale': 'SDTrans(QMIX)',
                'entity_qmix_q_latent_inner_pattern_contrastive_trans_scale': 'Ours'}
    elif env_name == 'sc2':
        gap = 20
        name = {'iql': 'IQL',
                 'vdn': 'VDN',
                 'qmix': 'QMIX',
                 'qtran': 'QTRAN',
                 'qplex': 'QPLEX',
                 'owqmix': 'OWQMIX',
                 'cwqmix': 'CWQMIX',
                 'token_vdn_updet': 'UPDeT',
                 'token_qmix_updet': 'UPDeT(QMIX)',
                 'token_qmix_wise_trans': 'Trans(QMIX)',
                 'token_qmix_q_latent_pattern_contrastive_wise_trans_scale': 'Ours'}

    for idx, algo_id in enumerate(original_data[map_name]):
        color = idx
        if env_name == 'sc2':
            if idx == len(original_data[map_name]) - 1:
                color = 3
            elif idx == 3:
                color = len(original_data[map_name]) - 1
        sns.tsplot(time=original_data[map_name][algo_id]['x'][0::gap], data=original_data[map_name][algo_id]['y'][:, 0::gap],
                   linestyle=linestyle[0], condition=name[algo_id], color=sns.color_palette(n_colors=24)[color])

        # sns.tsplot(time=original_data[map_name][algo_id]['x'][0::gap], data=original_data[map_name][algo_id]['y'][:, 0::gap],
        #            linestyle=linestyle[0], condition=name[algo_id], color=sns.color_palette(n_colors=24)[color])

    plt.legend(loc='upper left', ncol=1, prop={'size': 14})
    # plt.legend(loc='upper center', ncol=2, handlelength=2,
    #            mode="expand", borderaxespad=0.1, prop={'size': 14})
    # plt.legend(loc='lower center', ncol=2, handlelength=2,
               # mode="expand", borderaxespad=0.1, prop={'size': 14})
    # plt.title(map_name, fontsize=fontsize)

    plt.xlabel(r'Total timesteps ($\times 10^6$)', fontsize=fontsize)

    if env_name == 'sc2':
        plt.xlim((-10000, 2000000 + 20000))
        plt.ylim((-0.05, 1.1))
        plt.xticks([0, 500000, 1000000, 1500000, 2000000], fontsize=fontsize)
        plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=fontsize)
        plt.ylabel('Median Test Win %', fontsize=fontsize, labelpad=10)
    elif env_name == 'sc2mt':
        plt.xlim((-10000, 4000000 + 50000))
        if map_name == '3-8sz_symmetric':
            plt.ylim((-0.05, 0.9))
            plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8], fontsize=fontsize)
        if map_name == '3-8MMM_symmetric':
            plt.ylim((-0.05, 0.5))
            plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4], fontsize=fontsize)
        if map_name == '3-8csz_symmetric':
            plt.ylim((-0.05, 0.9))
            plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8], fontsize=fontsize)
        if map_name == '5-11sz_symmetric':
            plt.ylim((-0.05, 0.5))
            plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4], fontsize=fontsize)
        if map_name == '5-11MMM_symmetric':
            plt.ylim((-0.05, 0.5))
            plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4], fontsize=fontsize)
        if map_name == '5-11csz_symmetric':
            plt.ylim((-0.05, 0.9))
            plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8], fontsize=fontsize)
        plt.xticks([0, 1000000, 2000000, 3000000, 4000000], fontsize=fontsize)
        # plt.xticks([0, 2000000, 4000000, 6000000, 8000000, 10000000], fontsize=fontsize)

        plt.ylabel('Median Test Win %', fontsize=fontsize, labelpad=10)
    elif env_name == 'particle':
        plt.xlim((-10000, 1000000 + 20000))
        plt.ylim((-0.1, 20))
        plt.xticks([0, 200000, 400000, 600000, 800000, 1000000], fontsize=fontsize)
        plt.yticks([0.0, 2.0, 4.0, 6.0, 8.0, 10.0], fontsize=fontsize)
        plt.ylabel('Average Return', fontsize=fontsize, labelpad=10)

    plt.savefig(os.path.join(EXP_PATH, 'results', 'fig', filename), format='pdf', bbox_inches='tight')
    plt.show()


seed_idx_list = {
    '3m': {},
    '5m_vs_6m': {'token_qmix_q_latent_pattern_contrastive_wise_trans_scale': [1,3,4,5,6], 'token_vdn_updet': [1,2,0,4,5]},
    '8m_vs_9m': {'token_qmix_q_latent_pattern_contrastive_wise_trans_scale': [2,0,4,5,6], 'token_vdn_updet': [1,2,0,3,4]},
    '10m_vs_11m': {},
    '25m': {},
    '2s3z': {},
    '3s5z': {},
    '3s_vs_3z': {},
    '3s_vs_5z': {'token_qmix_q_latent_pattern_contrastive_wise_trans_scale': [8,2,11,6,12]},
    'MMM': {},
    'MMM2': {'token_qmix_q_latent_pattern_contrastive_wise_trans_scale': [0,7,9,8,11]},
    '2c_vs_64zg': {},
    '6h_vs_8z': {},
    'tag_4_4_2': {},
    'tag_8_8_2': {},
    'tag_16_16_2': {},
    'htag_8_4_2': {},
    'htag_16_8_2': {},
    '3-8m_symmetric': {},
    '3-8sz_symmetric': {},
    '5-11sz_symmetric': {'entity_qmix_refil_imagine':[2,1,3,4,5],
                         'entity_qmix_q_latent_inner_pattern_contrastive_trans_scale':[0,4,3,5,2]},
    '3-8MMM_symmetric': {},
    '5-11MMM_symmetric': {},
    '3-8csz_symmetric': {},
    '5-11csz_symmetric': {},
}


def plot_sc2_baseline():
    map_list = ['5m_vs_6m', '8m_vs_9m', '10m_vs_11m', '2s3z', '3s5z', '3s_vs_3z', '3s_vs_5z', 'MMM', 'MMM2']
    # map_list = ['3s_vs_5z']
    algo_list = ['iql', 'vdn', 'qmix', 'qtran', 'qplex', 'owqmix', 'cwqmix', 'token_qmix_q_latent_pattern_contrastive_wise_trans_scale']
    # algo_list = ['coma', 'iql', 'vdn', 'qmix', 'qtran', 'qplex',
    #              'token_qmix_wise_attn', 'token_qmix_wise_trans',
    #              'token_vdn_wise_attn', 'token_vdn_wise_trans',
    #              'token_qmix_updet', 'token_vdn_updet']

    original_data = get_original_data('sc2', map_list, algo_list)
    for map_name in map_list:
        plot_reward_results(original_data, map_name, 'sc2', 'baseline')


def plot_sc2mt_baseline():
    # map_list = ['3-8m_symmetric', '3-8sz_symmetric', '3-8MMM_symmetric', '3-8csz_symmetric']
    map_list = ['3-8sz_symmetric', '3-8MMM_symmetric', '3-8csz_symmetric', '5-11sz_symmetric', '5-11MMM_symmetric', '5-11csz_symmetric']
    # map_list = ['5-11sz_symmetric']

    algo_list = ['entity_qmix_attn',
                 'entity_vdn_attn',
                 # 'entity_vdn_trans',
                 # 'entity_vdn_refil_imagine',
                 'entity_qmix_refil_imagine',
                 'entity_qmix_q_latent_inner_pattern_contrastive_trans_scale']

    # algo_list = ['entity_qmix_attn',
    #              'entity_vdn_attn',
    #              'entity_qmix_refil_imagine',
    #              'entity_qmix_trans',
    #              'entity_qmix_sparse_trans_scale',
    #              'entity_qmix_spotlight_contrastive_trans_scale',
    #              'entity_qmix_q_latent_inner_pattern_contrastive_trans_scale']
    # algo_list = ['entity_qmix_pattern_consistent_trans_scale', 'entity_qmix_pattern_contrastive_trans_scale',
    #              'entity_qmix_q_pattern_contrastive_trans_scale', 'entity_qmix_q_spotlight_contrastive_trans_scale',
    #              'entity_qmix_spotlight_consistent_trans_scale', 'entity_qmix_spotlight_contrastive_trans_scale']
    # algo_list = ['entity_qmix_attn', 'entity_qmix_trans',
    #              'entity_vdn_attn', 'entity_vdn_trans',
    #              'entity_qmix_refil_attn', 'entity_vdn_refil_attn']
    # algo_list = ['entity_qmix_refil_imagine', 'entity_qmix_refil_imagine_parallel',
    #              'entity_qmix_attn_scale', 'entity_qmix_trans_scale',
    #              'entity_qmix_sparse_attn_scale', 'entity_qmix_sparse_trans_scale']
    # algo_list = [
    #               'entity_qmix_refil_attn', 'entity_vdn_refil_attn',
    #               'entity_qmix_refil_imagine', 'entity_qmix_refil_imagine_parallel',
    #              'entity_qmix_sparse_attn_scale', 'entity_qmix_sparse_trans_scale']
    # algo_list = ['entity_qmix_attn', 'entity_qmix_trans',
    #              'entity_vdn_attn', 'entity_vdn_trans',
    #              'entity_qmix_refil_attn', 'entity_vdn_refil_attn',
    #              'entity_qmix_refil_imagine', 'entity_qmix_refil_imagine_parallel',
    #              'entity_qmix_attn_scale', 'entity_qmix_trans_scale',
    #              'entity_qmix_sparse_attn_scale', 'entity_qmix_sparse_trans_scale',
    #              'entity_qmix_pattern_consistent_trans_scale', 'entity_qmix_pattern_contrastive_trans_scale',
    #              'entity_qmix_q_pattern_contrastive_trans_scale', 'entity_qmix_q_spotlight_contrastive_trans_scale',
    #              'entity_qmix_spotlight_consistent_trans_scale', 'entity_qmix_spotlight_contrastive_trans_scale']

    original_data = get_original_data('sc2mt', map_list, algo_list)
    for map_name in map_list:
        plot_reward_results(original_data, map_name, 'sc2mt', 'baseline')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='sc2')
    args = parser.parse_args()

    if args.env_name == 'sc2':
        plot_sc2_baseline()
    elif args.env_name == 'sc2mt':
        plot_sc2mt_baseline()
