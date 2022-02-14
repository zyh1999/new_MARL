import torch as th
import time
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
sns.set()
sns.set_style("darkgrid", {"axes.facecolor": "#f0f0f7"})
linestyle = [':', '--', '-.', '-']
fontsize = 20

#EXP_PATH = os.path.join(os.environ['NFS_HOME'], 'code/pymarl')


attn_feats = []


# def hook(model, input, output):
#     out = output[1].view(4, -1, output[1].size(1), output[1].size(2))
#     out = out.permute(1, 0, 2, 3).contiguous()
#     attn_feats.append(out.clone().detach().cpu().numpy())


def hook(model, input, output):
    out = output[1].view(4, -1, output[1].size(1), output[1].size(2))
    out = out.permute(1, 0, 2, 3).contiguous()
    attn_feats.append(out.clone().detach().cpu().numpy())


def statistic_attn(args, runner, learner):
    algo_name = args.name
    map_name = args.env_args['map_name']

    learner.mac.agent.transformer.transformer_blocks[0].attn.attention.register_forward_hook(hook)
    run(runner, test_mode=True)
    runner.close_env()

    n_heads = attn_feats[0].shape[1]
    plt.figure(figsize=(5, 20))

    n_steps = 20
    for i in range(n_steps):
        for j in range(n_heads):
            plt.subplot(n_steps, n_heads, i * n_heads + j + 1)
            attn_tmp = np.zeros((runner.env.n_agents, runner.env.n_agents + runner.env.n_enemies))
            attn_tmp[:runner.env.n_agents, :runner.env.n_agents] = attn_feats[i][0, j][:runner.env.n_agents, :runner.env.n_agents]
            attn_tmp[:runner.env.n_agents, runner.env.n_agents:] = attn_feats[i][0, j][:runner.env.n_agents, runner.env.max_n_agents:runner.env.max_n_agents + runner.env.n_enemies]

            # attn_tmp = np.zeros((runner.env.n_agents + runner.env.n_enemies, runner.env.n_agents + runner.env.n_enemies))
            # attn_tmp[:runner.env.n_agents, :runner.env.n_agents] = attn_feats[i][0, j][:runner.env.n_agents, :runner.env.n_agents]
            # attn_tmp[runner.env.n_agents:, :runner.env.n_agents] = attn_feats[i][0, j][runner.env.max_n_agents:runner.env.max_n_agents + runner.env.n_enemies, :runner.env.n_agents]
            # attn_tmp[:runner.env.n_agents, runner.env.n_agents:] = attn_feats[i][0, j][:runner.env.n_agents, runner.env.max_n_agents:runner.env.max_n_agents + runner.env.n_enemies]
            # attn_tmp[runner.env.n_agents:, runner.env.n_agents:] = attn_feats[i][0, j][runner.env.max_n_agents:runner.env.max_n_agents + runner.env.n_enemies, runner.env.max_n_agents:runner.env.max_n_agents + runner.env.n_enemies]
            sns.heatmap(attn_tmp, cmap=sns.cubehelix_palette(as_cmap=True, gamma=0.8), linewidths=.5, cbar=False)
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')
            plt.tight_layout()
    plt.show()

    # learner.mac.agent.attn.attention.register_forward_hook(hook)
    # while 1:
    #     run(runner, test_mode=True)
    # runner.save_replay()
    # runner.close_env()

    # n_steps = len(attn_feats)
    # n_agents = attn_feats[0].shape[0]
    # n_heads = attn_feats[0].shape[1]
    # n_tokens = attn_feats[0].shape[2]
    # plt.figure(figsize=(5, 100))

    # for i in range(n_steps):
    #     plt.subplot(n_steps, 1, i + 1)
    #     sns.heatmap(np.vstack([np.hstack([attn_feats[0][k, j] for j in range(n_heads)]) for k in range(n_agents)]), cmap=sns.cubehelix_palette(as_cmap=True, gamma=0.8), linewidths=.5, cbar=False)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.axis('off')
    #     plt.tight_layout()
    # plt.show()

    # n_steps = len(attn_feats)
    # n_agents = attn_feats[0].shape[0]
    # n_heads = attn_feats[0].shape[1]
    # n_tokens = attn_feats[0].shape[2]
    # plt.figure(figsize=(5, 100))

    # n_steps = 20
    # for k in range(n_steps):
    #     for i in range(n_agents):
    #         for j in range(n_heads):
    #             plt.subplot(n_steps * n_agents, n_heads, k * n_agents * n_heads + i * n_heads + j + 1)
    #             # sns.heatmap(1 - attn_agent_data[0, 0, 0, i - 1], vmin=0.5, vmax=1, cmap='rocket', linewidths=.5)
    #             sns.heatmap(attn_feats[k][i, j], cmap=sns.cubehelix_palette(as_cmap=True, gamma=0.8), linewidths=.5, cbar=False)
    #             plt.xticks([])
    #             plt.yticks([])
    #             plt.axis('off')
    #             plt.tight_layout()
    # plt.show()
    #
    # plt.savefig(os.path.join(EXP_PATH, 'results', 'fig', 'test.pdf'), format='pdf', bbox_inches='tight')

    pass


def run(runner, test_mode=False):
    runner.reset()

    terminated = False
    episode_return = 0
    runner.mac.init_hidden(batch_size=runner.batch_size)

    while not terminated:

        if runner.args.evaluate and runner.args.render:
            runner.env.render()
            time.sleep(0.2)

        pre_transition_data = runner._get_pre_transition_data()

        runner.batch.update(pre_transition_data, ts=runner.t)

        # Pass the entire batch of experiences up till now to the agents
        # Receive the actions for each agent at this timestep in a batch of size 1
        actions = runner.mac.select_actions(runner.batch, t_ep=runner.t, t_env=runner.t_env, test_mode=test_mode)

        reward, terminated, env_info = runner.env.step(actions[0])
        episode_return += reward

        post_transition_data = {
            "actions": actions,
            "reward": [(reward,)],
            "terminated": [(terminated != env_info.get("episode_limit", False),)],
        }

        runner.batch.update(post_transition_data, ts=runner.t)

        runner.t += 1

        # n_agents = attn_feats[-1].shape[0]
        # n_heads = attn_feats[-1].shape[1]
        # plt.figure(figsize=(5, 5))
        # for i in range(n_agents):
        #     for j in range(n_heads):
        #         plt.subplot(n_agents, n_heads, i * n_heads + j + 1)
        #         sns.heatmap(attn_feats[-1][i, j], cmap=sns.cubehelix_palette(as_cmap=True, gamma=0.8), linewidths=.5, cbar=False)
        #         plt.xticks([])
        #         plt.yticks([])
        #         plt.axis('off')
        #         plt.tight_layout()
        # plt.show()

        # n_heads = attn_feats[-1].shape[1]
        # plt.figure(figsize=(5, 2))
        # for i in range(n_heads):
        #     plt.subplot(1, n_heads, i + 1)
        #     attn_tmp = np.zeros((runner.env.n_agents + runner.env.n_enemies, runner.env.n_agents + runner.env.n_enemies))
        #     attn_tmp[:runner.env.n_agents, :runner.env.n_agents] = attn_feats[-1][0, i][:runner.env.n_agents, :runner.env.n_agents]
        #     attn_tmp[runner.env.n_agents:, :runner.env.n_agents] = attn_feats[-1][0, i][runner.env.max_n_agents:runner.env.max_n_agents + runner.env.n_enemies, :runner.env.n_agents]
        #     attn_tmp[:runner.env.n_agents, runner.env.n_agents:] = attn_feats[-1][0, i][:runner.env.n_agents, runner.env.max_n_agents:runner.env.max_n_agents + runner.env.n_enemies]
        #     attn_tmp[runner.env.n_agents:, runner.env.n_agents:] = attn_feats[-1][0, i][runner.env.max_n_agents:runner.env.max_n_agents + runner.env.n_enemies, runner.env.max_n_agents:runner.env.max_n_agents + runner.env.n_enemies]
        #     sns.heatmap(attn_tmp, cmap=sns.cubehelix_palette(as_cmap=True, gamma=0.8), linewidths=.5, cbar=False)
        #     plt.xticks([])
        #     plt.yticks([])
        #     plt.axis('off')
        #     plt.tight_layout()
        # plt.show()

    last_data = runner._get_pre_transition_data()
    runner.batch.update(last_data, ts=runner.t)

    # Select actions in the last stored state
    actions = runner.mac.select_actions(runner.batch, t_ep=runner.t, t_env=runner.t_env, test_mode=test_mode)
    runner.batch.update({"actions": actions}, ts=runner.t)

    cur_stats = {}
    cur_returns = []
    log_prefix = "test/" if test_mode else "run/"
    cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
    cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
    cur_stats["ep_length"] = runner.t + cur_stats.get("ep_length", 0)

    cur_returns.append(episode_return)

    if runner.args.evaluate:
        runner.logger.console_logger.info("episode_return: {:.4f}".format(episode_return))
    if test_mode and (len(runner.test_returns) == runner.args.test_nepisode):
        runner._log(cur_returns, cur_stats, log_prefix)

    return runner.batch




def statistic_q(args, runner, learner):
    method = args.checkpoint_path.split('/')[9]
    original_map = args.checkpoint_path.split('/')[8]
    current_map = args.env_args['map_name']

    file_name = method + '_' + original_map + '_to_' + current_map + '.pdf'

    with th.no_grad():
        episode_batch = runner.run(test_mode=True)
        q, sum_q, mix_q, returns = test(learner, episode_batch)
        plot(file_name, q, sum_q, mix_q, returns)


def test(learner, batch):
    rewards = batch["reward"][:, :-1]
    actions = batch["actions"][:, :-1]
    terminated = batch["terminated"][:, :-1].float()
    mask = batch["filled"][:, :-1].float()
    mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

    mac_out = []
    learner.mac.init_hidden(batch.batch_size)
    for t in range(batch.max_seq_length):
        agent_outs = learner.mac.forward(batch, t=t)
        mac_out.append(agent_outs)
    mac_out = th.stack(mac_out, dim=1)

    chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)
    if learner.args.mixer not in ['ext_qmix', 'latent_qmix']:
        mix_chosen_action_qvals = learner.mixer(chosen_action_qvals, batch["state"][:, :-1])
    else:
        mix_chosen_action_qvals = learner.mixer(chosen_action_qvals, batch["state"][:, :-1], learner.args.enemy_num, learner.args.ally_num)

    returns = rewards
    for t in range(batch.max_seq_length - 1):
        returns[:, t, :] = rewards[:, t:, :].sum(1, True)

    q = chosen_action_qvals.squeeze().cpu().numpy()
    sum_q = chosen_action_qvals.sum(2, True).squeeze().cpu().numpy()
    mix_q = mix_chosen_action_qvals.squeeze().cpu().numpy()
    returns = returns.squeeze().cpu().numpy()

    return q, sum_q, mix_q, returns


def plot(file_name, q, sum_q, mix_q, returns):
    label = ['sum', 'mix', 'return', 'agent']

    plt.figure(figsize=(7, 2.5))

    sns.tsplot(time=[i for i in range(len(sum_q))], data=sum_q, linestyle=linestyle[0], condition=label[0], color=sns.color_palette()[0])
    sns.tsplot(time=[i for i in range(len(mix_q))], data=mix_q, linestyle=linestyle[1], condition=label[1], color=sns.color_palette()[1])
    sns.tsplot(time=[i for i in range(len(returns))], data=returns, linestyle=linestyle[2], condition=label[2], color=sns.color_palette()[2])

    for a_id in range(len(q[0, :])):
        sns.tsplot(time=[i for i in range(len(q[:, a_id]))], data=q[:, a_id], linestyle=linestyle[3], condition=label[3] + str(a_id), color=sns.color_palette()[3])

    plt.legend(loc='upper right', ncol=2, fontsize=14)
    plt.xlim((-2, 71))
    plt.ylim((-10, 35))
    plt.yticks(fontsize=fontsize)

    plt.title(file_name[:-4], fontsize=fontsize)

    plt.ylabel('Q value', fontsize=fontsize, labelpad=10)
    plt.xlabel(r'Total timesteps', fontsize=fontsize)

    plt.savefig(os.path.join(EXP_PATH, 'results', 'fig', file_name), format='pdf', bbox_inches='tight')

    plt.show()