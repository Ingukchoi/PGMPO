import json
import random

from torch.distributions.categorical import Categorical
import sys
import numpy as np
import torch
import copy

"""
    agent utils
"""


def sample_action(p):
    """
        sample an action by the distribution p
    :param p: this distribution with the probability of choosing each action
    :return: an action sampled by p
    """
    dist = Categorical(p)
    s = dist.sample()  # index
    return s, dist.log_prob(s)


def eval_actions(p, actions):
    """
    :param p: the policy
    :param actions: action sequences
    :return: the log probability of actions and the entropy of p
    """
    softmax_dist = Categorical(p.squeeze())
    ret = softmax_dist.log_prob(actions).reshape(-1)
    entropy = softmax_dist.entropy().mean()
    return ret, entropy


def greedy_select_action(p):
    prob, index = torch.max(p, dim=1)
    return index, torch.log(prob)


def min_element_index(array):
    """
    :param array: an array with numbers
    :return: Index set corresponding to the minimum element of the array
    """
    min_element = np.min(array)
    candidate = np.where(array == min_element)
    return candidate


def max_element_index(array):
    """
    :param array: an array with numbers
    :return: Index set corresponding to the maximum element of the array
    """
    max_element = np.max(array)
    candidate = np.where(array == max_element)
    return candidate


def available_mch_list_for_job(chosen_job, env):
    """
    :param chosen_job: the selected job
    :param env: the production environment
    :return: the machines which can immediately process the chosen job
    """
    mch_state = ~env.candidate_process_relation[0, chosen_job]
    available_mch_list = np.where(mch_state == True)[0]
    mch_free_time = env.mch_free_time[0][available_mch_list]
    job_free_time = env.candidate_free_time[0][chosen_job]
    # case1 eg:
    # JF: 50
    # MchF: 55 60 65 70
    if (job_free_time < mch_free_time).all():
        chosen_mch_list = available_mch_list[min_element_index(mch_free_time)]
    # case2 eg:
    # JF: 50
    # MchF: 35 40 55 60
    else:
        chosen_mch_list = available_mch_list[np.where(mch_free_time <= job_free_time)]

    return chosen_mch_list


def heuristic_select_action(method, env):
    """
    :param method: the name of heuristic method
    :param env: the environment
    :return: the action selected by the heuristic method

    here are heuristic methods selected for comparison:

    FIFO: First in first out
    MOR(or MOPNR): Most operations remaining
    SPT: Shortest processing time
    MWKR: Most work remaining
    """
    chosen_job = -1
    chosen_mch = -1

    job_state = (env.mask[0] == 0)

    process_job_state = (env.candidate_free_time[0] <= env.next_schedule_time[0])
    job_state = process_job_state & job_state

    available_jobs = np.where(job_state == True)[0]
    available_ops = env.candidate[0][available_jobs]

    if method == 'FIFO':
        # selecting the earliest ready candidate operation
        candidate_free_time = env.candidate_free_time[0][available_jobs]
        chosen_job_list = available_jobs[min_element_index(candidate_free_time)]
        chosen_job = np.random.choice(chosen_job_list)

        # select the earliest ready machine
        mch_state = ~env.candidate_process_relation[0, chosen_job]
        available_mchs = np.where(mch_state == True)[0]
        mch_free_time = env.mch_free_time[0][available_mchs]
        chosen_mch_list = available_mchs[min_element_index(mch_free_time)]
        chosen_mch = np.random.choice(chosen_mch_list)

    elif method == 'MOR':
        remain_ops = env.op_match_job_left_op_nums[0][available_ops]
        chosen_job_list = available_jobs[max_element_index(remain_ops)]
        chosen_job = np.random.choice(chosen_job_list)

        # select a machine which can immediately process the chosen job
        chosen_mch_list = available_mch_list_for_job(chosen_job, env)
        chosen_mch = np.random.choice(chosen_mch_list)

    elif method == 'SPT':

        temp_pt = copy.deepcopy(env.candidate_pt[0])
        temp_pt[env.dynamic_pair_mask[0]] = float("inf")
        pt_list = temp_pt.reshape(-1)

        action_list = np.where(pt_list == np.min(pt_list))[0]

        action = np.random.choice(action_list)
        return action

    elif method == 'MWKR':
        job_remain_work_list = env.op_match_job_remain_work[0][available_ops]

        chosen_job = available_jobs[np.random.choice(max_element_index(job_remain_work_list)[0])]

        # select a machine which can immediately process the chosen job
        chosen_mch_list = available_mch_list_for_job(chosen_job, env)
        chosen_mch = np.random.choice(chosen_mch_list)

    else:
        print(f'Error From rule select: undefined method {method}')
        sys.exit()

    if chosen_job == -1 or chosen_mch == -1:
        print(f'Error From choosing action: choose job {chosen_job}, mch {chosen_mch}')
        sys.exit()

    action = chosen_job * env.number_of_machines + chosen_mch
    return action


"""
    common utils
"""


def save_default_params(config):
    """
        save parameters in the config
    :param config: a package of parameters
    :return:
    """
    with open('./config_default.json', 'wt') as f:
        json.dump(vars(config), f, indent=4)
    print("successfully save default params")


def nonzero_averaging(x):
    """
        remove zero vectors and then compute the mean of x
        (The deleted nodes are represented by zero vectors)
    :param x: feature vectors with shape [sz_b, node_num, d]
    :return:  the desired mean value with shape [sz_b, d]
    """
    b = x.sum(dim=-2)
    y = torch.count_nonzero(x, dim=-1)
    z = (y != 0).sum(dim=-1, keepdim=True)
    p = 1 / z
    p[z == 0] = 0
    return torch.mul(p, b)


def strToSuffix(str):
    if str == '':
        return str
    else:
        return '+' + str


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

import torch
from torch_geometric.data import HeteroData

def graph_generator_v2(op_feature, mc_feature, relation, pair_feature, verbose=True):
    """
    Bipartite graph (op -> mch) only: machine이 쿼리/target인 방향만 유지
    """
    device = op_feature.device
    B, N, d_o = op_feature.shape  # N = J*M
    _, M, d_m = mc_feature.shape
    d_p = pair_feature.shape[-1]

    # 1) 유효 엣지 마스크
    relation_bool = relation.bool()
    op_has_connection  = relation_bool.any(dim=2)   # [B, N]
    mch_has_connection = relation_bool.any(dim=1)   # [B, M]
    edge_mask = relation_bool & op_has_connection.unsqueeze(2) & mch_has_connection.unsqueeze(1)

    # 2) 엣지 인덱스 (op -> mch 만 유지)
    edge_indices = edge_mask.nonzero(as_tuple=False)
    hd = HeteroData()

    # 노드 피처/배치 인덱스
    hd['op'].x  = op_feature.reshape(-1, d_o)
    hd['mch'].x = mc_feature.reshape(-1, d_m)
    hd['op'].batch  = torch.arange(B, device=device).repeat_interleave(N)
    hd['mch'].batch = torch.arange(B, device=device).repeat_interleave(M)

    if edge_indices.numel() == 0:
        if verbose:
            print("Warning: The entire batch has no edges!")
        empty_index = torch.empty((2, 0), dtype=torch.long, device=device)
        hd['op', 'assign', 'mch'].edge_index = empty_index
        # 역방향 관계는 생성하지 않음
        return hd

    batch_idx     = edge_indices[:, 0]
    op_idx_local  = edge_indices[:, 1]
    mch_idx_local = edge_indices[:, 2]

    # 3) 전역 인덱스(op -> mch)
    op_idx_global  = op_idx_local  + batch_idx * N
    mch_idx_global = mch_idx_local + batch_idx * M

    # 4) edge_attr (유효 엣지만)
    edge_attr = pair_feature[batch_idx, op_idx_local, mch_idx_local]
    if verbose:
        zero_edges = (edge_attr.abs().sum(dim=1) == 0).sum().item()
        if zero_edges > 0:
            print(f"Warning: {zero_edges}/{len(edge_attr)} edges have all-zero features")

    # 5) 관계(op -> mch)만 할당
    hd['op', 'assign', 'mch'].edge_index = torch.stack([op_idx_global, mch_idx_global], dim=0)
    hd['op', 'assign', 'mch'].edge_attr  = edge_attr

    # 역방향('mch','recruit','op')는 생성하지 않음
    return hd


if __name__ == '__main__':
    print('123')
