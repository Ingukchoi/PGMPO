import time
import os
from common_utils import *
from params import configs
from tqdm import tqdm
from data_utils import pack_data_from_config
from common_utils import setup_seed
from model.main_model import DANIEL as Model
os.environ["CUDA_VISIBLE_DEVICES"] = configs.device_id
import torch

device = torch.device(configs.device)
test_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
model = Model(configs)


def test_sampling_strategy(config, data_set, model_path, sample_times, seed):
    """
        test the model on the given data using the sampling strategy
    :param data_set: test data
    :param model_path: the path of the model file
    :param seed: the seed for testing
    :return: the test results including the makespan and time
    """
    setup_seed(seed)
    test_result_list = []
    model.load_state_dict(torch.load(model_path, map_location='cuda'))
    model.eval()

    bit_dim = config.bit_dim
    n_j = data_set[0][0].shape[0]
    n_op, n_m = data_set[1][0].shape
    from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
    env = FJSPEnvForSameOpNums(n_j, n_m)

    for i in tqdm(range(len(data_set[0])), file=sys.stdout, desc="progress", colour='blue'):
        # copy the testing environment
        JobLength_dataset = np.tile(np.expand_dims(data_set[0][i], axis=0), (sample_times, 1))
        OpPT_dataset = np.tile(np.expand_dims(data_set[1][i], axis=0), (sample_times, 1, 1))

        state = env.set_initial_data(JobLength_dataset, OpPT_dataset)
        t1 = time.time()

        bit_seed = torch.arange(2 ** bit_dim, device=device)
        bit_vectors = ((bit_seed.unsqueeze(1) >> torch.arange(bit_dim - 1, -1, -1, device=device)) & 1).float()
        num = bit_vectors.size(0)

        if sample_times <= num:
            idx = torch.randperm(num, device=device)[:sample_times]
        else:
            idx = torch.randint(num, (sample_times,), device=device)
        bit_vector = bit_vectors[idx] 

        while True:
            with torch.no_grad():
                pi = model(fea_j=state.fea_j_tensor,  # [100, N, 10]
                                   op_mask=state.op_mask_tensor,  # [100, N, N]
                                   candidate=state.candidate_tensor,  # [100, J]
                                   fea_m=state.fea_m_tensor,  # [100, M, 8]
                                   mch_mask=state.mch_mask_tensor,  # [100, M, M]
                                   comp_idx=state.comp_idx_tensor,  # [100, M, M, J]
                                   dynamic_pair_mask=state.dynamic_pair_mask_tensor,  # [100, J, M]
                                   fea_pairs=state.fea_pairs_tensor, # [100, J, M]
                                   bit_vector = bit_vector) #bit_vector = bit_vector
                
            action_envs, _ = sample_action(pi)
            state, _, done = env.step(action_envs.cpu().numpy())

            if done.all():
                break

        t2 = time.time()

        best_makespan = np.min(env.current_makespan)
        test_result_list.append([best_makespan, t2 - t1])
    return np.array(test_result_list)


def main(config):
    """
        test the trained model following the config and save the results
    :param flag_sample: whether using the sampling strategy
    """
    setup_seed(config.seed_test)
    if not os.path.exists('./test_results'):
        os.makedirs('./test_results')

    # collect the path of test models
    test_model = []

    for model_name in config.test_model:
        test_model.append((f'./trained_network/{config.model_source}/{model_name}.pth', model_name))

    # collect the test data
    test_data = pack_data_from_config(config.data_source, config.test_data)
    model_prefix = "DANIELS"

    for data in test_data:
        print("-" * 25 + "Test Learned Model" + "-" * 25)
        print(f"test data name: {data[1]}")
        print(f"test mode: {model_prefix}")
        save_direc = f'./test_results/{config.data_source}/{data[1]}'
        if not os.path.exists(save_direc):
            os.makedirs(save_direc)

        for model in test_model:
            save_path = save_direc + f'/Result_{model_prefix}+{model[1]}_{data[1]}.npy'
            if (not os.path.exists(save_path)) or config.cover_flag:
                print(f"Model name : {model[1]}")
                print(f"data name: ./data/{config.data_source}/{data[1]}")
                print("Test mode: Sample")
                save_result = test_sampling_strategy(config, data[0], model[0], config.sample_times, config.seed_test)
                print(f"makespan(sampling): ", save_result[:, 0].mean())
                print(f"time: ", save_result[:, 1].mean())
                np.save(save_path, save_result)


if __name__ == '__main__':
    main(configs)