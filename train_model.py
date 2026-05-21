from common_utils import *
from params import configs
from tqdm import tqdm
from data_utils import load_data_from_files, CaseGenerator, SD2_instance_generator
from common_utils import strToSuffix, setup_seed
from fjsp_env_various_op_nums import FJSPEnvForVariousOpNums
import os
import random
import time
import sys
from model.main_model import DANIEL as Model
from torch.optim import Adam as Optimizer

str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
os.environ["CUDA_VISIBLE_DEVICES"] = configs.device_id
import torch

device = torch.device(configs.device)


class Trainer:
    def __init__(self, config):

        self.lr = config.lr
        self.n_j = config.n_j
        self.n_m = config.n_m
        self.low = config.low
        self.high = config.high
        self.op_per_job_min = int(0.8 * self.n_m)
        self.op_per_job_max = int(1.2 * self.n_m)
        self.data_source = config.data_source
        self.config = config
        self.max_updates = config.max_updates
        self.save_timestep = config.save_timestep
        self.num_envs = config.num_envs
        self.bit_dim = config.bit_dim
        self.accmulation_step = config.accmulation_step
        self.num_problems = config.num_problems//config.accmulation_step

        if not os.path.exists(f'./trained_network/{self.data_source}'):
            os.makedirs(f'./trained_network/{self.data_source}')
        if not os.path.exists(f'./train_log/{self.data_source}'):
            os.makedirs(f'./train_log/{self.data_source}')

        if device.type == 'cuda':
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        if self.data_source == 'SD1':
            self.data_name = f'{self.n_j}x{self.n_m}'
        elif self.data_source == 'SD2':
            self.data_name = f'{self.n_j}x{self.n_m}{strToSuffix(config.data_suffix)}'

        self.test_data_path = f'./data/{self.data_source}/{self.data_name}'
        self.model_name = f'{self.data_name}{strToSuffix(config.model_suffix)}'

        # seed
        self.seed_train = config.seed_train
        self.seed_test = config.seed_test
        self.test_data = load_data_from_files(self.test_data_path)
        self.total_batch = self.num_problems * self.num_envs
        self.env = FJSPEnvForVariousOpNums(self.n_j, self.n_m)

        self.model = Model(config)
        self.optimizer = Optimizer(self.model.parameters(), self.lr)

    def train(self):
        setup_seed(self.seed_train)
        self.log = []
        self.record = float('inf')

        print("-" * 25 + "Training Setting" + "-" * 25)
        print(f"source : {self.data_source}")
        print(f"model name :{self.model_name}")
        print(f"num_problems : {self.num_problems}, accmulation_step : {self.accmulation_step}, num_envs : {self.num_envs}, total_batch : {self.total_batch}")
        print("\n")

        self.train_st = time.time()

        bit_seed = torch.arange(2 ** self.bit_dim, device=device)
        bit_vectors = ((bit_seed.unsqueeze(1) >> torch.arange(self.bit_dim - 1, -1, -1, device=device)) & 1).float()
        bit_vectors_expanded = bit_vectors.repeat(self.num_problems, 1)

        for i_update in tqdm(range(self.max_updates), file=sys.stdout, desc="progress", colour='blue'):
            ep_st = time.time()

            dataset_job_length, dataset_op_pt = self.sample_training_instances()
            state = self.env.set_initial_data(dataset_job_length, dataset_op_pt)

            max_steps = int(self.env.max_number_of_ops)
            log_prob_buffer = torch.zeros(self.total_batch, max_steps, device=device)
            step_idx = 0

            while True:
                batch_idx = torch.from_numpy(~self.env.done_flag).to(device)
                incomplete_indices = batch_idx.nonzero(as_tuple=True)[0]

                pi_envs= self.model(
                    fea_j=state.fea_j_tensor[batch_idx],
                    op_mask=state.op_mask_tensor[batch_idx],
                    candidate=state.candidate_tensor[batch_idx],
                    fea_m=state.fea_m_tensor[batch_idx],
                    mch_mask=state.mch_mask_tensor[batch_idx],
                    comp_idx=state.comp_idx_tensor[batch_idx],
                    dynamic_pair_mask=state.dynamic_pair_mask_tensor[batch_idx],
                    fea_pairs=state.fea_pairs_tensor[batch_idx],
                    bit_vector=bit_vectors_expanded[batch_idx],
                )

                action_envs, action_logprob_envs = sample_action(pi_envs)
                log_prob_buffer[incomplete_indices, step_idx] = action_logprob_envs
                step_idx += 1

                state, _, done = self.env.step(actions=action_envs.cpu().numpy())

                if done.all():
                    break

            prob_list = log_prob_buffer[:, :step_idx] 
            makespan = torch.from_numpy(self.env.current_makespan).to(device) 

            makespan_2d = makespan.view(self.num_problems, self.num_envs)
            prob_list_3d = prob_list.view(self.num_problems, self.num_envs, -1) 

            un_log_prob = prob_list_3d.sum(dim=2)
            log_prob = un_log_prob / prob_list_3d.size(2)

            reward_2d = -makespan_2d

            best_idx = reward_2d.argmax(dim=1)
            anchor = reward_2d.gather(1, best_idx.unsqueeze(1)) 
            anchor_log_prob = log_prob.gather(1, best_idx.unsqueeze(1)) 
            preference = anchor > reward_2d

            epsilon = 1e-8
            reward_ratio = (reward_2d + epsilon) / anchor
            log_prob_pair = anchor_log_prob - log_prob 
            pf_log = torch.log(torch.sigmoid(reward_ratio * log_prob_pair) + epsilon) 

            # self-comparison mask
            mask = torch.ones_like(preference, dtype=torch.bool)
            mask.scatter_(1, best_idx.unsqueeze(1), False)

            pf_log_sel = pf_log[mask]
            preference_sel = preference[mask].float()

            loss_mean = -torch.mean(pf_log_sel * preference_sel)

            loss = loss_mean / self.accmulation_step
            loss.backward()

            if (i_update + 1) % self.accmulation_step == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            mean_makespan_all_env = makespan_2d.min(dim=1).values.mean().item()

            self.log.append([i_update, mean_makespan_all_env])
            if (i_update + 1) % self.save_timestep == 0:
                self.save_model()

            ep_et = time.time()

            # tqdm.write(
            #     'Episode: {}\t makespan(avg best): {:.2f}\t Mean_loss: {:.8f},  training time: {:.2f}'.format(
            #         i_update + 1, mean_makespan_all_env, loss_mean.item(), ep_et - ep_st))

        self.train_et = time.time()
        self.save_training_log()

    def save_training_log(self):
        file_writing_obj = open(f'./train_log/{self.data_source}/' + 'reward_' + self.model_name + '.txt', 'w')
        file_writing_obj.write(str(self.log))

        file_writing_obj3 = open(f'./train_time.txt', 'a')
        file_writing_obj3.write(
            f'model path: ./DANIEL_FJSP/trained_network/{self.data_source}/{self.model_name}\t\ttraining time: '
            f'{round((self.train_et - self.train_st), 2)}\t\t local time: {str_time}\n')

    def sample_training_instances(self):
        dataset_JobLength = []
        dataset_OpPT = []

        for _ in range(self.num_problems):
            if self.data_source == 'SD1':
                prepare_JobLength = [random.randint(self.op_per_job_min, self.op_per_job_max) for _ in range(self.n_j)]
                case = CaseGenerator(self.n_j, self.n_m, self.op_per_job_min, self.op_per_job_max,
                                     nums_ope=prepare_JobLength, path='./test', flag_doc=False)
                JobLength, OpPT, _ = case.get_case()
            else:
                JobLength, OpPT, _ = SD2_instance_generator(self.config)

            for _ in range(self.num_envs):
                dataset_JobLength.append(JobLength)
                dataset_OpPT.append(OpPT)

        return dataset_JobLength, dataset_OpPT

    def save_model(self):
        torch.save(self.model.state_dict(), f'./trained_network/{self.data_source}'
                                            f'/{self.model_name}.pth')

    def load_model(self):
        model_path = f'./trained_network/{self.data_source}/{self.model_name}.pth'
        self.model.load_state_dict(torch.load(model_path, map_location='cuda'))


def main():
    trainer = Trainer(configs)
    trainer.train()


if __name__ == '__main__':
    main()
