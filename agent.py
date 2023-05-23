import torch.nn.functional as F
import numpy as np
import torch

from timeit import default_timer as timer


# normal_attack, charged_attack, element_skill, element_bust
ACTIONS = [np.array([1,0,0,0]),
           np.array([0,1,0,0]),
           np.array([0,0,1,0]),
           np.array([0,0,0,1])]

TRAINING_EPOCH_NUM = 10000
gamma         = 0.98
batch_size    = 64


class AGENT:
    def __init__(self, character1, character2, boss, buff_table, q, q_target, device, memory, optimizer, is_upload=False):

        self.device = device
        self.ACTIONS = ACTIONS
        self.character1 = character1
        self.character2 = character2
        self.boss = boss
        self.buff_table = buff_table
        self.q = q.to(self.device)
        self.q_target = q_target.to(self.device)
        self.memory = memory
        self.optimizer = optimizer
        self.state = [0, 0, 0, 0,
                      0, 0, 0, 0,
                      0]
        self.loss_avg = 0

        if is_upload:
            qlearning_results = np.load('./result/qlearning.npz', allow_pickle=True)
            self.q = qlearning_results['Q']
            self.q_target = qlearning_results['Q_TARGET']
            self.memory = qlearning_results['MEMORY']
            self.optimizer = qlearning_results['OPTIMIZER']


    def initialize_episode(self):
           state = [0, 0, 0, 0,
                    0, 0, 0, 0,
                    self.boss.hp]
           return state


    def train(self, q, q_target, memory, optimizer):
        loss_list = torch.tensor(0, device = self.device)
        num = 10


        for i in range(num):
            s, a, r, s_prime, done_mask = memory.sample(64)

            q_out = (self.q(s)).to(self.device)
            q_a = q_out.gather(1, a)
            max_q_prime = self.q_target(s_prime).max(1)[0].unsqueeze(1)
            target = r + gamma * max_q_prime * done_mask
            loss = F.smooth_l1_loss(q_a, target)
            loss_list = loss_list + loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        loss_avg = loss_list / num


        return loss_avg



#    def state_floor(self, state):
#        for i in range(len(state)):
#            if (i%2) == 1:
#                state[i] = round(state[i], 2)
#        return state



    def Q_learning(self, discount=1.0, alpha=0.01, max_seq_len=9999,
                             decay_period=20000, decay_rate=0.9):

        total_time = 0
        avg_delay = 0
        action_history = []
        action_history2 = []
        delay_history = []
        delay_history2 = []
        time_history = 999999

        for episode in range(TRAINING_EPOCH_NUM):
            epsilon = max(0.01, 0.3 - 0.02 * (episode / 1000))
            state = torch.tensor(self.initialize_episode(), dtype = torch.float).to(self.device)
            done = False
            timeout = False
            seq_len = 0
            cum_reward = 0
            cum_delay = 0
            action = 0


            while not (done or timeout):
                # Next state and action generation
                action_check = False
                while not action_check:
                    action = self.get_action(state, epsilon)
                    if state[action] == 0:
                        action_check = True
                if action < 4: ## 첫번째 캐릭터를 선택한 경우
                    char_num = 0
                    character = self.character1
                else: ## 두번째 캐릭터를 선택한 경우
                    char_num = 1
                    character = self.character2
                next_state, reward, delay = self.boss.interaction(state, action, char_num, character, self.buff_table)
                if self.boss.is_goal(next_state):
                    delay = 0
                cum_delay = cum_delay + delay
                action_history2.append(action)
                delay_history2.append(delay)

                if self.boss.is_goal(next_state):
                    done_mask = torch.tensor([0.0], device = self.device)
                else:
                    done_mask = torch.tensor([1.0], device = self.device)

                self.memory.put((state[:8], action, reward, next_state[:8], done_mask))
                state = next_state  # agent를 다음 state로 이동
                cum_reward = cum_reward + reward
                seq_len += 1
                if (seq_len >= max_seq_len):
                    timeout = True

                done = self.boss.is_goal(state)


            if cum_delay <= time_history:
                action_history = action_history2.copy()
                time_history = cum_delay
            action_history2 = []
            delay_history2 = []

            total_time = total_time + cum_delay

            print("episode : ", episode)

            if self.memory.size() > 1000:
                self.loss_avg = self.train(self.q, self.q_target, self.memory, self.optimizer)


            if (episode % 100 == 0) and (episode > 0):
                avg_delay = total_time / 100
                print("Num of episodes = {:}, epsilon={:.4f}, loss avg={: .8f}".format(episode, epsilon, self.loss_avg))
                print("avg time : ", avg_delay)
                total_time = 0

            if (episode % 100 == 0) and (episode > 0):

            if episode % decay_period == 0:
                epsilon *= decay_rate


        np.savez('./result/qlearning.npz', Q=self.q, Q_TARGET=self.q_target, MEMORY=self.memory, OPTIMIZER=self.optimizer)
        return self.q, self.q_target, self.memory, self.optimizer


    def get_action(self, state, epsilon):  # epsilon-greedy
        if torch.cuda.is_available(): #gpu
            if np.random.rand() < epsilon:
                action = torch.tensor([np.random.choice(8)], device = self.device)
            else:
                action = torch.argmax(self.q(state[:8])).item()
                action = torch.tensor([action], device = self.device)

        else: #cpu
            if np.random.rand() < epsilon:
                action = np.random.choice(8)
            else:
                state_tensor = (torch.tensor(state[:8])).to(self.device)
                action = torch.argmax(self.q(state_tensor)).item()
        return action