from environment import boss
from agent import AGENT
from replaybuffer import ReplayBuffer
from qnet import Qnet
from BENNETT import BENNETT
from XINGQIU import XINGQIU
from buff_table import buff_table
import torch.optim as optim
import torch

learning_rate = 0.0005


USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    print("Cuda is available")
device = torch.device('cuda:0' if USE_CUDA else 'cpu')
#device = torch.device('cpu' if USE_CUDA else 'cpu')
print('학습을 진행하는 기기:', device)

bennett = BENNETT(device, level=90, base_hp=12397, add_hp=19563, base_atk=865, add_atk=562, cr=0.178, cd=0.834, bonus=0)
xingqiu = XINGQIU(device, level=90, base_hp=10222, add_hp=8177, base_atk=656, add_atk=873, cr=0.645, cd=1.176, bonus=0.666)
boss = boss(device, level=90, hp=500000, resist=0.1) #character1, level, hp, resist

q = Qnet().to(device)
q_target = Qnet().to(device)
buff_table = buff_table(device)
q_target.load_state_dict(q.state_dict())
memory = ReplayBuffer()
optimizer = optim.Adam(q.parameters(), lr=learning_rate)

agent = AGENT(bennett, xingqiu, boss, buff_table, q, q_target, device, memory, optimizer, is_upload=False)
agent.Q_learning(decay_period=10, decay_rate=0.8)