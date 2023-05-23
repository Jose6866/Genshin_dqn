import torch

class BENNETT:
    def __init__(self, device, level, base_hp, add_hp, base_atk, add_atk, cr, cd, bonus):
        self.device = device
        self.level = torch.tensor(level, device = self.device)
        self.base_hp = torch.tensor(base_hp, device=self.device)
        self.add_hp = torch.tensor(add_hp, device=self.device)
        self.base_atk = torch.tensor(base_atk, device = self.device)
        self.add_atk = torch.tensor(add_atk, device = self.device)
        self.cr = torch.tensor(cr, device = self.device)
        self.cd = torch.tensor(cd+1, device = self.device)
        self.bonus = torch.tensor(bonus+1, device = self.device)

        self.na_cd = 0
        self.na_delay = 0.5
        self.ca_cd = 0
        self.ca_delay = 1
        self.e_cd = 5
        self.e_delay = 1
        self.q_cd = 15
        self.q_delay = 2

    def na(self, buff_table):
        buff_tmp = buff_table.buff_return()
        buff_atk = 0
        if not len(buff_tmp) == 0:
            for i in range(len(buff_tmp)):
                buff_atk = buff_atk + buff_tmp[i][3]
        demage = (self.base_atk + self.add_atk + buff_atk) * 0.57 * (1 + self.cr * self.cd)

        creature_tmp = buff_table.creature_return()
        if not len(creature_tmp) == 0:
            for i in range(len(creature_tmp)):
                demage = demage + creature_tmp[i][3]

        return demage

    def ca(self, buff_table):
        buff_tmp = buff_table.buff_return()
        buff_atk = 0
        if not len(buff_tmp) == 0:
            for i in range(len(buff_tmp)):
                buff_atk = buff_atk + buff_tmp[i][3]
        demage = (self.base_atk + self.add_atk + buff_atk) * 1.492 * (1 + self.cr * self.cd)
        return demage

    def e(self, buff_table):
        buff_tmp = buff_table.buff_return()
        buff_atk = 0
        if not len(buff_tmp) == 0:
            for i in range(len(buff_tmp)):
                buff_atk = buff_atk + buff_tmp[i][3]
        demage = (self.base_atk + self.add_atk + buff_atk) * 2.34 * (1 + self.cr * self.cd) * self.bonus
        return demage


    def q(self, buff_table):
        buff_name = 0 # 베넷궁인걸 알기위해
        buff_type = 0 # atk_bonus
        duration = 12
        ratio = self.base_atk * 0.952
        total = torch.tensor([[buff_name, buff_type, duration, ratio]], device = self.device)
        check = False
        buff_tmp = buff_table.buff_return()
        for i in range(len(buff_tmp)):
            if buff_tmp[i][0] == buff_name:
                check = True
        if not check:
            buff_table.buff_cat(total)

        buff_tmp = buff_table.buff_return()
        buff_atk = 0
        if not len(buff_tmp) == 0:
            for i in range(len(buff_tmp)):
                buff_atk = buff_atk + buff_tmp[i][3]

        q_reward = (self.base_atk + self.add_atk + buff_atk) * 4.95 * (1 + self.cr * self.cd) * self.bonus

        return q_reward





