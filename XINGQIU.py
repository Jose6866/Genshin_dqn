import torch

class XINGQIU:
    def __init__(self, device, level, base_hp, add_hp, base_atk, add_atk, cr, cd, bonus):
        self.device = device
        self.level = torch.tensor(level, device = self.device)
        self.base_hp = torch.tensor(base_hp, device = self.device)
        self.add_hp = torch.tensor(add_hp, device = self.device)
        self.base_atk = torch.tensor(base_atk, device = self.device)
        self.add_atk = torch.tensor(add_atk, device = self.device)
        self.cr = torch.tensor(cr, device = self.device)
        self.cd = torch.tensor(cd+1, device = self.device)
        self.bonus = torch.tensor(bonus+1, device = self.device)

        self.na_cd = 0
        self.na_delay = 0.5
        self.ca_cd = 0
        self.ca_delay = 1
        self.e_cd = 21
        self.e_delay = 1
        self.q_cd = 20
        self.q_delay = 2

    def na(self, buff_table):
        buff_tmp = buff_table.buff_return()
        buff_atk = 0
        if not len(buff_tmp) == 0:
            for i in range(len(buff_tmp)):
                buff_atk = buff_atk + buff_tmp[i][3]
        demage = (self.base_atk + self.add_atk + buff_atk) * 0.504 * (1 + self.cr * self.cd)

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
        demage = (self.base_atk + self.add_atk + buff_atk) * 1.119 * (1 + self.cr * self.cd)
        return demage

    def e(self, buff_table):
        buff_tmp = buff_table.buff_return()
        buff_atk = 0
        if not len(buff_tmp) == 0:
            for i in range(len(buff_tmp)):
                buff_atk = buff_atk + buff_tmp[i][3]
        demage = (self.base_atk + self.add_atk + buff_atk) * 7.190 * (1 + self.cr * self.cd) * self.bonus
        return demage


    def q(self, buff_table):
        buff_name = 1 # 행추궁
        buff_type = 1 # 일반공격에 반응
        duration = 15

        buff_tmp = buff_table.buff_return()
        buff_atk = 0
        if not len(buff_tmp) == 0:
            for i in range(len(buff_tmp)):
                buff_atk = buff_atk + buff_tmp[i][3]
        demage = (self.base_atk + self.add_atk + buff_atk) * 1.15 * 2 * (1 + self.cr * self.cd) * self.bonus
        total = torch.tensor([[buff_name, buff_type, duration, demage]], device = self.device)

        creature_tmp = buff_table.creature_return()
        check = False
        for i in range(len(creature_tmp)):
            if creature_tmp[i][0] == buff_name:
                check = True
        if not check:
            buff_table.creature_cat(total)

        q_reward = 0

        return q_reward





