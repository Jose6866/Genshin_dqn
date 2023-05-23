import torch


class boss:
    def __init__(self, device, level, hp, resist):
        self.device = device
        self.level = torch.tensor(level, device = self.device)
        self.hp = torch.tensor(hp, device = self.device)
        self.defence = self.level * 5 + 500
        self.resist = torch.tensor(resist, device = self.device)
        self.goal = 0

    def is_goal(self, state):
        x = state[8]
        return x <= self.goal

    def reward(self, character, demage, boss_hp):
        defense_demage = (self.defence / (self.defence + character.level * 5 + 500))
        total_demage = (demage*(1 * defense_demage))//1
        if boss_hp <= total_demage:
            reward = 999999
        else:
            reward = total_demage
        return torch.tensor([reward], device = self.device)

    def interaction(self, state, action, char_num, character, buff_table):
        if torch.cuda.is_available():
            next_state = state.clone().detach()
        else :
            next_state = state.copy()

        if self.is_goal(state):
            next_state = state
        else:
            if (action%4) == 0: #평타
                for i in range(8):
                    next_state[i] = state[i] - character.na_delay
                    if next_state[i] < 0:
                        next_state[i] = 0
                initial_demage = character.na(buff_table)
                next_state[char_num * 4 + 0] = character.na_cd
                r = self.reward(character, initial_demage, state[8])
                next_state[8] = state[8] - r
                delay = character.na_delay

            elif (action%4) == 1:  #강공
                for i in range(8):
                    next_state[i] = state[i] - character.ca_delay
                    if next_state[i] < 0:
                        next_state[i] = 0
                initial_demage = character.ca(buff_table)
                next_state[char_num * 4 + 1] = character.ca_cd
                r = self.reward(character, initial_demage, state[8])
                next_state[8] = state[8] - r
                delay = character.ca_delay

            elif (action%4) == 2:  #e
                for i in range(8):
                        next_state[i] = state[i] - character.e_delay
                        if next_state[i] < 0:
                            next_state[i] = 0
                initial_demage = character.e(buff_table)
                next_state[char_num * 4 + 2] = character.e_cd
                r = self.reward(character, initial_demage, state[8])
                next_state[8] = state[8] - r
                delay = character.e_delay

            else:  #q
                for i in range(8):
                        next_state[i] = state[i] - character.q_delay
                        if next_state[i] < 0:
                            next_state[i] = 0
                initial_demage = character.q(buff_table)
                next_state[char_num * 4 + 3] = character.q_cd
                r = self.reward(character, initial_demage, state[8])
                next_state[8] = state[8]- r
                delay = character.q_delay

        buff_table.duration_decrease(delay)

        return next_state, r, delay
