import torch

class buff_table:
    def __init__(self, device):
        self.buff = torch.tensor((), device=device) # buff_name, buff_type, duration, ratio
        self.creature = torch.tensor((), device=device)  # buff_name, buff_type, duration, ratio,

    def size(self):
        return len(self.buff)

    def buff_cat(self, tensor):
        self.buff = torch.cat([self.buff, tensor])

    def creature_cat(self, tensor):
        self.creature = torch.cat([self.creature, tensor])

    def buff_return(self):
        return self.buff

    def creature_return(self):
        return self.creature

    def duration_decrease(self, delay):
        for i in range(len(self.buff)):
            self.buff[i][2] = self.buff[i][2] - delay
            if self.buff[i][2] <= 0:
                self.buff = torch.cat([self.buff[:i-1], self.buff[i+1:]])

        for i in range(len(self.creature)):
            self.creature[i][2] = self.creature[i][2] - delay
            if self.creature[i][2] <= 0:
                self.creature = torch.cat([self.creature[:i-1], self.creature[i+1:]])
