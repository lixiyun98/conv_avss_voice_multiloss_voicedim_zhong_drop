# coding:utf-8

class Adjust_lr(object):
    def __init__(self, init_lr, lr_patient, optimizer):
        self.last_loss = 1e5
        self.now_lr = init_lr
        self.lr_patient = lr_patient
        self.count = 0
        self.optimizer = optimizer

    def step(self, loss):
        if loss <= self.last_loss:
            self.last_loss = loss
            self.count = 0
        else:
            self.count += 1
        if self.count == self.lr_patient:
            lr = self.now_lr / 2
            optim_state = self.optimizer.state_dict()
            optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr']/2
            self.now_lr = lr
            self.count = 0  
            self.last_loss = loss
        return self.now_lr


class Early_stop(object):
    def __init__ (self, es_patient):
        self.last_loss = 1e5
        self.es_patient = es_patient
        self.count = 0

    def step(self, loss):
        if loss <= self.last_loss:
            self.last_loss = loss
            self.count = 0 
        else:
            self.count += 1
        if self.count == self.es_patient:
            return 0 
        else:
            return 1
