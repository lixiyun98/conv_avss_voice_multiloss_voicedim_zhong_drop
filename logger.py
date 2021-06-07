import random
import torch
from tensorboardX import SummaryWriter
import torchvision
import numpy as np

class Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Logger, self).__init__(logdir)

    def log_training(self, loss_all, iteration):
        self.add_scalar("training.loss", loss_all, iteration)

    def log_training_phaseloss(self, loss_mag, loss_phase, loss_all, iteration):
        self.add_scalar("training.loss_mag", loss_mag, iteration)
        self.add_scalar("training.loss_phase", loss_phase, iteration)
        self.add_scalar("training.loss", loss_all, iteration)

    def log_training_wavloss(self, loss_real, loss_img, wavloss, loss_all, iteration):
        self.add_scalar("training.loss_real", loss_real, iteration)
        self.add_scalar("training.loss_imag", loss_img, iteration)
        self.add_scalar("training.wavloss", wavloss, iteration)
        self.add_scalar("training.loss", loss_all, iteration)
    
    def log_training_sisnr(self, loss_real, loss_imag, loss_sisnr, loss_all, iteration):
        self.add_scalar("training.loss_real", loss_real, iteration)
        self.add_scalar("training.loss_imag", loss_imag, iteration)
        self.add_scalar("training.loss_sisnr", loss_sisnr, iteration)
        self.add_scalar("training.loss", loss_all, iteration)

    def log_test(self, loss_all, sdr_overall, sdr_ff, sdr_mm, sdr_fm, iteration):
        self.add_scalar("test.loss", loss_all, iteration)

    def log_validation(self, loss_all, sdr_overall, sdr_ff, sdr_mm, sdr_fm, iteration):
        self.add_scalar("validation.loss", loss_all, iteration)
    
    def log_lr(self, lr, iteration):
        self.add_scalar("model.lr", lr, iteration)
