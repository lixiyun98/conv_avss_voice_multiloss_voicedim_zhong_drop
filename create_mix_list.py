# coding:utf-8

import sys
import os
import numpy as np
import time
import random
import config as config
import soundfile as sf
import resampy
import shutil

"""
consider gender balance generate valid and test set (2.5 hours)
"""

mode = 'train'
random.seed(1)
# train valid or test
def GenerateSameGenderData(all_spk, number, file_name, train_or_test):
    for line in range(number):
        aim_spk_k = random.sample(all_spk, 2)
        # get mix_k(number) speakers from all_spk randomly
        line = ''
        ratio = round(5*np.random.rand()-2.5, 3)
        #ratio = 0 
        # SNR 0-5 Db
        for spk in aim_spk_k:
            sample_name = random.sample(os.listdir('{}/{}/{}/{}/'.format(data_path, 'lip_fea1', train_or_test, spk)),1)[0]
            sample_name = sample_name.replace('npy', 'wav')
            if spk == aim_spk_k[0]:
                line += '/data/audio/{}/{} {} '.format(spk, sample_name, ratio)
            elif spk == aim_spk_k[-1]:
                line += '/data/audio/{}/{} {} '.format(spk, sample_name, -1*ratio)
        line += '\n'
        file_name.write(line)


def GenerateDiffGenderData(spk_male, spk_female, number, file_name, train_or_test):
    for line in range(number):
        aim_spk_k = []
        aim_spk_k.append(random.sample(spk_male, 1))
        aim_spk_k.append(random.sample(spk_female, 1))
        line = ''
        ratio = round(5*np.random.rand()-2.5, 3)
        #ratio = 0
        for i, spk in enumerate(aim_spk_k):
            spk = spk[0]
            sample_name = random.sample(os.listdir('{}/{}/{}/{}/'.format(data_path, 'lip_fea1', train_or_test, spk)),1)[0]
            sample_name = sample_name.replace('npy', 'wav')
            line += '{}/{} {} '.format(spk, sample_name, ratio)
            if spk == aimdd_spk_k[0][0]:
                line += '/data/audio/{}/{} {} '.format(spk, sample_name, -1*ratio)
        line += '\n'
        file_name.write(line)


def create_mix_list_bn(train_or_test, mix_k, data_path, spk_male, spk_female, Num_samples_per_batch):
    list_path = data_path + '/list_mix_spk/'
    if os.path.exists(list_path + 'mix_{}_spk_{}1.txt'.format(mix_k, train_or_test)):
        os.remove(list_path + 'mix_{}_spk_{}1.txt'.format(mix_k, train_or_test))
    file_name = open(list_path + 'mix_{}_spk_{}1.txt'.format(mix_k, train_or_test), 'a')
    # generate same gender data F-F
    GenerateSameGenderData(spk_female, int(Num_samples_per_batch*0.21), file_name, train_or_test)
    # generate same gender data M-M
    GenerateSameGenderData(spk_male, int(Num_samples_per_batch*0.23), file_name, train_or_test)
    # generate diff gender data F-M
    GenerateDiffGenderData(spk_male, spk_female, int(Num_samples_per_batch), file_name, train_or_test)

def create_mix_list(train_or_test, mix_k, data_path, all_spk, Num_samples_per_batch):
    list_path = data_path + '/list_mix_spk/'
    file_name = open(list_path + 'mix_{}_spk_{}.txt'.format(mix_k, train_or_test), 'w')
    for line in range(Num_samples_per_batch):
        aim_spk_k = random.sample(all_spk, mix_k)
        line = ''
        ratio = round(5*np.random.rand()-2.5, 3)
        #ratio = 0
        for spk in aim_spk_k:
            sample_name = random.sample(os.listdir('{}/{}/{}/{}/'.format(data_path, 'lip_fea', train_or_test, spk)), 1)[0]
            sample_name = sample_name.replace('npy', 'wav')
            if spk == aim_spk_k[0]:
                line += '/data/resaudio/{}/{}/ {} '.format(spk, sample_name, ratio)
            elif spk == aim_spk_k[-1]:
                line += '/data/resaudio/{}/{}/ {} '.format(spk, sample_name, -1*ratio)
        line += '\n'
        file_name.write(line)

#data_path = './Datasets/' + 'data/'
data_path = '../Visual/Datasets/data/GRID/'
aim_list_path = data_path + 'lip_fea/' + mode
#"""
all_spk = os.listdir(aim_list_path)
print(len(all_spk))
#sys.exit()
create_mix_list(mode, 2, data_path, all_spk, 36001)
#create_mix_list_bn(mode, 2, data_path, spk_male_valid, spk_female_valid, 30)
#"""
#spk_all = os.listdir(aim_list_path)
print('num of spk {}'.format(len(all_spk))) 
#create_mix_list('train', 2, data_path, spk_all, 30, i)       
