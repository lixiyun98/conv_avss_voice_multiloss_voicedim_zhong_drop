# coding:utf-8
#"""
import ptvsd
ptvsd.enable_attach(address=('172.18.30.128', 5678))
print('wait for attach')
ptvsd.wait_for_attach()
print('succeed')
#"""
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

#Dataset = 'GRID'
Dataset = 'TCDTIMIT'
if Dataset == 'GRID':
    spk_female_test = ['s16','s18','s23']
    spk_male_test = ['s12','s17','s6']
    spk_female_valid = ['s11','s25','s4']
    spk_male_valid = ['s1','s27','s30']
if Dataset == 'TCDTIMIT':
    spk_female_test = ['30F','31F','32F','36F','40F']
    spk_male_test = ['27M','28M','29M','41M','53M','57M']
    spk_female_valid = ['17F','37F','44F','46F','50F']
    spk_male_valid = ['22M','24M','25M','47M','48M','56M']

random.seed(1)
# train valid or test
def GenerateF3(all_spk, number, file_name, train_or_test):
    for line in range(number):
        aim_spk_k = random.sample(all_spk, 3)
        # get mix_k(number) speakers from all_spk randomly
        line = ''
        ratio = round(5*np.random.rand()-2.5, 3)
        #ratio = 0 
        # SNR 0-5 Db
        for i, spk in enumerate(aim_spk_k):
            sample_name = random.sample(os.listdir('{}/{}/{}/{}/'.format(data_path, 'lip_fea', train_or_test, spk)),1)[0]
            # 从一个说话人中随机采样一句语音
            sample_name = sample_name.replace('npy', 'wav')
            if i == 0:
                line += '/data/audio/{}/{} {} '.format(aim_spk_k[i], sample_name, ratio)
            if i == 1:
                line += '/data/audio/{}/{} {} '.format(aim_spk_k[i], sample_name, -1*ratio)
            if i == 2:
                line += '/data/audio/{}/{} {} '.format(aim_spk_k[i], sample_name, 1)
        line += '\n'
        file_name.write(line)


def GenerateF2M1(spk_male, spk_female, number, file_name, train_or_test):
    for line in range(number):
        aim_spk_k = []
        aim_spk_k.append(random.sample(spk_male, 1)[0])
        for speaker in random.sample(spk_female, 2):
            aim_spk_k.append(speaker)
        #aim_spk_k.append(random.sample(spk_female, 2))
        line = ''
        ratio = round(5*np.random.rand()-2.5, 3)
        for i, spk in enumerate(aim_spk_k):
            sample_name = random.sample(os.listdir('{}/{}/{}/{}/'.format(data_path, 'lip_fea', train_or_test, spk)),1)[0]
            sample_name = sample_name.replace('npy', 'wav')
            if i == 0:
                line += '/data/audio/{}/{} {} '.format(spk, sample_name, ratio)
            if i == 1:
                line += '/data/audio/{}/{} {} '.format(spk, sample_name, -1*ratio)
            if i == 2:
                line += '/data/audio/{}/{} {} '.format(spk, sample_name, 1)
        line += '\n'
        file_name.write(line)


def GenerateF1M2(spk_male, spk_female, number, file_name, train_or_test):
    for line in range(number):
        aim_spk_k = []
        aim_spk_k.append(random.sample(spk_female, 1)[0])
        for speaker in random.sample(spk_male, 2):
            aim_spk_k.append(speaker)
        line = ''
        ratio = round(5*np.random.rand()-2.5, 3)
        for i, spk in enumerate(aim_spk_k):
            sample_name = random.sample(os.listdir('{}/{}/{}/{}/'.format(data_path, 'lip_fea', train_or_test, spk)),1)[0]
            sample_name = sample_name.replace('npy', 'wav')
            if i == 0:
                line += '/data/audio/{}/{} {} '.format(spk, sample_name, ratio)
            if i == 1:
                line += '/data/audio/{}/{} {} '.format(spk, sample_name, -1*ratio)
            if i == 2:
                line += '/data/audio/{}/{} {} '.format(spk, sample_name, 1)
        line += '\n'
        file_name.write(line)


def GenerateM3(all_spk, number, file_name, train_or_test):
    for line in range(number):
        aim_spk_k = random.sample(all_spk, 3)
        # get mix_k(number) speakers from all_spk randomly
        line = ''
        ratio = round(5*np.random.rand()-2.5, 3)
        #ratio = 0 
        # SNR 0-5 Db
        for i, spk in enumerate(aim_spk_k):
            sample_name = random.sample(os.listdir('{}/{}/{}/{}/'.format(data_path, 'lip_fea', train_or_test, spk)),1)[0]
            # 从一个说话人中随机采样一句语音
            sample_name = sample_name.replace('npy', 'wav')
            if i == 0:
                line += '/data/audio/{}/{} {} '.format(aim_spk_k[i], sample_name, ratio)
            if i == 1:
                line += '/data/audio/{}/{} {} '.format(aim_spk_k[i], sample_name, -1*ratio)
            if i == 2:
                line += '/data/audio/{}/{} {} '.format(aim_spk_k[i], sample_name, 1)
        line += '\n'
        file_name.write(line)


def create_mix_list_bn(train_or_test, mix_k, data_path, spk_male, spk_female, Num_samples_per_batch):
    list_path = data_path + '/list_mix_spk/'
    if os.path.exists(list_path + 'mix_{}_spk_{}.txt'.format(mix_k, train_or_test)):
        os.remove(list_path + 'mix_{}_spk_{}.txt'.format(mix_k, train_or_test))
    file_name = open(list_path + 'mix_{}_spk_{}.txt'.format(mix_k, train_or_test), 'a')
    GenerateF3(spk_female, int(Num_samples_per_batch*0.26), file_name, train_or_test)
    GenerateF2M1(spk_male, spk_female, int(Num_samples_per_batch*0.23), file_name, train_or_test)
    GenerateF1M2(spk_male, spk_female, int(Num_samples_per_batch*0.25), file_name, train_or_test)
    GenerateM3(spk_male, int(Num_samples_per_batch*0.26), file_name, train_or_test)


def create_mix_list(train_or_test, mix_k, data_path, all_spk, Num_samples_per_batch):
    list_path = data_path + '/list_mix_spk/'
    file_name = open(list_path + 'mix_{}_spk_{}.txt'.format(mix_k, train_or_test), 'w')
    for line in range(Num_samples_per_batch):
        aim_spk_k = random.sample(all_spk, mix_k)
        line = ''
        ratio = round(5*np.random.rand()-2.5, 3)
        #ratio = 0
        for i, spk in enumerate(aim_spk_k):
            sample_name = random.sample(os.listdir('{}/{}/{}/{}/'.format(data_path, 'lip_fea', train_or_test, spk)), 1)[0]
            sample_name = sample_name.replace('npy', 'wav')
            if i == 0:
                line += '/data/audio/{}/{} {} '.format(aim_spk_k[i], sample_name, ratio)
            elif i == 1:
                line += '/data/audio/{}/{} {} '.format(aim_spk_k[i], sample_name, -1*ratio)
            else:
                line += '/data/audio/{}/{} {} '.format(aim_spk_k[i], sample_name, 1)
        line += '\n'
        file_name.write(line)

#data_path = './Datasets/' + 'data/'
data_path = os.path.join('../Visual/Datasets/data', Dataset)
aim_list_path = data_path + '/lip_fea/' + '/train/'
all_spk = os.listdir(aim_list_path)
#print(len(all_spk))
create_mix_list('train', 3, data_path, all_spk, 21601)
create_mix_list_bn('valid', 3, data_path, spk_male_valid, spk_female_valid, 1800)
create_mix_list_bn('test', 3, data_path, spk_male_test, spk_female_test, 1800)
#print('num of spk {}'.format(len(all_spk))) 
