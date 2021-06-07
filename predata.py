"""
import ptvsd
ptvsd.enable_attach(address=('172.18.30.128', 5678))
print('wait for attach')
ptvsd.wait_for_attach()
print('succeed')
"""
import sys
import os
import numpy as np
import time
import random
import config as config
import re
import soundfile as sf
import resampy
import shutil

"""
Generate a batch of data(audio, visual feature)
"""

def create_mix_list(train_or_test, mix_k, data_path, all_spk, Num_samples_per_batch):
    list_path = data_path + '/list_mix_spk/'
    file_name = open(list_path + 'mix_{}_spk_{}0.txt'.format(mix_k, train_or_test), 'w')
    for line in range(Num_samples_per_batch):
        aim_spk_k = random.sample(all_spk, mix_k)
        line = ''
        ratio = round(5*np.random.rand()-2.5, 3)
        #ratio = 0
        for spk in aim_spk_k:
            sample_name = random.sample(os.listdir('{}/{}/{}/{}/'.format(data_path, 'lip_fea', train_or_test, spk)), 1)[0]
            sample_name = sample_name.replace('npy', 'wav')
            if spk == aim_spk_k[0]:
                #line += '/data/resaudio/{}/{} {} '.format(spk, sample_name, ratio)
                line += '{}/{}/{}'.format(spk, sample_name, ratio)
            elif spk == aim_spk_k[-1]:
                line += '/{}/{}/{}'.format(spk, sample_name, -1*ratio)
        line += '\n'
        file_name.write(line)
def prepare_data(mode, train_or_test,drop,ratio):
    """
    params
    mode: type = str, 'global' or 'once'
    global: get params
    once: get data for train or valid or test
    train_or_test: type = str, 'train','valid' or 'test'
    return: a batch data or params
    """
    print(ratio)
    ratio_forvideo =ratio
    if train_or_test == 'train':
        BATCH_SIZE = config.BATCH_SIZE
    else:
        BATCH_SIZE = config.BATCH_SIZE_EVAL
    mix_speechs = []
    aim_spkid = [] 
    aim_spkname = [] 
    query = []
    multi_spk_wav_list=[]  # the waveform of multi-spk-speech
    multi_video_list=[]  # the video of multi-spk
    multi_video_fea_list=[]  # the video's feature of multi-spk
    aim_wav_len = []

    # the path of total data
    data_path = os.path.join(config.aim_path, config.DATASET)
    all_spk = os.listdir(os.path.join(data_path, 'lip_fea', train_or_test))
    list_path = data_path + '/list_mix_spk/'
    mix_k = config.mix_spk
    print('The number of {} spk is {}' .format(train_or_test, len(all_spk)))
    if train_or_test == 'train':
        if config.DATA_AUG == True:
            create_mix_list(train_or_test, mix_k, data_path, all_spk, config.Num_samples_per_epoch)
            aim_list_path = list_path + 'mix_{}_spk_train0.txt'.format(mix_k)
        else:
            aim_list_path = list_path + 'mix_{}_spk_train.txt'.format(mix_k)
            
    if train_or_test == 'valid':
        aim_list_path = list_path + 'mix_{}_spk_valid.txt'.format(mix_k)
    if train_or_test == 'test':
        aim_list_path = list_path + 'mix_{}_spk_test.txt'.format(mix_k)

    sample_idx = 0
    batch_idx = 0
    all_samples_list = open(aim_list_path).readlines()
    # all mixed samples' name list
    number_samples_all = len(all_samples_list)
    # the number of mixed samples
    batch_mix = (number_samples_all-1) / BATCH_SIZE

    if train_or_test == 'train' and config.SHUFFLE_BATCH:
        random.shuffle(all_samples_list)
        print('\nshuffle train list success!')

    print('batch_total_num:', batch_mix)

    for ___ in range(number_samples_all):
        aim_spk_k = []
        aim_spk_db_k = []
        aim_spk_samplename_k = []
        if ___ == number_samples_all-1:
            #print('ends here.')
            yield False
        print(mix_k, 'mixed sample_idx[mix_k]:',sample_idx, batch_idx)
        if sample_idx >= batch_mix * BATCH_SIZE:
            print(mix_k, 'mixed data is over, trun to the others number.')
            yield False
        if config.DATASET == 'lrs2':
            sample = all_samples_list[sample_idx].strip('\n').split('/')
            aim_spk_db_k.append(sample[2]), aim_spk_db_k.append(sample[5])
            #aim_spk_db_k
            aim_spk_db_k = list(map(float, aim_spk_db_k))
            aim_spk_k.append(sample[0]), aim_spk_k.append(sample[3])
            aim_spk_samplename_k.append(sample[1][:-4]), aim_spk_samplename_k.append(sample[4][:-4])
            assert len(aim_spk_k) == mix_k == len(aim_spk_db_k) == len(aim_spk_samplename_k)
        if config.DATASET == 'GRID':
            aim_spk_k = re.findall('/(.{2,4})/.{6}\.wav ', all_samples_list[sample_idx])
            aim_spk_db_k = list(map(float, re.findall(' (.*?) ', all_samples_list[sample_idx])))
            aim_spk_samplename_k = re.findall('/(.{6})\.wav ', all_samples_list[sample_idx])
            #print(aim_spk_k, aim_spk_db_k, aim_spk_samplename_k)
            assert len(aim_spk_k) == mix_k == len(aim_spk_db_k) == len(aim_spk_samplename_k)
           
        if config.DATASET == 'TCDTIMIT':
            aim_spk_k = re.findall('\d+[M,F]', all_samples_list[sample_idx])
            aim_spk_db_k = list(map(float, re.findall(' (.*?) ', all_samples_list[sample_idx])))
            aim_spk_samplename_k = re.findall('[a-z][a-z]\d{1,4}', all_samples_list[sample_idx])
            assert len(aim_spk_k) == mix_k == len(aim_spk_db_k) == len(aim_spk_samplename_k)

        multi_fea_dict_this_sample = {}
        multi_wav_dict_this_sample = {}
        multi_video_dict_this_sample = {}
        multi_video_fea_dict_this_sample = {}
        multi_db_dict_this_sample = {}

        for k, spk in enumerate(aim_spk_k):
            # aim_spk_k ['s1','s3']
            sample_name = aim_spk_samplename_k[k]
            spk_speech_path = data_path + '/' + 'resaudio/' + spk + '/' + sample_name + '.wav'
            signal, rate = sf.read(spk_speech_path)  
            # 如果语音为多通道，则取第一个通道
            if len(signal.shape) > 1:
                signal = signal[:, 0]
            # 语音去均值和归一化
            signal -= np.mean(signal)  
            signal /= np.max(np.abs(signal))
            aim_wav_len.append(signal.shape[0])
            # 为了使一个batch中的语音长度相等
            if signal.shape[0] < config.MAX_LEN_SPEECH: 
            # zero padding if don't satisfy requirement
                signal = np.append(signal, np.zeros(config.MAX_LEN_SPEECH - signal.shape[0]))
            else:
                signal = signal[:config.MAX_LEN_SPEECH]
            # 目标说话人的语音单独记录
            if k == 0:
                # if the signal is first spk's speech
                ratio = 10 ** (aim_spk_db_k[k] / 20.0)
                signal = ratio * signal
                aim_spkname.append(aim_spk_k[0])
                aim_spk_speech = signal
                aim_spkid.append(aim_spkname)
                wav_mix = signal
                multi_wav_dict_this_sample[spk] = signal
                aim_spk_fea_video_path = data_path + '/lip_fea/' + train_or_test + '/' + spk + '/' +sample_name + '.npy'
                video_fea = np.load(aim_spk_fea_video_path)#93,256
                if video_fea.shape[0] < config.MAX_LEN_VIDEO:
                    shape = video_fea.shape
                    video_fea = np.vstack((video_fea, np.zeros((config.MAX_LEN_VIDEO-shape[0], shape[1]))))
                else:
                    video_fea = video_fea[:config.MAX_LEN_VIDEO, :]
                if drop=="random":
                    num=int(75*ratio_forvideo)
                    print(ratio)
                    print(num)
                    rs = random.sample(range(25,75),num)
                    for time in rs:
                        res = time-1
                        while res in rs:
                            res = res-1
                        video_fea[time,:]=video_fea[res,:]
                if drop=="continous":
                    num=int(75*ratio_forvideo)
                    re = 75-num
                    for time in range(re,75):
                        video_fea[time,:]=video_fea[re-1,:]
                multi_video_fea_dict_this_sample[spk] = video_fea
            # 第2、3...说话人
            else:
                ratio = 10 ** (aim_spk_db_k[k] / 20.0)
                signal = ratio*signal
                wav_mix = wav_mix + signal
                multi_wav_dict_this_sample[spk] = signal
                aim_spk_fea_video_path = data_path + '/lip_fea/' + train_or_test + '/' + spk + '/' + sample_name + '.npy'
                video_fea = np.load(aim_spk_fea_video_path)
                if video_fea.shape[0] < config.MAX_LEN_VIDEO:
                    shape = video_fea.shape
                    video_fea = np.vstack((video_fea, np.zeros((config.MAX_LEN_VIDEO-shape[0], shape[1]))))
                else:
                    video_fea = video_fea[:config.MAX_LEN_VIDEO, :]
                if drop=="random":
                    num=int(75*ratio_forvideo)
                    rs = random.sample(range(25,75),num)
                    for time in rs:
                        #res = time-1
                        #while res in rs:
                        #    res = res-1
                        #video_fea[time,:]=video_fea[res,:]
                        video_fea[time,:]=0
                if drop=="continous":
                    num=int(75*ratio_forvideo)
                    re = 75-num
                    for time in range(re,75):
                        video_fea[time,:]=video_fea[re-1,:]
                multi_video_fea_dict_this_sample[spk] = video_fea

        multi_spk_wav_list.append(multi_wav_dict_this_sample) 
        multi_video_list.append(multi_video_dict_this_sample) 
        multi_video_fea_list.append(multi_video_fea_dict_this_sample) 

        mix_speechs.append(wav_mix)
                                                                    
        batch_idx += 1
        if batch_idx == BATCH_SIZE: 
                    
            aim_wav_len = np.array(aim_wav_len)
            query = np.array(query)
            mix_speechs = np.array(mix_speechs)
            print('aimspk list from this gen:{}'.format(aim_spkname))
            print('\nspk list from this gen:', [one.keys() for one in multi_spk_wav_list])
            if mode == 'global':
                all_spk=sorted(all_spk)
                dict_spk_to_idx = {spk:idx for idx,spk in enumerate(all_spk)}
                dict_idx_to_spk = {idx:spk for idx,spk in enumerate(all_spk)}
                yield all_spk, dict_spk_to_idx, dict_idx_to_spk,\
                      aim_fea.shape[1], aim_fea.shape[2], config.MAX_LEN_VIDEO, len(all_spk), batch_total
            elif mode == 'once':
                yield {'mix_wav':mix_speechs,
                       'aim_spkname':aim_spkname,
                       'query':query,
                       'num_all_spk':len(all_spk),
                       'multi_spk_wav_list':multi_spk_wav_list,
                       'multi_video_list':multi_video_list,
                       'multi_video_fea_list':multi_video_fea_list,
                       'batch_total':batch_mix,
                       'top_k':mix_k,
                       'aim_wav_len':aim_wav_len
                       }
            # next batch initia setting
            mix_k = config.mix_spk
            batch_idx = 0
            mix_speechs = []
            aim_spkid = [] 
            aim_spkname = []
            query = []
            multi_spk_fea_list = []
            multi_spk_wav_list = []
            multi_video_list = []
            multi_video_fea_list = []
            aim_wav_len = []
        sample_idx += 1


if __name__ == "__main__":
    generator = prepare_data('once', 'train')
    while True:
        data = generator.__next__()
        if data == False:
            break
        print(data['aim_wav_len'])
        print(data['mix_wav'].shape) 
