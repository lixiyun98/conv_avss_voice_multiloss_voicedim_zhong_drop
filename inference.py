# coding:utf-8
"""
inference stage, run the trained model on test sets and compute SDR
@zhangpeng
"""
"""
import ptvsd
ptvsd.enable_attach(address=('10.1.18.26', 5678))
print('wait for attach')
ptvsd.wait_for_attach()
print('succeed')
"""
import sys
import torch
from torch import nn 
from torch.autograd import Variable
import torch.nn.functional as F 
import numpy as np 
import random 
import time 
import config as config 
from predata import prepare_data
from logger import Logger 
import os 
import shutil 
from librosa.core import spectrum
import soundfile as sf 
#import bss_test
from model import TasNetVisual
import argparse
from loss import cal_sisnr_pit_loss,cal_sisnr_order_loss
from utils import Adjust_lr, Early_stop
from separation import bss_eval_sources
#from pypesq import pesq
from pystoi import stoi


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

#global count

def mask_audio(audio, length):
    return audio[:length]


def compute_metric(source, predict, wav_len):
    source = source.squeeze(1).data.cpu().numpy()
    predict = predict.squeeze(1).data.cpu().numpy()
    B = source.shape[0]
    print('Batchsize:{}'.format(B))
    #SDR = []
    STOI = []
    PESQ = []
    for i in range(int(B)):
        source_idx = source[i,:]
        predict_idx = predict[i,:]        
        source_idx = mask_audio(source_idx, wav_len[i])
        predict_idx = mask_audio(predict_idx, wav_len[i])
        print(source_idx.shape, predict_idx.shape)
        #sys.exit()
        #speech_metric = bss_eval_sources(source_idx, predict_idx)
        #STOI_ = stoi(source_idx, predict_idx/np.max(np.abs(predict_idx)), 16000)
        STOI_ = stoi(source_idx, predict_idx, 16000)
        #PESQ_ = pesq(source_idx, predict_idx/np.max(np.abs(predict_idx)), 16000)
        #PESQ_ = pesq(source_idx, predict_idx, 16000)
        #print(speech_metric)
        #sdr = speech_metric[0].mean()
        #SDR.append(sdr)
        STOI.append(STOI_)
        #PESQ.append(PESQ_)
    #SDR = np.array(SDR)
    STOI = np.array(STOI)
    #PESQ = np.array(PESQ)
    #SDR = SDR.mean()
    STOI = STOI.mean()
    #PESQ = PESQ.mean()
    print('STOI PESQ this batch:{} {}'.format(STOI, PESQ))
    #return STOI, PESQ
    return STOI

def compute_sdr(source, predict, mix, wav_len):
    source = source.squeeze(1).data.cpu().numpy()
    predict = predict.squeeze(1).data.cpu().numpy()
    B = source.shape[0]
    print('Batchsize:{}'.format(B))
    SDR = []
    SDRn = []
    print(source.shape, mix.shape, wav_len)
    #STOI = []
    #PESQ = []
    for i in range(B):
        source_idx = source[i,:]
        predict_idx = predict[i,:]
        mix_idx = mix[int(i/(config.mix_spk)),:]
        source_idx = mask_audio(source_idx, wav_len[i])
        predict_idx = mask_audio(predict_idx, wav_len[i])
        mix_idx = mask_audio(mix_idx, wav_len[i])
        speech_metric = bss_eval_sources(source_idx, predict_idx)
        speech_metric_n = bss_eval_sources(source_idx, mix_idx)
        #STOI_ = stoi(source_idx, predict_idx, 16000)
        #PESQ_ = pesq(source_idx, predict_idx, 16000)
        print(speech_metric)
        sdr = speech_metric[0].mean()
        sdrn = speech_metric_n[0].mean()
        SDR.append(sdr)
        SDRn.append(sdrn)
        #STOI.append(STOI_)
        #PESQ.append(PESQ_)
    SDR = np.array(SDR)
    SDRn = np.array(SDRn)
    #STOI = np.array(STOI)
    #PESQ = np.array(PESQ)
    SDR = SDR.mean()
    SDRn = SDRn.mean()
    #STOI = STOI.mean()
    #PESQ = PESQ.mean()
    print('SDR and SDRn this batch:{} and {}'.format(SDR, SDRn))
    return SDR, SDRn


def convert2numpy(data_list, top_k, BATCH_SIZE):
    key = list(data_list[0].keys())
    output_size = (BATCH_SIZE, top_k) + np.array(data_list[0][key[0]]).shape
    output_array = np.zeros(output_size, dtype=np.float32)
    for idx, dict_sample in enumerate(data_list):
        spk_all = sorted(dict_sample.keys())
        for jdx, spk in enumerate(spk_all):
            output_array[idx, jdx] = np.array(data_list[idx][spk])
    return output_array

def savewav(path, mix_wav, true_wav, predict_wav):
    predict_wav = predict_wav.squeeze(1).data.cpu().numpy()
    true_wav = true_wav.squeeze(1).data.cpu().numpy()
    print(mix_wav.shape, true_wav.shape, predict_wav.shape)
    BS = mix_wav.shape[0]
    for i in range(BS):
       label = time.time()
       #print(label)
       #sys.exit()
       #sf.write(path+'{}_mix.wav'.format(label), mix_wav[i], 16000)
       sf.write(path+'{}_pre1.wav'.format(label), predict_wav[i*2]/np.max(predict_wav[i*2]), 16000)
       sf.write(path+'{}_pre2.wav'.format(label), predict_wav[i*2+1]/np.max(predict_wav[i*2+1]), 16000)
       #sf.write(path+'{}_true1.wav'.format(label), true_wav[i*2], 16000)
       #sf.write(path+'{}_true2.wav'.format(label), true_wav[i*2+1], 16000)
    print('save wav completed!')
       

def inference(model):
    model.eval()
    print("*" * 40 + 'test stage' + "*" * 40)
    test_data_gen = prepare_data('once','test','random',0.1)
    SDR_SUM = np.array([])
    SDRn_SUM = np.array([])
    STOI_SUM = np.array([])
    PESQ_SUM = np.array([])
    loss_total = []
    while True:
        print('\n')
        test_data = test_data_gen.__next__()
        if test_data == False:
            break
        top_k_num = test_data['top_k']
        print('top-k this batch:', top_k_num)
        mix_wav = test_data['mix_wav'].astype('float32')
        mix_speech_orignial = Variable(torch.from_numpy(mix_wav)).cuda()
        mix_speech = mix_speech_orignial
        multi_video_fea = convert2numpy(test_data['multi_video_fea_list'], top_k_num, config.BATCH_SIZE).astype('float32')
        images_query = Variable(torch.from_numpy(multi_video_fea)).cuda()
        True_wav = convert2numpy(test_data['multi_spk_wav_list'], top_k_num, config.BATCH_SIZE).astype('float32')
        wav_len = test_data['aim_wav_len']
        True_wav = Variable(torch.from_numpy(True_wav)).cuda()
        shape = True_wav.shape
        True_wav = True_wav.contiguous().view(-1,shape[2])
        True_wav_len = Variable(torch.from_numpy(np.zeros((shape[0]*shape[1], 1), 'int32')+shape[2])).cuda()
        True_wav = True_wav.unsqueeze(1)
        predict_wav,pre2,ou1,ou2,ou4,ou5 = model(mix_speech, images_query)
        predict_wav = predict_wav.unsqueeze(1)
        mix_speech = mix_speech.unsqueeze(1) 
        print('true_wav:{} mix_wav:{}'.format(True_wav.shape, mix_speech.shape))
        loss = cal_sisnr_order_loss(True_wav, predict_wav, True_wav_len)
        print('loss:{}'.format(loss.item()))
        loss_total += [loss.item()]
        #savewav('./wav_output_LL5/', mix_wav, True_wav, predict_wav)
        #STOI, PESQ = compute_metric(True_wav, predict_wav, wav_len)
        STOI = compute_metric(True_wav, predict_wav, wav_len)
        sdr, sdrn = compute_sdr(True_wav, predict_wav, mix_wav, wav_len)
        SDR_SUM = np.append(SDR_SUM, sdr)
        SDRn_SUM = np.append(SDRn_SUM, sdrn)
        STOI_SUM = np.append(STOI_SUM, STOI)
        #PESQ_SUM = np.append(PESQ_SUM, PESQ)
        #print('SDR PESQ STOI', SDR_SUM.mean(), PESQ_SUM.mean(), STOI_SUM.mean())
        print('SDR PESQ STOI', SDR_SUM.mean(),  STOI_SUM.mean())
    SDR_aver = SDR_SUM.mean()
    SDRn_aver = SDRn_SUM.mean()
    STOI_aver = STOI_SUM.mean()
    #PESQ_aver = PESQ_SUM.mean() 
    SDR_aver_FF = SDR_SUM[:57].mean()
    SDR_aver_MM = SDR_SUM[57:110].mean()
    SDR_aver_FM = SDR_SUM[110:].mean()
    loss_aver = np.array(loss_total).mean()
    #logger.log_validation(loss_aver, SDR_aver, SDR_aver_FF, SDR_aver_MM, SDR_aver_FM, step)
    #print('SDR{}, SDRn{}, PESQ{}, STOI{}' .format(SDR_aver, SDRn_aver, PESQ_aver, STOI_aver))
    print('SDR{}, SDRn{}, STOI{}' .format(SDR_aver, SDRn_aver, STOI_aver))
    #print('SDR_overall{}, SDR_FF{}, SDR_MM{}, SDR_FM{}' .format(SDR_aver, SDR_aver_FF, SDR_aver_MM, SDR_aver_FM))
    print('test loss:{}' .format(loss_aver))
    print("*" * 40 + 'eval end' + "*" * 40)
    #sys.exit()
    return loss_aver

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ModelPath', type=str, default='1/best/ModelParams_lrs2_24', help='the model we loaded')
    parser.add_argument('--gpus', type=int, default=4, help='number of gpu we use')
    opt = parser.parse_args()
    model = TasNetVisual(causal=True, num_spk=config.mix_spk).cuda()
    model = nn.DataParallel(model, device_ids=range(opt.gpus))
    params_path = opt.ModelPath
    model.load_state_dict(torch.load(params_path)['state_dict'])
    print('Params:',params_path, 'loaded successfully!\n')
    # run inference
    np.random.seed(1234)  
    torch.manual_seed(1234)
    random.seed(1234)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    with torch.no_grad():
        inference(model)
