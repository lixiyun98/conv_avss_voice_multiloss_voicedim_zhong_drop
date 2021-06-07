# coding:utf-8

"""
import ptvsd
ptvsd.enable_attach(address=('172.18.30.128', 5678))
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
from model import TasNetVisual
import argparse
from loss import cal_sisnr_order_loss
from utils import Adjust_lr, Early_stop
from separation import bss_eval_sources
from torchsummary import summary
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
#os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
def compute_sdr(source, predict):
    source = source.squeeze(1).data.cpu().numpy()
    predict = predict.squeeze(1).data.cpu().numpy()
    B = source.shape[0]
    SDR = []
    for i in range(int(B/2)):
        source_idx = source[2*i:2*(i+1),:]
        predict_idx = predict[2*i:2*(i+1),:]
        speech_metric = bss_eval_sources(source_idx, predict_idx)
        print(speech_metric)
        sdr = speech_metric[0].mean()
        SDR.append(sdr)
    SDR = np.array(SDR)
    SDR = SDR.mean()
    print('SDR this batch:{}'.format(SDR))
    return SDR

def convert2numpy(data_list, top_k, BATCH_SIZE):
    key = list(data_list[0].keys())
    output_size = (BATCH_SIZE, top_k) + np.array(data_list[0][key[0]]).shape
    output_array = np.zeros(output_size, dtype=np.float32)
    for idx, dict_sample in enumerate(data_list):
        spk_all = sorted(dict_sample.keys())
        for jdx, spk in enumerate(spk_all):
            output_array[idx, jdx] = np.array(data_list[idx][spk])
    return output_array
    

def test(model, logger, step):
    model.eval()
    print("*" * 40 + 'test stage' + "*" * 40)
    test_data_gen = prepare_data('once', 'test')
    SDR_SUM = np.array([])
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
        True_wav = Variable(torch.from_numpy(True_wav)).cuda()
        shape = True_wav.shape
        True_wav = True_wav.contiguous().view(-1,shape[2])
        True_wav_len = Variable(torch.from_numpy(np.zeros((shape[0]*shape[1], 1), 'int32')+shape[2])).cuda()
        True_wav = True_wav.unsqueeze(1)
        predict_wav,pre2,ou1,ou2,ou4,ou5 = model(mix_speech, images_query)
        predict_wav = predict_wav.unsqueeze(1)
        predict_wav2 = pre2.unsqueeze(1)
        loss = cal_sisnr_order_loss(True_wav, predict_wav, True_wav_len)
        loss2 = cal_sisnr_order_loss(True_wav, predict_wav2, True_wav_len)
        print('loss:{}'.format(loss.item()))
        loss_total += [loss.item()]
    loss_aver = np.array(loss_total).mean()
    logger.log_test(loss_aver, 0, 0, 0, 0, step)
    print('test loss:{}' .format(loss_aver))
    print("*" * 40 + 'eval end' + "*" * 40)
    return loss_aver


def eval(model, logger, step):
    model.eval()
    print("*" * 40 + 'eval stage' + "*" * 40)
    eval_data_gen = prepare_data('once', 'valid')
    SDR_SUM = np.array([])
    loss_total = []
    while True:
        print('\n')
        eval_data = eval_data_gen.__next__()
        if eval_data == False:
            break
        top_k_num = eval_data['top_k']
        print('top-k this batch:', top_k_num)
        mix_wav = eval_data['mix_wav'].astype('float32')
        mix_speech_orignial = Variable(torch.from_numpy(mix_wav)).cuda()
        mix_speech = mix_speech_orignial
        multi_video_fea = convert2numpy(eval_data['multi_video_fea_list'], top_k_num, config.BATCH_SIZE).astype('float32')
        images_query = Variable(torch.from_numpy(multi_video_fea)).cuda()
        True_wav = convert2numpy(eval_data['multi_spk_wav_list'], top_k_num, config.BATCH_SIZE).astype('float32')
        True_wav = Variable(torch.from_numpy(True_wav)).cuda()
        shape = True_wav.shape
        True_wav = True_wav.contiguous().view(-1,shape[2])
        True_wav_len = Variable(torch.from_numpy(np.zeros((shape[0]*shape[1], 1), 'int32')+shape[2])).cuda()
        True_wav = True_wav.unsqueeze(1)
        predict_wav,pre2,ou1,ou2,ou4,ou5 = model(mix_speech, images_query)
        predict_wav = predict_wav.unsqueeze(1)
        predict_wav2 = pre2.unsqueeze(1)
        loss = cal_sisnr_order_loss(True_wav, predict_wav, True_wav_len)
        loss2 = cal_sisnr_order_loss(True_wav, predict_wav2, True_wav_len)
        print('loss:{}'.format(loss.item()))
        loss_total += [loss.item()]
    loss_aver = np.array(loss_total).mean()
    logger.log_validation(loss_aver, 0, 0, 0, 0, step)
    print('eval loss:{}' .format(loss_aver))
    print("*" * 40 + 'eval end' + "*" * 40)
    return loss_aver


def train(epoch_idx, optimizer, init_lr, step, model,  opt, logger):
    model.train()
    global_id = config.DATASET
    batch_idx = 0
    train_data_gen = prepare_data('once', 'train')
    while True:
        print("*" * 40, epoch_idx, batch_idx, "*" * 40)
        train_data = train_data_gen.__next__()
        if train_data == False:
            break
        top_k_num = train_data['top_k']
        print('mixed speakers this batch : ', top_k_num)
        mix_wav = train_data['mix_wav'].astype('float32')
        mix_speech_orignial = Variable(torch.from_numpy(mix_wav)).cuda()
        mix_speech = mix_speech_orignial
        multi_video_fea = convert2numpy(train_data['multi_video_fea_list'], top_k_num, config.BATCH_SIZE).astype('float32')
        images_query = Variable(torch.from_numpy(multi_video_fea)).cuda()
        True_wav = convert2numpy(train_data['multi_spk_wav_list'], top_k_num, config.BATCH_SIZE).astype('float32')
        True_wav = Variable(torch.from_numpy(True_wav)).cuda()
        shape = True_wav.shape
        True_wav = True_wav.contiguous().view(-1,shape[2])
        True_wav_len = Variable(torch.from_numpy(np.zeros((shape[0]*shape[1], 1), 'int32')+shape[2])).cuda()
        True_wav = True_wav.unsqueeze(1)
        #predict_wav,pre2,ou1,ou2,ou3,ou4,ou5,ou6 = model(mix_speech, images_query)
        print("the shape of mix_speech",mix_speech.size())
        print("the shape of image_query",images_query.size())
        predict_wav,pre2,ou1,ou2,ou4,ou5 = model(mix_speech, images_query)
        #ou = [ou1,ou2,ou3,ou4,ou5,ou6]
        ou = [ou1,ou2,ou4,ou5]
        predict_wav = predict_wav.unsqueeze(1)
        predict_wav2 = pre2.unsqueeze(1)
        loss = cal_sisnr_order_loss(True_wav, predict_wav, True_wav_len)
        loss2 = cal_sisnr_order_loss(True_wav, predict_wav2,  True_wav_len)
        loss3 = 0
        print("zzz",len(ou))
        for i in (0,len(ou)-1):
            ou[i] = ou[i].unsqueeze(1)
            print("hhh",ou[i].shape)
            loss3 += cal_sisnr_order_loss(True_wav, ou[i], True_wav_len)
        loss =(loss+loss2+loss3)/6
        logger.log_training(loss.item(), step)
        optimizer.zero_grad()
        loss.backward()
        w_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        print('loss:{} grad_norm:{}'.format(loss.item(), w_norm))
        optimizer.step()
        batch_idx += 1
        step += 1
    return global_id, step


def main(opt):
    print("*" * 80)
    print('Build Model')
    print("*" * 80)
    if config.DATA_AUG == True:
        print('use data aug')
    model = TasNetVisual(causal=config.causal, num_spk=config.mix_spk).cuda()
    #print(summary(model,input_size=[(1,48000),(1,2,256,75)]))
    model = nn.DataParallel(model, device_ids=range(opt.gpus))
    #print(model.state_dict())
    # compute total parameters in model
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    print('Params size in this model is {}' .format(param_count))
    # tensorboad's setting
    #if os.path.exists(opt.TensorboardPath) and opt.ModelPath == None:
    #    shutil.rmtree(opt.TensorboardPath)
    #    os.mkdir(opt.TensorboardPath)
    logger = Logger(opt.TensorboardPath)
    init_lr = opt.lr
    optimizer = torch.optim.Adam([{'params':model.parameters()}], lr=init_lr)
    lr_decay = Adjust_lr(init_lr, opt.lr_patient, optimizer)
    early_stop = Early_stop(opt.es_patient)
    epoch_idx = 0
    step = 0
    # load params, load pretrained model
    if 1 and opt.ModelPath is not None:
        params_path = opt.ModelPath
        model.load_state_dict(torch.load(params_path)['state_dict'],strict=False)
        epoch_idx = torch.load(params_path)['epoch_idx']
        step = int(36000/config.BATCH_SIZE) * epoch_idx
        print('Model:', params_path, 'load successfully')
        print('\n')
    print('*'*40 + 'Begin to train' + '*'*40)
    lr = init_lr
    print('learning rate:{}'.format(lr))
    while True:
        global_id, step = train(epoch_idx, optimizer, init_lr, step, model, opt, logger)
        epoch_idx += 1
        # save model's params
        if epoch_idx >= 1 and epoch_idx % opt.save_epoch == 0:
            print("save model and optimizer state at iteration {} to {}/V1modelparams_{}_{}".format(
                epoch_idx, opt.ParamsPath, global_id, epoch_idx))
            torch.save(
                {'state_dict':model.state_dict(),
                 'epoch_idx':epoch_idx
                }, '{}/ModelParams_{}_{}'.format(opt.ParamsPath, global_id, epoch_idx))
        # eval the model
        if 1 and epoch_idx >= 1 and epoch_idx < 10 and epoch_idx % 4 == 0:
            with torch.no_grad():
                loss_eval = eval(model, logger, step)
                loss_test = test(model, logger, step)
            key_value = early_stop.step(loss_test)
            if key_value == 0:
                print("End the whole train process in {} epoch!" .format(epoch_idx))
                sys.exit(0)
        if 1 and epoch_idx >= 10 and epoch_idx % opt.eval_epoch == 0:
            with torch.no_grad():
                loss_eval = eval(model, logger, step)
                loss_test = test(model, logger, step)
            if lr >= opt.lr / 8:
                lr = lr_decay.step(loss_test)
            key_value = early_stop.step(loss_test)
            if key_value == 0:
                print("End the whole train process in {} epoch!" .format(epoch_idx))
                sys.exit(0)
        #if lr >= opt.lr / 8:
        #    lr = lr_decay.step(loss_test)
        #key_value = early_stop.step(loss_test)
        logger.log_lr(lr, epoch_idx)
        print('this epoch {} learning rate is {}' .format(epoch_idx, lr))
        #if key_value == 0:
        #    print("End the whole train process in {} epoch!" .format(epoch_idx))
        #    sys.exit(0)
        # when the epoch equal to setting max epoch, end!
        if epoch_idx == config.MAX_EPOCH:
            sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ############### training settings ###################
    parser.add_argument('--gpus', type=int, default=4, help='number of gpus we demand')
    parser.add_argument('--TensorboardPath', type=str, default='log', help='path of saving tensorboard data')
    parser.add_argument('--ParamsPath', type=str, default='1', help='path of saving model params')
    parser.add_argument('--ModelPath', type=str, default=None, help='path of pretrained model')
    parser.add_argument('--eval_epoch', type=int, default=1, help='num of epochs to eval')
    parser.add_argument('--save_epoch', type=int, default=1, help='num of epochs to save model')
    parser.add_argument('--seed', type=int, default=1, help='the seed')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    #parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--lr_patient', type=int, default=3, help='patient for lr decay')
    parser.add_argument('--es_patient', type=int, default=10, help='patient for early stop')
    opt = parser.parse_args()
    np.random.seed(1)
    torch.manual_seed(1)
    print('seed now {}' .format(opt.seed))
    random.seed(1)
    np.random.seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    print("curr_device",torch.cuda.current_device())
    #torch.cuda.set_device(0)
    print("curr_device",torch.cuda.current_device())
    main(opt)
    
