# coding:utf-8

"""
    Configuration Profile
"""

import time
import configparser
import soundfile as sf
import resampy
import numpy as np
from scipy import signal
DATASET = 'lrs2' # 包含GRID、TCDTIMIT和LRS2
#aim_path='../../zhangpeng/CONV_AVSS/Datasets/data'
aim_path='../data'
Num_samples_per_epoch = 36001 # 固定了训练list,GRID LRS2:36000, TCDTIMIT:21600
# 最大epoch
MAX_EPOCH = 200
# 训练时的batch_size
#BATCH_SIZE = 4
BATCH_SIZE = 4
# 评估的batch size
#BATCH_SIZE_EVAL = 4
BATCH_SIZE_EVAL = 4
# 语音的采样率
FRAME_RATE = 2*8000
# 是否shuffle训练list
SHUFFLE_BATCH = True
# 设定最小混叠说话人数，Minimum number of mixed speakers for training
MIN_MIX = 2
# 设定最大混叠说话人数，Maximum number of mixed speakers for training
MAX_MIX = 2
# 语音混合的信噪比
dB = 5
# 设置训练/开发/验证模型的最大语音长度(秒)
MAX_LEN = 3 # GRID:3s TCD-TIMIT,LRS2:5s
MAX_LEN_SPEECH = FRAME_RATE*MAX_LEN
mix_spk = 2 # 混合语音中说话人的个数
VIDEO_RATE = 25
MAX_LEN_VIDEO = MAX_LEN*VIDEO_RATE
#MODAL_FUSION = 'CAT' # CAT或DF
MODAL_FUSION = 'DF' # CAT或DF
FUSION_POSITION = 'ALL' # '0','8','16',ALL
VISUAL_DIM = 64
SKIP = True
causal = True
mode_LN  = 'cLN' # or cLN 只在causal时起作用
DATA_AUG = False # 是否采用数据扩充
