import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys

from utility import models, sdr
from utility.models import cLN
import config

# Conv-TasNet
class TasNetVisual(nn.Module):
    def __init__(self, enc_dim=512, feature_dim=128, visual_dim=config.VISUAL_DIM, sr=16000, win=2, layer=8, stack=3, 
                 kernel=3, causal=False, num_spk=2):
        super(TasNetVisual, self).__init__()
        """
        encoder output dim: enc_dim
        TCN hidden dim: feature_dim
        """
        
        # hyper parameters
        self.num_spk = num_spk
        self.enc_dim = enc_dim
        self.feature_dim = feature_dim
        if not causal:
            self.visual_dim = visual_dim
        else:
            self.visual_dim = visual_dim * 2
        self.win = int(sr*win/1000)
        self.stride = self.win // 2  
        self.layer = layer
        self.stack = stack
        self.kernel = kernel
        self.causal = causal
        
        # mix audio encoder
        self.encoder = nn.Conv1d(1, self.enc_dim, self.win, bias=False, stride=self.stride)
        self.encoder2 = nn.Conv1d(1, self.enc_dim, self.win, bias=False, stride=self.stride)
        # LayerNorm
        if not causal:
            self.LN = nn.GroupNorm(1, self.enc_dim, eps=1e-8)
        else:
            if config.mode_LN == 'cLN':
                self.LN = cLN(self.enc_dim, eps=1e-8)
                self.LN2 = cLN(self.enc_dim, eps=1e-8)
                print('use cLN')
            else:
                self.LN = nn.BatchNorm1d(self.enc_dim, eps=1e-08)
                print('use BN')
        # 瓶颈层
        self.BN = nn.Conv1d(self.enc_dim, self.feature_dim, 1)
        self.BN2 = nn.Conv1d(self.enc_dim, self.feature_dim, 1)
        self.BN3 = nn.Conv1d(self.enc_dim, self.feature_dim, 1)

        # visual feature encoder
        if not causal:
            self.visual_encoder = nn.LSTM(256, self.visual_dim, 3, batch_first=True, bidirectional=True)
        else:
            self.visual_encoder = nn.LSTM(256, self.visual_dim, 3, batch_first=True, bidirectional=False)
        self.voiceshape = 128
        if not causal:
            self.voice_encoder1 = nn.LSTM(128, self.voiceshape, 3, batch_first=True, bidirectional=True)
        else:
            self.voice_encoder1 = nn.LSTM(128, self.voiceshape, 3, batch_first=True, bidirectional=False)
        # separator
        self.separator = models.TCN(self.feature_dim, self.feature_dim*4, self.layer, self.stack, self.num_spk,
                                causal=self.causal, dilated=True)
        self.separator2 = models.TCN(self.feature_dim, self.feature_dim*4, self.layer, self.stack, self.num_spk,
                                causal=self.causal, dilated=True)
        
        self.output = nn.Sequential(nn.PReLU(),
                                    nn.Conv1d(self.feature_dim, self.enc_dim, 1))
        self.output2 = nn.Sequential(nn.PReLU(),
                                    nn.Conv1d(self.feature_dim, self.enc_dim, 1))
        # output decoder
        self.decoder = nn.ConvTranspose1d(self.enc_dim, 1, self.win, bias=False, stride=self.stride)
        self.decoder2 = nn.ConvTranspose1d(self.enc_dim, 1, self.win, bias=False, stride=self.stride)

    def pad_signal(self, input):
        # input is the waveforms: (B, T) or (B, 1, T)
        # reshape and padding
        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")
        
        if input.dim() == 2:
            input = input.unsqueeze(1)
        batch_size = input.size(0)
        nsample = input.size(2)
        rest = self.win - (self.stride + nsample % self.win) % self.win
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, 1, rest)).type(input.type())
            input = torch.cat([input, pad], 2)
        
        pad_aux = Variable(torch.zeros(batch_size, 1, self.stride)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest
        
    def forward(self, input, visual):
        
        # padding
        output, rest = self.pad_signal(input)
        batch_size = output.size(0)
        #mix = output 
        # mix audio encoder & layer norm & 瓶颈层
        enc_output = self.encoder(output)  # B, N, L
        shape = enc_output.shape
        print(shape)
        enc_output = enc_output.unsqueeze(1).expand(shape[0], self.num_spk, shape[1], shape[2]).contiguous().view(-1, shape[1], shape[2])
        mix = self.BN(self.LN(enc_output))

        # visual encoder
        visual_size = visual.shape
        visual = visual.contiguous().view(-1,visual_size[2],visual_size[3])
        self.visual_encoder.flatten_parameters()
        self.voice_encoder1.flatten_parameters()
        #print("visual1",visual.shape)
        query, _ = self.visual_encoder(visual)
        #print("query1",query.shape)
        query = F.upsample(query.transpose(1,2), mix.shape[2], mode='linear')
        #shape  = query.shape
        #query = query.view(batch_size, self.num_spk, shape[1], shape[2])
        # 分离网络
        output,ou1,ou2 = self.separator(mix,query,0,mix)
        #res = torch.tensor(res).cuda()
        #for i in range(0,len(res)):
        #    masks = torch.sigmoid(self.output(res[i]))
        #    masked_output = enc_output * masks  # B, C, N, L
        #    output = self.decoder(masked_output)  # B*C, 1, L
        #    output2 = output.squeeze(1)
        #    output2 = output2[:,self.stride:-(rest+self.stride)].contiguous()  # B*C, 1, L
        #    res[i] = output2   
        # 通过mask后获得预测语音
        masks = torch.sigmoid(self.output(output))
        masked_output = enc_output * masks  # B, C, N, L
        #print("maskoutput1",masked_output.shape)
     
        output = self.decoder(masked_output)  # B*C, 1, L
        output2 = output.squeeze(1)
        output2 = output2[:,self.stride:-(rest+self.stride)].contiguous()  # B*C, 1, L
        masks = torch.sigmoid(self.output(ou1))
        masked_output = enc_output * masks  # B, C, N, L
        ou1 = self.decoder(masked_output)  # B*C, 1, L
        ou1 = ou1.squeeze(1)
        ou1 = ou1[:,self.stride:-(rest+self.stride)].contiguous()  # B*C, 1, L
        masks = torch.sigmoid(self.output(ou2))
        masked_output = enc_output * masks  # B, C, N, L
        ou2 = self.decoder(masked_output)  # B*C, 1, L
        ou2 = ou2.squeeze(1)
        ou2 = ou2[:,self.stride:-(rest+self.stride)].contiguous()  # B*C, 1, L
        #masks = torch.sigmoid(self.output(ou3))
        #masked_output = enc_output * masks  # B, C, N, L
        #ou3 = self.decoder(masked_output)  # B*C, 1, L
        #ou3 = ou3.squeeze(1)
        #ou3 = ou3[:,self.stride:-(rest+self.stride)].contiguous()  # B*C, 1, L

        #print("output1",output.shape)
        #output = output[:,self.stride:-(rest+self.stride)].contiguous()  # B*C, 1, L
        #print(output.shape)
        enc_output2 = self.encoder(output)  # B, N, L
        #enc_output2 = self.encoder2(output)  # B, N, L
        #print("output2",output.shape)
        #print(self.BN(masked_output).shape)
        voice,_ = self.voice_encoder1(self.BN2(enc_output2).transpose(1,2))
        voice = torch.relu(voice)
        print("voice2",voice.shape)#2,3002,128
        voice = torch.cumsum(voice, dim=1)
        shape = voice.shape
        a = torch.ones((shape[1]))
        a = torch.cumsum(a, dim=0)
        a = a.view(1,-1,1).expand_as(voice)
        a = a.cuda()
        voice = voice/a
        print("voice3",voice.shape)#2,3002,128
        # voice = torch.mean(voice,dim=1,keepdim=True) # B,L,N
        shape = voice.shape
        voice = voice.expand(shape[0],3002,shape[2])
        print("voice2",voice.shape)#2,3002,128
        #print(mix.shape)
        #print("masked_output",masked_output.shape)
        voice = voice.transpose(1,2)
        output_new, rest_new = self.pad_signal(input)
        batch_size = output_new.size(0)
        #mix = output 
        # mix audio encoder & layer norm & 瓶颈层
        enc_output_new = self.encoder2(output_new)  # B, N, L
        shape = enc_output_new.shape
        print(shape)
        enc_output = enc_output_new.unsqueeze(1).expand(shape[0], self.num_spk, shape[1], shape[2]).contiguous().view(-1, shape[1], shape[2])
        mix = self.BN3(self.LN2(enc_output))
        output,ou4,ou5 = self.separator2(mix, query, 1,voice)
        #output = self.separator2(mix, query, voice)
        #print("output",output.shape)
         
        #res1 = torch.tensor(res1).cuda()
        #for i in range(0,len(res1)):
        #    masks = torch.sigmoid(self.output(res1[i]))
        #    masked_output = enc_output * masks  # B, C, N, L
        #    output = self.decoder(masked_output)  # B*C, 1, L
        #    output2 = output.squeeze(1)
        #    output2 = output2[:,self.stride:-(rest+self.stride)].contiguous()  # B*C, 1, L
        #    res1[i] = output2   
        masks = torch.sigmoid(self.output2(output))
        masked_output = enc_output * masks  # B, C, N, L
        #print("mask2",masked_output.shape) 
        # audio decoder
        output = self.decoder2(masked_output).squeeze(1)  # B*C, 1, L
        output = output[:,self.stride:-(rest+self.stride)].contiguous()  # B*C, 1, L
        print(output.shape)
        output1 = output
        masks = torch.sigmoid(self.output2(ou4))
        masked_output = enc_output * masks  # B, C, N, L
        ou4 = self.decoder2(masked_output)  # B*C, 1, L
        ou4 = ou4.squeeze(1)
        ou4 = ou4[:,self.stride:-(rest+self.stride)].contiguous()  # B*C, 1, L
        masks = torch.sigmoid(self.output2(ou5))
        masked_output = enc_output * masks  # B, C, N, L
        ou5 = self.decoder2(masked_output)  # B*C, 1, L
        ou5 = ou5.squeeze(1)
        ou5 = ou5[:,self.stride:-(rest+self.stride)].contiguous()  # B*C, 1, L
        #masks = torch.sigmoid(self.output(ou6))
        #masked_output = enc_output * masks  # B, C, N, L
        #ou6 = self.decoder(masked_output)  # B*C, 1, L
        #ou6 = ou6.squeeze(1)
        #ou6 = ou6[:,self.stride:-(rest+self.stride)].contiguous()  # B*C, 1, L
        print("ggg1",ou1.shape)        
        print("ggg2",ou2.shape)        
        #print("ggg3",ou3.shape)        
        print("ggg4",ou4.shape)        
        print("ggg5",ou5.shape)        
        #print("ggg",ou6.shape)        
        #return output1,output2,ou1,ou2,ou3,ou4,ou5,ou6
        return output1,output2,ou1,ou2,ou4,ou5
       
     

