import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import config
import sys

# TODO: multiloss, Deep fusion

class cLN(nn.Module):
    def __init__(self, dimension, eps = 1e-8, trainable=True):
        super(cLN, self).__init__()
        
        self.eps = eps
        if trainable:
            self.gain = nn.Parameter(torch.ones(1, dimension, 1))
            self.bias = nn.Parameter(torch.zeros(1, dimension, 1))
        else:
            self.gain = Variable(torch.ones(1, dimension, 1), requires_grad=False)
            self.bias = Variable(torch.zeros(1, dimension, 1), requires_grad=False)

    def forward(self, input):
        # input size: (Batch, Freq, Time)
        # cumulative mean for each time step
        
        batch_size = input.size(0)
        channel = input.size(1)
        time_step = input.size(2)
        
        step_sum = input.sum(1)  # B, T
        step_pow_sum = input.pow(2).sum(1)  # B, T
        cum_sum = torch.cumsum(step_sum, dim=1)  # B, T
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=1)  # B, T
        
        entry_cnt = np.arange(channel, channel*(time_step+1), channel)
        entry_cnt = torch.from_numpy(entry_cnt).type(input.type())
        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)
        
        cum_mean = cum_sum / entry_cnt  # B, T
        cum_var = (cum_pow_sum - 2*cum_mean*cum_sum) / entry_cnt + cum_mean.pow(2)  # B, T
        cum_std = (cum_var + self.eps).sqrt()  # B, T
        
        cum_mean = cum_mean.unsqueeze(1)
        cum_std = cum_std.unsqueeze(1)
        
        x = (input - cum_mean.expand_as(input)) / cum_std.expand_as(input)
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())
            
            
class DepthConv1d(nn.Module):

    def __init__(self, input_channel, hidden_channel, kernel, padding, dilation=1, skip=True, causal=False):
        super(DepthConv1d, self).__init__()
        
        self.causal = causal
        self.skip = skip
        
        self.conv1d = nn.Conv1d(input_channel, hidden_channel, 1)
        if self.causal:
            self.padding = (kernel - 1) * dilation
        else:
            self.padding = padding
        self.dconv1d = nn.Conv1d(hidden_channel, hidden_channel, kernel, dilation=dilation,
          groups=hidden_channel,
          padding=self.padding)
        self.res_out = nn.Conv1d(hidden_channel, input_channel, 1)
        self.nonlinearity1 = nn.PReLU()
        self.nonlinearity2 = nn.PReLU()
        if self.causal:
            if config.mode_LN == 'cLN':
                self.reg1 = cLN(hidden_channel, eps=1e-08)
                self.reg2 = cLN(hidden_channel, eps=1e-08)
                print('use cLN')
            else:
                self.reg1 = nn.BatchNorm1d(hidden_channel, eps=1e-08)
                self.reg2 = nn.BatchNorm1d(hidden_channel, eps=1e-08)
                print('use BN')
        else:
            self.reg1 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
            self.reg2 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
        
        if self.skip:
            self.skip_out = nn.Conv1d(hidden_channel, input_channel, 1)

    def forward(self, input):
        output = self.reg1(self.nonlinearity1(self.conv1d(input)))
        if self.causal:
            output = self.reg2(self.nonlinearity2(self.dconv1d(output)[:,:,:-self.padding]))
        else:
            output = self.reg2(self.nonlinearity2(self.dconv1d(output)))
        residual = self.res_out(output)
        if self.skip:
            skip = self.skip_out(output)
            return residual, skip
        else:
            return residual

        
class TCN(nn.Module):
    def __init__(self, BN_dim, hidden_dim,
                 layer, stack, num_spk, kernel=3, skip=config.SKIP, 
                 causal=False, dilated=True):
        super(TCN, self).__init__()
        
        # input is a sequence of features of shape (B, N, L)
        # TCN for feature extraction
        self.receptive_field = 0
        self.dilated = dilated
        self.num_spk = config.mix_spk
        vs = 128
        if config.FUSION_POSITION == 'ALL':
            if config.MODAL_FUSION == 'DF' or config.MODAL_FUSION == 'DAF':
                self.fc = nn.ModuleList([
                nn.Linear(128+2*2*config.VISUAL_DIM, 128, bias=True),
                nn.Linear(128+2*2*config.VISUAL_DIM, 128, bias=True),
                nn.Linear(128+2*2*config.VISUAL_DIM, 128, bias=True),
                nn.Linear(128+2*2*config.VISUAL_DIM, 128, bias=True)])
                self.voicefc1 = nn.ModuleList([
                    nn.Linear(128+vs+2*2*config.VISUAL_DIM, 128, bias=True),
                    nn.Linear(128+vs+2*2*config.VISUAL_DIM, 128, bias=True),
                    nn.Linear(128+vs+2*2*config.VISUAL_DIM, 128, bias=True),
                    nn.Linear(128+vs+2*2*config.VISUAL_DIM, 128, bias=True)])
            else:
                self.fc = nn.ModuleList([
                    nn.Linear(128+2*config.VISUAL_DIM, 128, bias=True),
                    nn.Linear(128+2*config.VISUAL_DIM, 128, bias=True),
                    nn.Linear(128+2*config.VISUAL_DIM, 128, bias=True),
                    nn.Linear(128+2*config.VISUAL_DIM, 128, bias=True)])
                self.voicefc = nn.ModuleList([
                    nn.Linear(128*2+2*config.VISUAL_DIM, 128, bias=True),
                    nn.Linear(128*2+2*config.VISUAL_DIM, 128, bias=True),
                    nn.Linear(128*2+2*config.VISUAL_DIM, 128, bias=True),
                    nn.Linear(128*2+2*config.VISUAL_DIM, 128, bias=True)])
        else:
            if config.MODAL_FUSION == 'DF' or config.MODAL_FUSION == 'DAF':
                self.fc = nn.Linear(128+2*2*config.VISUAL_DIM, 128, bias=True)
            else:
                self.fc = nn.Linear(128+2*config.VISUAL_DIM, 128, bias=True)
        self.TCN = nn.ModuleList([])
        for s in range(stack):
            for i in range(layer):
                if self.dilated:
                    self.TCN.append(DepthConv1d(BN_dim, hidden_dim, kernel, dilation=2**i, padding=2**i, skip=skip, causal=causal)) 
                else:
                    self.TCN.append(DepthConv1d(BN_dim, hidden_dim, kernel, dilation=1, padding=1, skip=skip, causal=causal))   
                if i == 0 and s == 0:
                    self.receptive_field += kernel
                else:
                    if self.dilated:
                        self.receptive_field += (kernel - 1) * 2**i
                    else:
                        self.receptive_field += (kernel - 1)
                    
        #print("Receptive field: {:3d} frames.".format(self.receptive_field))
        self.skip = skip

    def forward(self, input, query,voicen=1,voice=None):
        print("gggg",voice.shape)
        #print(query.shape)
        # input shape: (B, N, L)
        # pass to TCN
        # 确定特征融合的位置
        if config.FUSION_POSITION == '8':
            fusion_position = ['8']
        elif config.FUSION_POSITION == '16':
            fusion_position = ['16']
        elif config.FUSION_POSITION == '0':
            fusion_position = ['0']
        else:
            fusion_position = ['0', '8', '16','24']
        cal_position = ['8', '16','24']
        output = input
        out = []
        if self.skip:
            skip_connection = 0.
            for i in range(len(self.TCN)):
                if str(i) in fusion_position:
                    if config.MODAL_FUSION == 'CAT':
                        output = torch.cat((output, query), dim=1)
                        if voicen==1:
                            #print("query",query.size())
                            #print("query",output.size())
                            #print("query",voice.size())
                            output=torch.cat((output,voice),dim=1)
                        output = output.transpose(1, 2)
                        if len(fusion_position) > 1:
                            if voicen==0:
                                output = self.fc[i//8](output)
                            else:
                                #print("output",i,output.size())
                                output=self.voicefc[i//8](output) 
                        else:
                            output = self.fc(output)
                        output = output.transpose(1, 2)
                    if config.MODAL_FUSION == 'DF':
                        shape = output.shape
                        voiceshape = 128
                        output = output.view(-1, self.num_spk, shape[1], shape[2])
                        qshape = query.shape
                        queryy = query.view(-1, self.num_spk, qshape[1], qshape[2]) # shape:[B,SPK,N,L]
                        if config.mix_spk == 2:
                        #    print(output.shape)
                        #    print(voice.unsqueeze(0).shape)
                        #    print(queryy.shape)
                            if voicen==1:
                                output0 = torch.cat((output[:,0,:,:],queryy[:,0,:,:],queryy[:,1,:,:],voice.unsqueeze(0)[:,0,:,:]), dim=1)
                                output1 = torch.cat((output[:,1,:,:],queryy[:,1,:,:],queryy[:,0,:,:],voice.unsqueeze(0)[:,1,:,:]), dim=1)
                                output = torch.cat((output0.unsqueeze(1), output1.unsqueeze(1)), dim=1)
                                output = output.view(-1, shape[1]+voiceshape+2*qshape[1], shape[2])
                            else:
                                output0 = torch.cat((output[:,0,:,:],queryy[:,0,:,:],queryy[:,1,:,:]), dim=1)
                                output1 = torch.cat((output[:,1,:,:],queryy[:,1,:,:],queryy[:,0,:,:]), dim=1)
                                output = torch.cat((output0.unsqueeze(1), output1.unsqueeze(1)), dim=1)
                                output = output.view(-1, shape[1]+2*qshape[1], shape[2])
                        if config.mix_spk == 3:
                            output0 = torch.cat((output[:,0,:,:],query[:,0,:,:],query[:,1,:,:]+query[:,2,:,:]), dim=1)
                            output1 = torch.cat((output[:,1,:,:],query[:,1,:,:],query[:,0,:,:]+query[:,2,:,:]), dim=1)
                            output2 = torch.cat((output[:,2,:,:],query[:,2,:,:],query[:,0,:,:]+query[:,1,:,:]), dim=1)
                            output = torch.cat((output0.unsqueeze(1), output1.unsqueeze(1), output2.unsqueeze(1)), dim=1)
                            output = output.view(-1, shape[1]+2*qshape[1], shape[2])
                        output = output.transpose(1, 2)
                        if len(fusion_position) > 1:
                            if voicen==0:
                                output = self.fc[i//8](output)
                            else:
                                output = self.voicefc1[i//8](output)
                        else:
                            output = self.voicefc1(output)
                        output = output.transpose(1, 2)
                    if config.MODAL_FUSION == 'DAF':
                        # output & query's shape:[B,N,L]
                        output = output.transpose(1, 2) # output's shape:[B,L,N]
                        D = torch.matmul(output, query)
                        A = F.softmax(D, dim=1)
                        query = query.transpose(1, 2)
                        new_query = torch.matmul(A, query) # shape:[B,L,N]
                        output = torch.cat((output,query,new_query), dim=2)
                        if len(fusion_position) > 1:
                            output = self.fc[i//8](output)
                        else:
                            output = self.fc(output)
                        output = output.transpose(1, 2)
                residual, skip = self.TCN[i](output)
                output = output + residual
                if str(i)=='8':
                    out1=output
                if str(i)=='16':
                    out2=output
                if str(i)=='24':
                    out3=output
                #if str(i) in fusion_position:
                 #   out.append(output) 
                skip_connection = skip_connection + skip
        else:
            for i in range(len(self.TCN)):
                if str(i) in fusion_position:
                    output = torch.cat((output, query), dim=1)
                    output = output.transpose(1, 2)
                    if len(fusion_position) > 1:
                        output = self.fc[i//8](output)
                    else:
                        output = self.fc(output)
                    output = output.transpose(1, 2)
                residual = self.TCN[i](output)
                output = output + residual
            
        # output layer
        if self.skip:
            output = skip_connection
        else:
            output = output
        #out = torch.from_numpy(np.array(out))
        return output,out1,out2
