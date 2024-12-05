#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import getopt
import sys
import os
import math
import time
import argparse
from visdom import Visdom
from tqdm import tqdm
import plotly.express as px
import pandas as pd

sys.path.insert(0, os.path.join('..', '..'))

import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
import torch.optim as optim

from torch.nn.utils import clip_grad_norm_

from dnc.dnc import DNC
from dnc.sdnc import SDNC
from dnc.sam import SAM
from dnc.util import *

from dnc.lib import exp_loss, InputStorage, mse, criterion

parser = argparse.ArgumentParser(description='PyTorch Differentiable Neural Computer')
parser.add_argument('-input_size', type=int, default=6, help='dimension of input feature')
parser.add_argument('-rnn_type', type=str, default='lstm', help='type of recurrent cells to use for the controller')
parser.add_argument('-nhid', type=int, default=64, help='number of hidden units of the inner nn')
parser.add_argument('-dropout', type=float, default=0, help='controller dropout')
parser.add_argument('-memory_type', type=str, default='dnc', help='dense or sparse memory: dnc | sdnc | sam')

parser.add_argument('-nlayer', type=int, default=1, help='number of layers')
parser.add_argument('-nhlayer', type=int, default=2, help='number of hidden layers')
parser.add_argument('-lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('-optim', type=str, default='adam', help='learning rule, supports adam|rmsprop')
parser.add_argument('-clip', type=float, default=50, help='gradient clipping')

parser.add_argument('-batch_size', type=int, default=100, metavar='N', help='batch size')
parser.add_argument('-mem_size', type=int, default=20, help='memory dimension')
parser.add_argument('-mem_slot', type=int, default=16, help='number of memory slots')
parser.add_argument('-read_heads', type=int, default=4, help='number of read heads')
parser.add_argument('-sparse_reads', type=int, default=10, help='number of sparse reads per read head')
parser.add_argument('-temporal_reads', type=int, default=2, help='number of temporal reads')

parser.add_argument('-sequence_max_length', type=int, default=4, metavar='N', help='sequence_max_length')
parser.add_argument('-curriculum_increment', type=int, default=0, metavar='N', help='sequence_max_length incrementor per 1K iterations')
parser.add_argument('-curriculum_freq', type=int, default=1000, metavar='N', help='sequence_max_length incrementor per 1K iterations')
parser.add_argument('-cuda', type=int, default=-1, help='Cuda GPU ID, -1 for CPU')

parser.add_argument('-iterations', type=int, default=100000, metavar='N', help='total number of iteration')
parser.add_argument('-summarize_freq', type=int, default=100, metavar='N', help='summarize frequency')
parser.add_argument('-check_freq', type=int, default=100, metavar='N', help='check point frequency')
parser.add_argument('-visdom', action='store_true', help='plot memory content on visdom per -summarize_freq steps')

args = parser.parse_args()
print(args)

viz = Visdom()
# assert viz.check_connection()

if args.cuda != -1:
  print('Using CUDA.')
  T.manual_seed(1111)
else:
  print('Using CPU.')


def llprint(message):
  sys.stdout.write(message)
  sys.stdout.flush()

def debprint(*args):
  DEBUG = True
  if DEBUG:
    print(*args)

st = InputStorage()


def generate_data(batch_size, length, maxlength, cuda=-1):
  length = length + 0

  input_data = np.zeros((batch_size, maxlength, maxlength), dtype=np.float32)
  target_output = np.zeros((batch_size, maxlength, maxlength), dtype=np.float32)


  sequence1 = np.random.binomial(1, 0.5, (batch_size, length, 1))
  sequence2 = np.random.binomial(1, 0.5, (batch_size, length, 1))

  input_data[:, 0:length, 0:1] = sequence1 #first sequence
  input_data[:, length, 1] = 9  #pause
  input_data[:, length+1:length*2+1, 2:3] = sequence2 #second sequence

  def calcsum(sequenceA, sequenceB): #calculate sum of two binary numbers
    sumsequence = np.zeros((batch_size, length + 1, length +1))
    assert len(sequenceA) == len(sequenceB)
    for k in range(len(sequenceA)):
      carry = 0 # carry bit
      for j in reversed(range(len(sequenceA[k]))):
            if sequenceA[k][j][0] == 1 and sequenceB[k][j][0] == 1:
                sumsequence[k][j+1][-1] = 0+carry
                carry = 1
            elif (sequenceA[k][j][0] == 1 and sequenceB[k][j][0] == 0) or (sequenceA[k][j][0] == 0 and sequenceB[k][j][0] == 1):
                if carry == 1:
                    sumsequence[k][j+1][-1] = 0
                    carry = 1
                else:
                    sumsequence[k][j+1][-1] = 1
                    carry = 0
            else:
                sumsequence[k][j+1][-1] = 0+carry
                carry = 0
      sumsequence[k][0][-1] = carry
    return sumsequence
  
  target_output[:, -(length+1):, -(length+1):] = calcsum(sequence1, sequence2) #write sum to target output
  return input_data, target_output




def combLoss(prediction, target):
  return exp_loss(prediction, target)



if __name__ == '__main__':

  datas = []

  dirname = os.path.dirname(__file__)
  ckpts_dir = os.path.join(dirname, 'checkpoints')
  if not os.path.isdir(ckpts_dir):
    os.mkdir(ckpts_dir)

  batch_size = 1
  sequence_length = 3
  sequence_max_length = 5
  iterations = 500000 #200000
  summarize_freq = 100
  check_freq = 100
  curriculum_freq = 2500


  # input_size = output_size = args.input_size
  mem_slot = 32
  mem_size = 1
  read_heads = 2
  curriculum_increment = 1
  input_size = 2*sequence_max_length + 1
  output_size = 64

  replaceWithWrong = True

  debprint(input_size, output_size)

  rnn = DNC(
        input_size=input_size,
        hidden_size=output_size,
        rnn_type='lstm',
        num_layers=1,
        num_hidden_layers=2,
        dropout=0,
        nr_cells=mem_slot,
        cell_size=mem_size,
        read_heads=read_heads,
        gpu_id=-1,
        debug='store_true',
        batch_first=True,
        independent_linears=True
    )
  
  debprint(rnn)

  last_save_losses = []

  optimizer = optim.Adam(rnn.parameters(), lr=0.001, eps=1e-9, betas=[0.9, 0.98]) # 0.0001
 

  (chx, mhx, rv) = (None, None, None)
  for epoch in tqdm(range(iterations + 1)):
    llprint("\rIteration {ep}/{tot}".format(ep=epoch, tot=iterations))
    optimizer.zero_grad()


    input_data, target_output = generate_data(batch_size, sequence_length, input_size, -1)

    if replaceWithWrong:
      errordatList = st.getHighestErrorInputs(int(batch_size/16)+2)
      for count, value in enumerate(errordatList):
        if input_data[count].shape[0] != value["input"].shape[0]:
          input_data[count] = np.zeros(input_data[count].shape, dtype=np.float32)
          target_output[count] = np.zeros(target_output[count].shape, dtype=np.float32)
          i = 0
          offset = 1
          while i < value["input"].shape[0]:
            input_data[count][i+offset] = value["input"][i]
            if value["input"][i] == 9:
              offset = offset + 1
            i = i+1
          i = 0
          offset = 2
          while i < value["target"].shape[0]:
            target_output[count][i+offset] = value["target"][i]
            i = i+1
          st.removeInputWithError(value["input"])
        else:
          input_data[count] = value["input"]
          target_output[count] = value["target"]

    input_data = var(T.from_numpy(input_data))
    target_output = var(T.from_numpy(target_output))


    if rnn.debug:
      output, (chx, mhx, rv), v = rnn(input_data, (None, mhx, None), reset_experience=True, pass_through_memory=True)
    else:
      output, (chx, mhx, rv) = rnn(input_data, (None, mhx, None), reset_experience=True, pass_through_memory=True)


    for i in range(batch_size):
      st.saveInputWithError(input_data[i].numpy(), target_output[i].numpy(), combLoss(output[i], target_output[i]).item())


    loss = combLoss((output), target_output)

    datas.append({"epoch": epoch, "loss": loss.item(), "sequencelength": sequence_length})

    loss.backward()

    T.nn.utils.clip_grad_norm_(rnn.parameters(), 50)
    optimizer.step()
    loss_value = loss.item()

    summarize = (epoch % summarize_freq == 0)
    take_checkpoint = (epoch != 0) and (epoch % check_freq == 0)
    

    # detach memory from graph
    mhx = { k : (v.detach() if isinstance(v, var) else v) for k, v in mhx.items() }

    last_save_losses.append(loss_value)
    loss = np.mean(last_save_losses)

    if summarize:
      llprint("\n\tAvg. Loss: %.4f\n" % (loss))
      if np.isnan(loss):
        raise Exception('nan Loss')
      print("CE Loss: ", str(mse(output, target_output).item()))
      print("Log Loss: ", str(criterion(output, target_output).item()))
      print("Exp Loss: ", str(exp_loss(output, target_output).item()))
      print("\n")

    if summarize and rnn.debug:
      last_save_losses = []

      viz.heatmap(
            v['memory'],
            opts=dict(
                xtickstep=10,
                ytickstep=2,
                title='Memory, t: ' + str(epoch) + ', loss: ' + str(loss),
                ylabel='layer * time',
                xlabel='mem_slot * mem_size'
            )
        )

      viz.heatmap(
            v['link_matrix'][-1].reshape(mem_slot, mem_slot),
            opts=dict(
                xtickstep=10,
                ytickstep=2,
                title='Link Matrix, t: ' + str(epoch) + ', loss: ' + str(loss),
                ylabel='mem_slot',
                xlabel='mem_slot'
            )
      )
     
      viz.heatmap(
            v['precedence'],
            opts=dict(
                xtickstep=10,
                ytickstep=2,
                title='Precedence, t: ' + str(epoch) + ', loss: ' + str(loss),
                ylabel='layer * time',
                xlabel='mem_slot'
            )
      )


    increment_curriculum = (epoch != 0) and (epoch % curriculum_freq == 0) and (sequence_length < sequence_max_length)

    if increment_curriculum:
      sequence_length = sequence_length + curriculum_increment
      print("Increasing max length to " + str(sequence_length))

    if take_checkpoint:
      llprint("\nSaving Checkpoint ... "),
      check_ptr = os.path.join(ckpts_dir, 'step_{}.pth'.format(epoch))
      cur_weights = rnn.state_dict()
      T.save(cur_weights, check_ptr)
      llprint("Done!\n")

  df = pd.DataFrame(datas)
  fig = px.scatter(df, x="epoch", y="loss", color="sequencelength", trendline="ols")
  fig.show()

  for i in range(int((iterations + 1) / 10)):
    llprint("\nIteration %d/%d" % (i, iterations))
    # We test now the learned generalization using sequence_max_length examples
    random_length = np.random.randint(2, sequence_length  + 1)
    input_data, target_output = generate_data(batch_size, random_length, 1, -1)

    if rnn.debug:
      output, (chx, mhx, rv), v = rnn(input_data, (None, mhx, None), reset_experience=True, pass_through_memory=True)
    else:
      output, (chx, mhx, rv) = rnn(input_data, (None, mhx, None), reset_experience=True, pass_through_memory=True)

    output = output[:, -1, :].sum().data.cpu().numpy()[0]
    target_output = target_output.sum().data.cpu().numpy()

    print("\n\n")
    print("Input: ", torch.flatten(input_data[0]))
    print("Output: ", torch.flatten(torch.round(output[0], decimals=1)))
    print("Target: ", torch.flatten(target_output[0]))
    print("CE Loss: ", str(mse(output[0], target_output[0]).item()))
    print("Log Loss: ", str(criterion(output[0], target_output[0]).item()))
    print("Exp Loss: ", str(exp_loss(output[0], target_output[0]).item()))
    print("\n\n")
    print("CE Loss: ", str(mse(output, target_output).item()))
    print("Log Loss: ", str(criterion(output, target_output).item()))
    print("Exp Loss: ", str(exp_loss(output, target_output).item()))
    print("\n\n")

    try:
      print("\nReal value: ", ' = ' + str(int(target_output[0])))
      print("Predicted:  ", ' = ' + str(int(output // 1)) + " [" + str(output) + "]")
    except Exception as e:
      pass

