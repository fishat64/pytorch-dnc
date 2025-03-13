#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from uuid import uuid4
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
import plotly.graph_objects as go
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

from dnc.lib import exp_loss, InputStorage, mse, criterion, ENDSYM, tensor2string

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


def generate_data(batch_size, length, maxlength, testoccurance=True, transposeInput=False, transposeOutput=False):
  input_data = np.zeros((batch_size, maxlength, maxlength), dtype=np.float32)
  target_output = np.zeros((batch_size, maxlength, maxlength), dtype=np.float32)
  sequence1 = np.random.binomial(1, 0.5, (batch_size, length, 1))
  sequence2 = np.random.binomial(1, 0.5, (batch_size, length, 1))

  if testoccurance: # test if the sequence is in the test data, replace if so
    for i in range(batch_size):
      input_test_data = np.zeros((1, maxlength, maxlength), dtype=np.float32)
      input_test_data[0, 0:length, 0:1] = sequence1[i] #first sequence
      input_test_data[0, length, 1] = ENDSYM  #pause
      input_test_data[0, length+1:length*2+1, 2:3] = sequence2[i] #second sequence
      input_test_data[0, length*2+1, 3] = ENDSYM  #pause
      while st.isSaved(input_test_data[0], flag="testData"):
        if np.random.binomial(1, 0.5, 1) == 1: # replace first sequence
          sequence1[i] = np.random.binomial(1, 0.5, (length, 1))
          input_test_data[0, 0:length, 0:1] = sequence1[i]
        else: # replace second sequence
          sequence2[i] = np.random.binomial(1, 0.5, (length, 1))
          input_test_data[0, length+1:length*2+1, 2:3] = sequence2[i]

  input_data[:, 0:length, 0:1] = sequence1 #first sequence
  input_data[:, length, 1] = ENDSYM  #pause
  input_data[:, length+1:length*2+1, 2:3] = sequence2 #second sequence
  input_data[:, length*2+1, 3] = ENDSYM  #pause
  if transposeInput:
    for i in range(batch_size):
      input_data[i] = input_data[i].T

  def calcsum(sequenceA, sequenceB): #calculate sum of two binary numbers
    sumsequence = np.zeros((batch_size, length + 1, length +1))
    assert len(sequenceA) == len(sequenceB)
    for k in range(len(sequenceA)):
      carry = 0 # carry bit
      for j in reversed(range(len(sequenceA[k]))):
            if sequenceA[k][j][0] == 1 and sequenceB[k][j][0] == 1: #1+1=10
                sumsequence[k][j+1][-1] = 0+carry
                carry = 1
            elif (sequenceA[k][j][0] == 1 and sequenceB[k][j][0] == 0) or (sequenceA[k][j][0] == 0 and sequenceB[k][j][0] == 1): #1+0=1 and 0+1=1
                if carry == 1:
                    sumsequence[k][j+1][-1] = 0
                    carry = 1
                else:
                    sumsequence[k][j+1][-1] = 1
                    carry = 0
            else:
                sumsequence[k][j+1][-1] = 0+carry #0+0=0
                carry = 0
      sumsequence[k][0][-1] = carry
    return sumsequence
  
  cs = calcsum(sequence1, sequence2)
  for i in range(batch_size):
    target_output[i, -(length+1):, -(length+1):] = cs[i] #write sum to target output
    if transposeOutput:
      target_output[i] = target_output[i].T

  return input_data, target_output




def combLoss(prediction, target):
  return mse(prediction, target)+exp_loss(prediction, target)

def incrementCurriculum(trainError, epoch, sequence_length, maxsequence_length, curriculum_fre):
  return epoch != 0 and sequence_length < maxsequence_length and epoch % curriculum_fre == 0


if __name__ == '__main__':

  name = 'add_' + str(uuid4().hex)[:2] + ': \n'
  
  datas = []

  dirname = os.path.dirname(__file__)
  ckpts_dir = os.path.join(dirname, 'checkpoints')
  if not os.path.isdir(ckpts_dir):
    os.mkdir(ckpts_dir)

  batch_size = 100
  sequence_length = 3
  sequence_max_length = 6 
  iterations = 10**4 #200000
  summarize_freq = 100
  check_freq = 100
  curriculum_freq = 1500


  # input_size = output_size = args.input_size
  mem_slot = 32
  mem_size = 1
  read_heads = 1
  curriculum_increment = 1
  input_size = 2*sequence_max_length + 2
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
        independent_linears=True,
        nonlinearity='sigmoid',
    )
  
  debprint(rnn)

  last_save_losses = []

  optimizer = optim.Adam(rnn.parameters(), lr=0.001, eps=1e-9, betas=[0.9, 0.98]) # 0.0001
 
  for i in range(3, sequence_max_length,1): # generate test data
    inputdataspace = 2**i*2 # 2 i bit sequences
    testdatasize = int(inputdataspace*0.15)+1 #15%
    input_data, target_output = generate_data(testdatasize, i, input_size)
    for i in range(testdatasize):
      st.saveInput(input_data[i], output=target_output[i], withoutIncrement=True, flag="testData") #saveData


  (chx, mhx, rv) = (None, None, None)
  Testloss = 0 # loss of test data
  for epoch in tqdm(range(iterations + 1)):
    llprint("\rIteration {ep}/{tot}".format(ep=epoch, tot=iterations))
    optimizer.zero_grad()


    input_data, target_output = generate_data(batch_size, sequence_length, input_size) # generate data
    input_data = var(T.from_numpy(input_data))
    target_output = var(T.from_numpy(target_output))


    if rnn.debug:
      output, (chx, mhx, rv), v = rnn(input_data, (None, mhx, None), reset_experience=True, pass_through_memory=True)
    else:
      output, (chx, mhx, rv) = rnn(input_data, (None, mhx, None), reset_experience=True, pass_through_memory=True)

    loss = combLoss((output), target_output)

    if epoch % 100 == 0:
      testset = st.getDataByFlag("testData") # get test data
      testlosses = []
      for k in range(int(len(testset) / batch_size)+1): # split testdata into batch_size chunks
        input_TEST_data = np.zeros((batch_size, input_size, input_size), dtype=np.float32)
        target_TEST_output = np.zeros((batch_size, input_size, input_size), dtype=np.float32)
        for i in range(batch_size):
          if i + k * batch_size < len(testset):
            input_TEST_data[i] = testset[k*batch_size+i]["input"]
            target_TEST_output[i] = testset[k*batch_size+i]["output"]
          else: # if there is not enough test data fill the remaining slots with random entries
            robj = random.choice(testset)
            input_TEST_data[i] = robj["input"]
            target_TEST_output[i] = robj["output"]

        input_TEST_data = var(T.from_numpy(input_TEST_data))
        target_TEST_output = var(T.from_numpy(target_TEST_output))
        if rnn.debug:
          TEST_output, _, _ = rnn(input_data, (None, mhx, None), reset_experience=True, pass_through_memory=True)
        else:
          TEST_output, _ = rnn(input_data, (None, mhx, None), reset_experience=True, pass_through_memory=True)

        ri = random.randint(0, batch_size-1)
        print("TEST_input: \n", tensor2string(input_TEST_data[ri]))
        print("TEST_output: \n", tensor2string(TEST_output[ri]))
        print("target_TEST_output: \n", tensor2string(target_TEST_output[ri]))
        print("CE Loss: ", str(mse(TEST_output[ri], target_TEST_output[ri]).item()))

        MyTestloss = combLoss((TEST_output), target_TEST_output).item() # calculate test loss
        testlosses.append(MyTestloss)
      Testloss = np.mean(testlosses) # calculate test loss mean

    datas.append({"epoch": epoch, "loss": loss.item(), "testloss": Testloss, "sequencelength": sequence_length}) #append to the datas df
    
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
      llprint("\n\tAvg. Test Loss: %.4f\n" % (Testloss))
      if np.isnan(loss):
        raise Exception('nan Loss')
      print("MSE Loss: ", str(mse(output, target_output).item()))
      print("CE Loss: ", str(criterion(output, target_output).item()))
      print("EXP Loss: ", str(exp_loss(output, target_output).item()))
      print("\n")

    if summarize and rnn.debug:
      last_save_losses = []

      viz.heatmap(
            v['memory'],
            opts=dict(
                xtickstep=10,
                ytickstep=2,
                title= name + 'Memory, t: ' + str(epoch) + ', loss: ' + str(loss),
                ylabel='layer * time',
                xlabel='mem_slot * mem_size'
            )
        )

      viz.heatmap(
            v['link_matrix'][-1].reshape(mem_slot, mem_slot),
            opts=dict(
                xtickstep=10,
                ytickstep=2,
                title=name + 'Link Matrix, t: ' + str(epoch) + ', loss: ' + str(loss),
                ylabel='mem_slot',
                xlabel='mem_slot'
            )
      )
     
      viz.heatmap(
            v['precedence'],
            opts=dict(
                xtickstep=10,
                ytickstep=2,
                title=name + 'Precedence, t: ' + str(epoch) + ', loss: ' + str(loss),
                ylabel='layer * time',
                xlabel='mem_slot'
            )
      )

    if incrementCurriculum(loss, epoch, sequence_length, sequence_max_length, curriculum_freq):
      sequence_length = sequence_length + curriculum_increment
      print("Increasing max length to " + str(sequence_length))

    if take_checkpoint:
      llprint("\nSaving Checkpoint ... "),
      check_ptr = os.path.join(ckpts_dir, 'step_{}.pth'.format(epoch))
      cur_weights = rnn.state_dict()
      T.save(cur_weights, check_ptr)
      llprint("Done!\n")

  df = pd.DataFrame(datas) # plot loss 
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=df["epoch"], y=df["loss"], mode='lines', name='Train Data'))
  fig.add_trace(go.Scatter(x=df["epoch"], y=df["testloss"], mode='lines', name='Test Data'))
  fig.update_layout(title='Losses', xaxis_title='Epoch', yaxis_title='Loss')
  fig.show()

  #stepByStep = { "stepByStep": True , }

  for i in range(100):
    llprint("\nIteration %d/%d" % (i, iterations))
    # We test now the learned generalization using sequence_max_length examples
    random_length = np.random.randint(2, sequence_length  + 1)
    input_data, target_output = generate_data(batch_size, random_length, input_size)

    input_data = var(T.from_numpy(input_data))
    target_output = var(T.from_numpy(target_output))

    if rnn.debug:
      output, (chx, mhx, rv), v = rnn(input_data, (None, mhx, None), reset_experience=True, pass_through_memory=True, stepByStep=False)
    else:
      output, (chx, mhx, rv) = rnn(input_data, (None, mhx, None), reset_experience=True, pass_through_memory=True, stepByStep=False)

    #output = output[:, -1, :].sum().data.cpu().numpy()
    #target_output = target_output.sum().data.cpu().numpy()

    print("\n\n")
    print("Input: ", tensor2string(input_data[0]))
    print("Output: ", tensor2string(torch.round(output[0], decimals=1)))
    print("Target: ", tensor2string(target_output[0]))
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

