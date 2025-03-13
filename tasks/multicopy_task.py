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

import pandas as pd
import plotly.graph_objects as go

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

from dnc.lib import *

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


st = InputStorage()

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
  DEBUG = False
  if DEBUG:
    print(*args)


def generate_data(batch_size, length, maxnumberofcopies=3, currentmaxnocopies=3, testoccurance=True):
  length=length+1
  size=1 # ignore size
  
  numberOfCopies = np.random.randint(low=2, high=currentmaxnocopies, size=(batch_size, 1))


  input_data = np.zeros((batch_size, length*maxnumberofcopies, size), dtype=np.float32)
  target_output = np.zeros((batch_size, length*maxnumberofcopies, size), dtype=np.float32)

  sequence = np.random.binomial(1, 0.5, (batch_size, length-1, size))



  print(batch_size, length, size, maxnumberofcopies, currentmaxnocopies)

  finputdata = np.zeros((batch_size, length*maxnumberofcopies, size))
  finputdata[:, :length-1, :] = sequence
  finputdata[:, length-1, :] = ENDSYM  # the end symbol
  finputdata[:, -2, :] = ENDSYM
  finputdata[:, -1, :] = numberOfCopies  # the end symbol

  for i in range(batch_size): # assure no empty sequences
    while np.sum(sequence[i]) == 0 or (testoccurance and st.isSaved(finputdata[i], flag="testData")):
      sequence[i] = np.random.binomial(1, 0.5, (1, length-1, size))
    
  input_data[:, :length-1, :] = sequence
  input_data[:, length-1, :] = ENDSYM  # the end symbol
  input_data[:, -2, :] = ENDSYM
  input_data[:, -1, :] = numberOfCopies  # the end symbol

  for i in range(batch_size): # save input data to avoid intersection between train, test and validation data
    for j in range(numberOfCopies[i][0]):
      target_output[i, (j*length)-j:((j+1)*length) - (j+1), :] = sequence[i]

  debprint("inputdata:", input_data.shape)
  debprint("outpudtata:",target_output.shape)


  return input_data, target_output 


def combLoss(output, target):
  return mse(output, target)

def incrementCurriculum(trainError, epoch, sequence_length, maxsequence_length, curriculum_fre):
  return epoch != 0 and sequence_length < maxsequence_length and epoch % curriculum_fre == 0


if __name__ == '__main__':

  dirname = os.path.dirname(__file__)
  ckpts_dir = os.path.join(dirname, 'checkpoints')
  if not os.path.isdir(ckpts_dir):
    os.mkdir(ckpts_dir)

  datas = []

  name = 'mc_ ' + str(uuid4().hex)[:3] + ': \n'

  batch_size = 1
  sequence_max_length = 4
  iterations = 200000
  summarize_freq = 100
  check_freq = 100
  curriculum_freq = 5000
  curriculumMaxNoCopies_freq = 1000


  # input_size = output_size = args.input_size
  mem_slot = 16 # number of memory slots
  mem_size = 1 # size of each memory slot
  read_heads = 1


  maxnumberofcopies=7
  currentmaxnocopies=3

  input_length = 6
  input_size = 1#input_length*maxnumberofcopies #+10 memory = input
  output_size = 64

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
  
  print(rnn)

  last_save_losses = []

  optimizer = optim.Adam(rnn.parameters(), lr=0.001, eps=1e-9, betas=[0.9, 0.98]) # 0.0001

  print(sequence_max_length)

  for i in range(1, sequence_max_length+1):
    inputdataspace = 2**i*maxnumberofcopies # i bit numbers amd up to maxnumberofcopies
    testdatasize = int(inputdataspace*0.1)+1 # at least 1
    input_data, target_output = generate_data(testdatasize, i, maxnumberofcopies=maxnumberofcopies, currentmaxnocopies=maxnumberofcopies, testoccurance=False)

    for i in range(testdatasize):
      st.saveInput(input_data[i], output=target_output[i], withoutIncrement=True, flag="testData") 



  (chx, mhx, rv) = (None, None, None)
  Testloss = 0
  for epoch in tqdm(range(iterations + 1)):
    llprint("\rIteration {ep}/{tot}".format(ep=epoch, tot=iterations))
    optimizer.zero_grad()

    random_length = np.random.randint(1, sequence_max_length + 1)
    input_data, target_output = generate_data(batch_size, random_length, maxnumberofcopies=maxnumberofcopies, currentmaxnocopies=currentmaxnocopies)
    input_data = var(T.from_numpy(input_data))
    target_output = var(T.from_numpy(target_output)) 
    # generate test data

    if rnn.debug:
      output, (chx, mhx, rv), v = rnn(input_data, (None, mhx, None), reset_experience=True, pass_through_memory=True)
    else:
      output, (chx, mhx, rv) = rnn(input_data, (None, mhx, None), reset_experience=True, pass_through_memory=True)


    loss = combLoss((output), target_output)

    if epoch % 100 == 0: # calculate test loss
      testset = st.getDataByFlag("testData") # get test data
      testlosses = []

      for k in range(int(len(testset) / batch_size)+1): # split to batches
        input_TEST_data = np.zeros((batch_size, (sequence_max_length+1)*maxnumberofcopies, 1), dtype=np.float32)
        target_TEST_output = np.zeros((batch_size, (sequence_max_length+1)*maxnumberofcopies, 1), dtype=np.float32)
        for i in range(batch_size):
          if i + k * batch_size < len(testset):
            sh1 = testset[k*batch_size+i]["input"].shape[0]
            sh2 = testset[k*batch_size+i]["output"].shape[0]
            input_TEST_data[i,:sh1] = testset[k*batch_size+i]["input"]
            target_TEST_output[i,:sh2] = testset[k*batch_size+i]["output"]
          else: # fill batch with random elements
            rel = random.choice(testset)
            sh1 = rel["input"].shape[0]
            sh2 = rel["output"].shape[0]
            input_TEST_data[i, :sh1] = rel["input"]
            target_TEST_output[i, :sh1] = rel["output"]
        input_TEST_data = var(T.from_numpy(input_TEST_data))
        target_TEST_output = var(T.from_numpy(target_TEST_output))

        TEST_output = np.zeros(target_TEST_output.shape)
        if rnn.debug:
          eTEST_output, _, _ = rnn(input_data, (None, mhx, None), reset_experience=True, pass_through_memory=True)
        else:
          eTEST_output, _ = rnn(input_data, (None, mhx, None), reset_experience=True, pass_through_memory=True)

        TEST_output[:eTEST_output.shape[0], :eTEST_output.shape[1], :eTEST_output.shape[2]] = eTEST_output.data.cpu().numpy()
        TEST_output = var(T.from_numpy(TEST_output))
        MyTestloss = combLoss((TEST_output), target_TEST_output).item() # calculate test loss
        testlosses.append(MyTestloss)
      Testloss = np.mean(testlosses) # calculate average

    datas.append({"epoch": epoch, "loss": loss.item(), "testloss": Testloss, "sequencelength": input_length})



    loss.backward()

    T.nn.utils.clip_grad_norm_(rnn.parameters(), 50)
    optimizer.step()
    loss_value = loss.item()

    summarize = (epoch % summarize_freq == 0)
    take_checkpoint = (epoch != 0) and (epoch % check_freq == 0)
    

    # detach memory from graph
    mhx = { k : (v.detach() if isinstance(v, var) else v) for k, v in mhx.items() }

    last_save_losses.append(loss_value)

    if summarize:
      loss = np.mean(last_save_losses)
      llprint("\n\tAvg. Loss: %.4f\n" % (loss))
      if np.isnan(loss):
        raise Exception('nan Loss')

    if summarize and rnn.debug:
      print(v.keys())

      loss = np.mean(last_save_losses)
      last_save_losses = []

      viz.heatmap(
            v['memory'],
            opts=dict(
                xtickstep=10,
                ytickstep=2,
                title=name + 'Memory, t: ' + str(epoch) + ', loss: ' + str(loss),
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

    if incrementCurriculum(loss, epoch, currentmaxnocopies, maxnumberofcopies, curriculumMaxNoCopies_freq):
      currentmaxnocopies = currentmaxnocopies + 1
      if currentmaxnocopies > maxnumberofcopies:
        currentmaxnocopies = maxnumberofcopies
      print("Increasing max number of copies to " + str(currentmaxnocopies))


    if incrementCurriculum(loss, epoch, sequence_max_length, sequence_max_length + 1, curriculum_freq):
      sequence_max_length = sequence_max_length + 1
      print("Increasing max length to " + str(sequence_max_length))

    if take_checkpoint:
      llprint("\nSaving Checkpoint ... "),
      check_ptr = os.path.join(ckpts_dir, 'step_{}.pth'.format(epoch))
      cur_weights = rnn.state_dict()
      T.save(cur_weights, check_ptr)
      llprint("Done!\n")
    
    if summarize:
      random_length = np.random.randint(2, sequence_max_length + 1)
      input_data, target_output = generate_data(batch_size, random_length,  maxnumberofcopies=maxnumberofcopies, currentmaxnocopies=currentmaxnocopies)

      input_data = var(T.from_numpy(input_data))
      target_output = var(T.from_numpy(target_output))

      if rnn.debug:
        output, (chx, mhx, rv), v = rnn(input_data, (None, mhx, None), reset_experience=True, pass_through_memory=True)
      else:
        output, (chx, mhx, rv) = rnn(input_data, (None, mhx, None), reset_experience=True, pass_through_memory=True)

      print("\n\n")
      print("Input: ", torch.flatten(input_data[0]))
      print("Output: ", torch.flatten(torch.round(output[0], decimals=1)))
      print("Target: ", torch.flatten(target_output[0]))
      print("MSE Loss: ", str(mse(output, target_output).item()))
      print("CE Loss: ", str(criterion(output, target_output).item()))
      print("EXP Loss: ", str(exp_loss(output, target_output).item()))
      print("\n\n")

  df = pd.DataFrame(datas)
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=df["epoch"], y=df["loss"], mode='lines', name='Train Data'))
  fig.add_trace(go.Scatter(x=df["epoch"], y=df["testloss"], mode='lines', name='Test Data'))
  fig.update_layout(title='Losses', xaxis_title='Epoch', yaxis_title='Loss')
  fig.show()


  input_data, target_output = generate_data(1, random_length, maxnumberofcopies=maxnumberofcopies, currentmaxnocopies=maxnumberofcopies)

  input_data = var(T.from_numpy(input_data))
  target_output = var(T.from_numpy(target_output))
  print(input_data)
  print(target_output)
  output, (chx, mhx, rv), v = rnn(input_data, (None, mhx, None), reset_experience=True, pass_through_memory=True, stepByStep=True)



  for i in range(10):#range(int((iterations + 1) / 100)):
    llprint("\nIteration %d/%d" % (i, iterations))
    # We test now the learned generalization using sequence_max_length examples
    random_length = np.random.randint(2, sequence_max_length + 1)
    input_data, target_output = generate_data(batch_size, random_length, maxnumberofcopies=maxnumberofcopies, currentmaxnocopies=maxnumberofcopies)

    input_data = var(T.from_numpy(input_data))
    target_output = var(T.from_numpy(target_output))
    if rnn.debug:
      output, (chx, mhx, rv), v = rnn(input_data, (None, mhx, None), reset_experience=True, pass_through_memory=True)
    else:
      output, (chx, mhx, rv) = rnn(input_data, (None, mhx, None), reset_experience=True, pass_through_memory=True)

    print("\n\n")
    print("Input: ", torch.flatten(input_data[0]))
    print("Output: ", torch.flatten(torch.round(output[0], decimals=1)))
    print("Target: ", torch.flatten(target_output[0]))
    print("MSE Loss: ", str(mse(output, target_output).item()))
    print("CE Loss: ", str(criterion(output, target_output).item()))
    print("EXP Loss: ", str(exp_loss(output, target_output).item()))
    print("\n\n")
    output = output[:, -1, :].sum().data.cpu().numpy()
    target_output = target_output.sum().data.cpu().numpy()

    
    try:
      print("\nReal value: ", ' = ' + str(int(target_output[0])))
      print("Predicted:  ", ' = ' + str(int(output // 1)) + " [" + str(output) + "]")
    except Exception as e:
      pass

