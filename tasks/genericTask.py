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



viz = Visdom()
# assert viz.check_connection()



def llprint(message):
  sys.stdout.write(message)
  sys.stdout.flush()

def debprint(*args):
  DEBUG = False
  if DEBUG:
    print(*args)




def combLoss(output, target):
  return mse(output, target)


def generateDataExample(batchsize, input_sizeP, input_lengthP, datalength, maxdatalength, opts, testData=False, st=None):
  input_data = np.zeros((batchsize, input_lengthP, input_sizeP))
  # input_sizeP parallel inputs
  target_output = np.zeros((batchsize, input_lengthP, input_sizeP))
  sequence = np.random.binomial(1, 0.5, (batchsize, datalength, input_sizeP))
  
  input_data[:, :datalength+1, 0] = sequence
  input_data[:, datalength+1, 0] = -9
  input_data[:, -2, 0] = -9
  input_data[:, -1, 0] = np.random.randint(low=2, high=opts["maxcopies"], size=(batchsize, 1))
  for i in range(batchsize):
    while not testData and st.isSaved(input_data[i], flag="testData"):
      newsequence = np.random.binomial(1, 0.5, (1, datalength, input_sizeP))
      input_data[:, :datalength+1, :] = newsequence

  for i in range(batchsize):
    seq = input_data[i, :datalength+1, 0]
    copy = input_data[i, -1, 0]
    for j in range(int(copy)):
        target_output[:,((datalength)*j):((datalength)*(j+1)+1),0] = seq
    st.saveInput(input_data[i], output=target_output[i], withoutIncrement=True, flag="testData")


  


def genericTask(name, generateData, opts):
  dirname = os.path.dirname(__file__)
  ckpts_dir = os.path.join(dirname, 'checkpoints')
  if not os.path.isdir(ckpts_dir):
    os.mkdir(ckpts_dir)

  datas = []

  name = 'mc_ ' + str(uuid4().hex)[:3] + ': \n'

  batch_size = 100
  iterations = 20000
  summarize_freq = 100
  check_freq = 100
  curriculum_freq = 5000
  curriculumMaxNoCopies_freq = 1000


  # input_size = output_size = args.input_size
  mem_slot = 16 # number of memory slots
  mem_size = 1 # size of each memory slot
  read_heads = 1
  curriculum_increment = 1

  input_size = 1#input_length*maxnumberofcopies #+10 memory = input
  output_size = 64

  start_length = 1
  end_length = 10

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

  st = InputStorage()
  maxdatalength = end_length + 2
  for i in range(start_length, end_length):
    inputdataspace = 2**i*3
    testdatasize = int(inputdataspace*0.1)+1
    generateData(testdatasize, input_size, maxdatalength, i, end_length, testData=True, st=st)

  testhashset = st.getHashSet("testData")
  print("Test data size: ", len(testhashset))
 

  (chx, mhx, rv) = (None, None, None)
  Testloss = 0
  for epoch in tqdm(range(iterations + 1)):
    llprint("\rIteration {ep}/{tot}".format(ep=epoch, tot=iterations))
    optimizer.zero_grad()

    random_length = np.random.randint(1, sequence_max_length + 1)

    input_data, target_output = generateData(batch_size, input_size, maxdatalength, start_length, end_length, testData=False, st=st)

    input_data = var(T.from_numpy(input_data))
    target_output = var(T.from_numpy(target_output))

    debprint(input_data.shape, target_output.shape)

    if rnn.debug:
      output, (chx, mhx, rv), v = rnn(input_data, (None, mhx, None), reset_experience=True, pass_through_memory=True)
    else:
      output, (chx, mhx, rv) = rnn(input_data, (None, mhx, None), reset_experience=True, pass_through_memory=True)

    debprint(output.shape, target_output.shape)

    debprint("input_data:", input_data[0, :, :].T)
    debprint("target_output:", target_output[0, :, :].T)
    debprint("output:", output[0, :, :].T)

    loss = combLoss((output), target_output)

    if epoch % 100 == 0:
      testset = st.getDataByFlag("testData") # get test data
      #print(testset)

      testlosses = []

      for k in range(int(len(testset) / batch_size)+1):
        input_TEST_data = np.zeros((batch_size, (sequence_max_length+1)*maxnumberofcopies, 1), dtype=np.float32)
        target_TEST_output = np.zeros((batch_size, (sequence_max_length+1)*maxnumberofcopies, 1), dtype=np.float32)
        #input_TEST_data = np.zeros(input_data.shape)
        #target_TEST_output = np.zeros(target_output.shape)
        for i in range(batch_size):
          if i + k * batch_size < len(testset):
            sh1 = testset[k*batch_size+i]["input"].shape[0]
            sh2 = testset[k*batch_size+i]["output"].shape[0]
            input_TEST_data[i,:sh1] = testset[k*batch_size+i]["input"]
            target_TEST_output[i,:sh2] = testset[k*batch_size+i]["output"]
          else:
            rel = random.choice(testset)
            sh1 = rel["input"].shape[0]
            sh2 = rel["output"].shape[0]

            input_TEST_data[i, :sh1] = rel["input"]
            target_TEST_output[i, :sh1] = rel["output"]
        #print("i", input_TEST_data)
        #print("t", target_TEST_output)
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
      #print(testlosses)
      Testloss = np.mean(testlosses)


      
    datas.append({"epoch": epoch, "loss": loss.item(), "testloss": Testloss, "sequencelength": input_length})



    loss.backward()

    T.nn.utils.clip_grad_norm_(rnn.parameters(), 50)
    optimizer.step()
    loss_value = loss.item()

    summarize = (epoch % summarize_freq == 0)
    take_checkpoint = (epoch != 0) and (epoch % check_freq == 0)
    increment_curriculum = (epoch != 0) and (epoch % curriculum_freq == 0)

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

    if epoch != 0 and epoch % curriculumMaxNoCopies_freq == 0:
      currentmaxnocopies = currentmaxnocopies + 1
      if currentmaxnocopies > maxnumberofcopies:
        currentmaxnocopies = maxnumberofcopies
      print("Increasing max number of copies to " + str(currentmaxnocopies))


    if increment_curriculum:
      sequence_max_length = sequence_max_length + curriculum_increment
      print("Increasing max length to " + str(sequence_max_length))

    if take_checkpoint:
      llprint("\nSaving Checkpoint ... "),
      check_ptr = os.path.join(ckpts_dir, 'step_{}.pth'.format(epoch))
      cur_weights = rnn.state_dict()
      T.save(cur_weights, check_ptr)
      llprint("Done!\n")
    
    if summarize:
      random_length = np.random.randint(2, sequence_max_length + 1)
      input_data, target_output = generate_data(batch_size, random_length, input_length, -1, maxnumberofcopies=maxnumberofcopies, currentmaxnocopies=currentmaxnocopies, fillbatch=True)

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
  fig.add_trace(go.Scatter(x=df["epoch"], y=df["loss"], mode='lines', name='loss'))
  fig.add_trace(go.Scatter(x=df["epoch"], y=df["testloss"], mode='lines', name='testloss'))
  fig.update_layout(title='Losses', xaxis_title='Epoch', yaxis_title='Loss')
  fig.show()

  for i in range(10):#range(int((iterations + 1) / 100)):
    llprint("\nIteration %d/%d" % (i, iterations))
    # We test now the learned generalization using sequence_max_length examples
    random_length = np.random.randint(2, sequence_max_length + 1)
    input_data, target_output = generate_data(batch_size, random_length, input_length, -1, maxnumberofcopies=maxnumberofcopies, currentmaxnocopies=maxnumberofcopies)

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

