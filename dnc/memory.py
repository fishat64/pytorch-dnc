#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
import numpy as np

from .util import *
from .debugger import CustomDataCollector

# LSTM() returns tuple of (tensor, (recurrent state))
class extract_tensor(nn.Module):
    def forward(self,x):
        # Output shape (batch, features, hidden)
        tensor, _ = x
        # Reshape shape (batch, hidden)
        return tensor


class Memory(nn.Module):

  def __init__(self, input_size, mem_size=512, cell_size=32, read_heads=4, gpu_id=-1, independent_linears=True, address_every_slot=False):
    super(Memory, self).__init__()

    self.cdc = CustomDataCollector()

    self.mem_size = mem_size
    self.cell_size = cell_size
    self.read_heads = read_heads
    self.gpu_id = gpu_id
    self.input_size = input_size
    self.independent_linears = independent_linears

    m = self.mem_size
    w = self.cell_size
    r = self.read_heads

    self.address_every_slot = address_every_slot
    self.readmodesize = m if address_every_slot else 3 # read from every slot or backward, forward, content
    self.freegatessize = m*r if address_every_slot else r

    # exponent = 2

    # self.allocation_weight_transform = nn.Sequential(
    #     nn.Linear(m, m**exponent),
    #     nn.Sigmoid(),
    #     nn.Linear(m**exponent, m**exponent),
    #     nn.Sigmoid(),
    #     nn.Linear(m**exponent, m**exponent),
    #     nn.Sigmoid(),
    #     nn.Linear(m**exponent, m),        
    # )

    if self.independent_linears:
      self.read_keys_transform = nn.Linear(self.input_size, w * r)
      self.read_strengths_transform = nn.Linear(self.input_size, r)
      self.write_key_transform = nn.Linear(self.input_size, w)
      self.write_strength_transform = nn.Linear(self.input_size, 1)
      self.erase_vector_transform = nn.Linear(self.input_size, w)
      self.write_vector_transform = nn.Linear(self.input_size, w)
      self.free_gates_transform = nn.Linear(self.input_size, self.freegatessize)
      self.allocation_gate_transform = nn.Linear(self.input_size, 1)
      self.write_gate_transform = nn.Linear(self.input_size, 1)
      self.read_modes_transform = nn.Linear(self.input_size, r * self.readmodesize)
    else:
      self.interface_size = (w * r) + (3 * w) + (3 * r) + 3 + self.freegatessize + r*self.readmodesize
      self.interface_weights = nn.Linear(self.input_size, self.interface_size)

    self.I = cuda(1 - T.eye(m).unsqueeze(0), gpu_id=self.gpu_id)  # (1 * n * n)

  def generateUsageVector(self, b, m, generateZero=True):
    if generateZero:
      return cuda(T.zeros(b, m), gpu_id=self.gpu_id)
    usage_vector = np.zeros((b, m))
    for bi in range(b):
      usage_vector[bi] = np.arange(m) / m

    #print("usage_vector", usage_vector)
    usage_vector = cuda(T.from_numpy(usage_vector).float(), gpu_id=self.gpu_id)
    return usage_vector
     
  def reset(self, batch_size=1, hidden=None, erase=True):
    m = self.mem_size
    w = self.cell_size
    r = self.read_heads
    b = batch_size

    if hidden is None:
     

      return {
          'memory': cuda(T.zeros(b, m, w).fill_(0), gpu_id=self.gpu_id),
          'link_matrix': cuda(T.zeros(b, 1, m, m), gpu_id=self.gpu_id),
          'precedence': cuda(T.zeros(b, 1, m), gpu_id=self.gpu_id),
          'read_weights': cuda(T.zeros(b, r, m).fill_(0), gpu_id=self.gpu_id),
          'write_weights': cuda(T.zeros(b, 1, m).fill_(0), gpu_id=self.gpu_id),
          #'usage_vector': cuda(T.zeros(b, m), gpu_id=self.gpu_id)
          'usage_vector': self.generateUsageVector(b, m)
      }
    else:
      hidden['memory'] = hidden['memory'].clone()
      hidden['link_matrix'] = hidden['link_matrix'].clone()
      hidden['precedence'] = hidden['precedence'].clone()
      hidden['read_weights'] = hidden['read_weights'].clone()
      hidden['write_weights'] = hidden['write_weights'].clone()
      hidden['usage_vector'] = hidden['usage_vector'].clone()

      if erase:
        hidden['memory'].data.fill_(0)
        hidden['link_matrix'].data.zero_()
        hidden['precedence'].data.zero_()
        hidden['read_weights'].data.fill_(0)
        hidden['write_weights'].data.fill_(0)
        hidden['usage_vector'] = self.generateUsageVector(b, m)
    return hidden

  def get_usage_vector(self, usage, free_gates, read_weights, write_weights):
    if self.address_every_slot:
      write_weights = write_weights.squeeze(1)
      usage = usage * free_gates + (1- usage) * write_weights
      return usage
      

    # TODO: How to modify usage vector st. 0,1,1,1,1 -> 1,0,1,1,1
    # write_weights = write_weights.detach()  # detach from the computation graph
    usage = usage + (1 - usage) * (1 - T.prod(1 - write_weights, 1))
    ψ = T.prod(1 - free_gates.unsqueeze(2) * read_weights, 1)
    return usage * ψ

  def allocate(self, usage, write_gate, allocationweighttransform=False, mytime=None, mylayer=None):
    # ensure values are not too small prior to cumprod.
    

    # if allocationweighttransform:
    #   if mytime is not None and mylayer is not None and False:
    #     print(mytime, mylayer, "\n", self.allocation_weight_transform(usage).unsqueeze(1))
    #   return self.allocation_weight_transform(usage).unsqueeze(1), usage

    usage = δ + (1 - δ) * usage

    batch_size = usage.size(0)
    # free list
    sorted_usage, φ = T.topk(usage, self.mem_size, dim=1, largest=False)

    # cumprod with exclusive=True
    # https://discuss.pytorch.org/t/cumprod-exclusive-true-equivalences/2614/8
    v = var(sorted_usage.data.new(batch_size, 1).fill_(1))
    cat_sorted_usage = T.cat((v, sorted_usage), 1)
    prod_sorted_usage = T.cumprod(cat_sorted_usage, 1)[:, :-1]

    sorted_allocation_weights = (1 - sorted_usage) * prod_sorted_usage.squeeze()

    # construct the reverse sorting index https://stackoverflow.com/questions/2483696/undo-or-reverse-argsort-python
    _, φ_rev = T.topk(φ, k=self.mem_size, dim=1, largest=False)
    allocation_weights = sorted_allocation_weights.gather(1, φ_rev.long())

    return allocation_weights.unsqueeze(1), usage

  def write_weighting(self, memory, write_content_weights, allocation_weights, write_gate, allocation_gate, onlyoneslot=False):
    ag = allocation_gate.unsqueeze(-1)
    wg = write_gate.unsqueeze(-1)
    
    return wg * (ag * allocation_weights + (1 - ag) * write_content_weights)

  def get_link_matrix(self, link_matrix, write_weights, precedence):
    precedence = precedence.unsqueeze(2)
    write_weights_i = write_weights.unsqueeze(3)
    write_weights_j = write_weights.unsqueeze(2)

    prev_scale = 1 - write_weights_i - write_weights_j
    new_link_matrix = write_weights_i * precedence

    link_matrix = prev_scale * link_matrix + new_link_matrix
    # trick to delete diag elems
    return self.I.expand_as(link_matrix) * link_matrix

  def update_precedence(self, precedence, write_weights):
    return (1 - T.sum(write_weights, 2, keepdim=True)) * precedence + write_weights

  def write(self, write_key, write_vector, erase_vector, free_gates, read_strengths, write_strength, write_gate, allocation_gate, hidden, stepByStep=False, returnOther=None, mytime=None, mylayer=None):
    if stepByStep != False:
      stepByStep["currentObj"]["mem_before_w"] = hidden['memory'].detach().numpy()
    
    # get current usage
    hidden['usage_vector'] = self.get_usage_vector(
        hidden['usage_vector'],
        free_gates,
        hidden['read_weights'],
        hidden['write_weights']
    )

    if returnOther is not None:
      if returnOther["usage_vector_bool"]:
        returnOther["usage_vector"].append(hidden['usage_vector'])

    # lookup memory with write_key and write_strength
    write_content_weights = self.content_weightings(hidden['memory'], write_key, write_strength)

    # get memory allocation
    alloc, _ = self.allocate(
        hidden['usage_vector'],
        allocation_gate * write_gate, 
        mytime=mytime, mylayer=mylayer
    )

    # get write weightings
    hidden['write_weights'] = self.write_weighting(
        hidden['memory'],
        write_content_weights,
        alloc,
        write_gate,
        allocation_gate,
    )

    #print("write_content_weights", write_content_weights.shape)
    #print("alloc", alloc.shape)

    if stepByStep != False:
      stepByStep["currentObj"]["write_content_weights"] = write_content_weights.detach().numpy()
      stepByStep["currentObj"]["alloc"] = alloc.detach().numpy()
      stepByStep["currentObj"]["write_weights"] = hidden['write_weights'].detach().numpy()
      stepByStep["currentObj"]["write_vector"] = write_vector.detach().numpy()

    if self.address_every_slot:
      hidden['memory'] = hidden['memory'] * free_gates.unsqueeze(2)
    else:
      weighted_resets = hidden['write_weights'].unsqueeze(3) * erase_vector.unsqueeze(2)
      reset_gate = T.prod(1 - weighted_resets, 1)
      # Update memory
      hidden['memory'] = hidden['memory'] * reset_gate

    if stepByStep != False:
      stepByStep["currentObj"]["mem_after_reset"] = hidden['memory'].detach().numpy()

    hidden['memory'] = hidden['memory'] + \
        T.bmm(hidden['write_weights'].transpose(1, 2), write_vector)

    # update link_matrix
    hidden['link_matrix'] = self.get_link_matrix(
        hidden['link_matrix'],
        hidden['write_weights'],
        hidden['precedence']
    )
    hidden['precedence'] = self.update_precedence(hidden['precedence'], hidden['write_weights'])

    if stepByStep != False:
      stepByStep["currentObj"]["mem_after_w"] = hidden['memory'].detach().numpy()
      stepByStep["currentObj"]["link_matrix"] = hidden['link_matrix'].detach().numpy()
      stepByStep["currentObj"]["precedence"] = hidden['precedence'].detach().numpy()

    return hidden

  def content_weightings(self, memory, keys, strengths):
    d = θ(memory, keys)
    return σ(d * strengths.unsqueeze(2), 2)

  def directional_weightings(self, link_matrix, read_weights):
    rw = read_weights.unsqueeze(1)

    f = T.matmul(link_matrix, rw.transpose(2, 3)).transpose(2, 3)
    b = T.matmul(rw, link_matrix)
    return f.transpose(1, 2), b.transpose(1, 2)

  def read_weightings(self, memory, content_weights, link_matrix, read_modes, read_weights):
    forward_weight, backward_weight = self.directional_weightings(link_matrix, read_weights)

    content_mode = read_modes[:, :, 2].contiguous().unsqueeze(2) * content_weights
    backward_mode = T.sum(read_modes[:, :, 0:1].contiguous().unsqueeze(3) * backward_weight, 2)
    forward_mode = T.sum(read_modes[:, :, 1:2].contiguous().unsqueeze(3) * forward_weight, 2)

    return backward_mode + content_mode + forward_mode

  def read_vectors(self, memory, read_weights):
    return T.bmm(read_weights, memory)

  def read(self, read_keys, read_strengths, read_modes, hidden, stepByStep=False, returnOther=None):
    if self.address_every_slot:
      hidden['read_weights'] = read_modes
    else:
      content_weights = self.content_weightings(hidden['memory'], read_keys, read_strengths)

      hidden['read_weights'] = self.read_weightings(
          hidden['memory'],
          content_weights,
          hidden['link_matrix'],
          read_modes,
          hidden['read_weights']
      )
      if stepByStep != False:
        stepByStep["currentObj"]["content_weights"] = content_weights.detach().numpy()
        #stepByStep["currentObj"]["read_weights"] = hidden['read_weights'].detach().numpy()
    read_vectors = self.read_vectors(hidden['memory'], hidden['read_weights'])

    if stepByStep != False:
      #stepByStep["currentObj"]["content_weights"] = content_weights.detach().numpy()
      stepByStep["currentObj"]["read_vectors"] = read_vectors.detach().numpy()
      stepByStep["currentObj"]["read_weights"] = hidden['read_weights'].detach().numpy()

    return read_vectors, hidden, returnOther

  def forward(self, ξ, hidden, stepByStep=False, returnOther=None, mytime=None, mylayer=None):

    # ξ = ξ.detach()
    m = self.mem_size
    w = self.cell_size
    r = self.read_heads
    b = ξ.size()[0]


    if self.independent_linears:
      # r read keys (b * r * w)
      read_keys = T.tanh(self.read_keys_transform(ξ).view(b, r, w))
      # r read strengths (b * r)
      read_strengths = F.softplus(self.read_strengths_transform(ξ).view(b, r))
      # write key (b * 1 * w)
      write_key = T.tanh(self.write_key_transform(ξ).view(b, 1, w))
      # write strength (b * 1)
      write_strength = F.softplus(self.write_strength_transform(ξ).view(b, 1))
      # erase vector (b * 1 * w)
      erase_vector = T.sigmoid(self.erase_vector_transform(ξ).view(b, 1, w))
      # write vector (b * 1 * w)
      write_vector = T.tanh(self.write_vector_transform(ξ).view(b, 1, w)) #F.softplus
      # r free gates (b * r)
      free_gates = T.sigmoid(self.free_gates_transform(ξ).view(b, self.freegatessize))
      # allocation gate (b * 1)
      allocation_gate = T.sigmoid(self.allocation_gate_transform(ξ).view(b, 1))
      # write gate (b * 1)
      write_gate = T.sigmoid(self.write_gate_transform(ξ).view(b, 1))
      # read modes (b * r * 3)
      read_modes = σ(self.read_modes_transform(ξ).view(b, r, self.readmodesize), -1)
    else:
      ξ = self.interface_weights(ξ)
      # r read keys (b * w * r)
      read_keys = T.tanh(ξ[:, :r * w].contiguous().view(b, r, w))
      # r read strengths (b * r)
      read_strengths = F.softplus(ξ[:, r * w:r * w + r].contiguous().view(b, r))
      # write key (b * w * 1)
      write_key = T.tanh(ξ[:, r * w + r:r * w + r + w].contiguous().view(b, 1, w))
      # write strength (b * 1)
      write_strength = F.softplus(ξ[:, r * w + r + w].contiguous().view(b, 1))
      # erase vector (b * w)
      erase_vector = T.sigmoid(ξ[:, r * w + r + w + 1: r * w + r + 2 * w + 1].contiguous().view(b, 1, w))
      # write vector (b * w)
      write_vector = T.tanh(ξ[:, r * w + r + 2 * w + 1: r * w + r + 3 * w + 1].contiguous().view(b, 1, w)) #F.softplus
      # r free gates (b * r)
      free_gates = T.sigmoid(ξ[:, r * w + r + 3 * w + 1: r * w + self.freegatessize + r + 3 * w + 1].contiguous().view(b, self.freegatessize))
      # allocation gate (b * 1)
      allocation_gate = T.sigmoid(ξ[:, r * w +  self.freegatessize + r + 3 * w + 1].contiguous().unsqueeze(1).view(b, 1))
      # write gate (b * 1)
      write_gate = T.sigmoid(ξ[:, r * w + self.freegatessize + r + 3 * w + 2].contiguous()).unsqueeze(1).view(b, 1)
      # read modes (b * 3*r)
      read_modes = σ(ξ[:, r * w + self.freegatessize + r + 3 * w + 3: r * w + self.freegatessize + (1+self.readmodesize) * r + 3 * w + 3].contiguous().view(b, r, self.readmodesize), -1)

    if stepByStep != False:
      stepByStep["currentObj"]["read_keys"] = read_keys.detach().numpy()
      stepByStep["currentObj"]["read_strengths"] = read_strengths.detach().numpy()
      stepByStep["currentObj"]["write_key"] = write_key.detach().numpy()
      stepByStep["currentObj"]["write_strength"] = write_strength.detach().numpy()
      stepByStep["currentObj"]["erase_vector"] = erase_vector.detach().numpy()
      stepByStep["currentObj"]["write_vector"] = write_vector.detach().numpy()
      stepByStep["currentObj"]["free_gates"] = free_gates.detach().numpy()
      stepByStep["currentObj"]["allocation_gate"] = allocation_gate.detach().numpy()
      stepByStep["currentObj"]["write_gate"] = write_gate.detach().numpy()
      stepByStep["currentObj"]["read_modes"] = read_modes.detach().numpy()


    
    
    if returnOther is not None:
      write_content_weights = σ(θ(hidden['memory'].clone().detach(), write_key) * write_strength.unsqueeze(2), 2)
      alloc, _ = self.allocate(hidden['usage_vector'], allocation_gate * write_gate)
      write_weights = self.write_weighting(hidden['memory'], write_content_weights, alloc, write_gate, allocation_gate, onlyoneslot=True)
      
      if returnOther["allocation_weights_bool"]:
        returnOther["allocation_weights"].append(alloc)
      if returnOther["allocation_gate_bool"]:
        returnOther["allocation_gate"].append(allocation_gate)
      if returnOther["write_gate_bool"]:
        returnOther["write_gate"].append(write_gate)
      if returnOther["read_modes_bool"]:
        returnOther["read_modes"].append(read_modes)


      if returnOther["write_weights_bool"]:
        returnOther["write_weights"].append(write_weights)
      if returnOther["read_weights_bool"]:
        returnOther["read_weights"].append(hidden['read_weights'])
      if returnOther["free_gates_bool"]:
        returnOther["free_gates"].append(free_gates)
      if returnOther["erase_vector_bool"]:
        returnOther["erase_vector"].append(erase_vector)
      if returnOther["write_vector_bool"]:
        returnOther["write_vector"].append(write_vector)
      



    hidden = self.write(write_key, write_vector, erase_vector, free_gates,
                        read_strengths, write_strength, write_gate, allocation_gate, hidden, stepByStep=stepByStep, returnOther=returnOther, mytime=mytime, mylayer=mylayer)
    return self.read(read_keys, read_strengths, read_modes, hidden, stepByStep=stepByStep, returnOther=returnOther)
