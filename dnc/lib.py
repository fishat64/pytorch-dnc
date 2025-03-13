import torch as T
import torch.nn.functional as F
import hashlib
import numpy as np


CELoss = T.nn.CrossEntropyLoss()

def criterion(predictions, targets): # cross entropy loss
  return T.mean(
      -1 * F.logsigmoid(predictions) * (targets) - T.log(1 - F.sigmoid(predictions) + 1e-9) * (1 - targets)
  )

# 
def crossentropy(predictions, targets): # cross entropy loss
   return CELoss(predictions, targets)

def L1loss(prediction, target): # L1 loss
  return T.mean(T.abs(prediction - target))

def mse(prediction, target): # mean squared error loss
  return T.mean((prediction - target)**2)

def huber_loss(prediction, target, delta=0.25): # huber loss
  residual = T.abs(prediction - target)
  loss = T.where(residual < delta, 0.5 * residual ** 2, delta * (residual - 0.5 * delta))
  return T.mean(loss)

def exp_loss(prediction, target): # exponential loss
  return T.mean(T.exp(1+T.abs(prediction - target))-T.exp(T.ones_like(prediction)))

def createlossfn(lfn1, lfn2, alpha=0.5): # weighted loss function between two loss functions
  def lossfn(prediction, target):
    return alpha * lfn1(prediction, target) + (1 - alpha) * lfn2(prediction, target)
  return lossfn

ENDSYM = -9
NULLSYM = -1
ONESYM = 1

def calcAccuracy(predictions, targets): 
  assert predictions.shape == targets.shape
  return T.mean((predictions, targets)**2)*predictions.shape[1]*predictions.shape[2]

STEPBYSTEPOBJ = { "stepByStep": True, 
                "CurrI": 0, 
                "time":0, 
                "layer":0, 
                "currentObj": None, 
                "input" : None,
                "output" : None,
                "target": None,
                "loss": None,
                "MEMORYCOLUMNS":None,
                "INPUTSIZE":None,
                "OUTPUTSIZE":None,
                "defObj":{ "i": 0, "time": 0, "layer":0, "inputs":None, "outputs": None, "o": None, "total_o": None, "chx": None, "read_vectors": None, "read_keys":None, "read_strengths": None, "content_weights": None,
      "write_key": None, "write_strength": None, "erase_vector": None, "write_vector": None, "free_gates": None, "allocation_gate": None, "read_modes": None,  "write_gate": None, 
      "read_vectors": None, "mem_after_w": None, "mem_before_w": None, "link_matrix": None, "precedence": None, "read_weights": None, "mem_after_reset":None, "write_weights": None, "write_content_weights": None, "alloc": None} , 
                "objects": [] }

LEARNABLEOBJECTIVES = [
   "general_loss", 
   "allocation_weights", 
   "allocation_gate", 
   "write_gate", 
   "write_weights",
   "usage_vector",
   "read_modes", 
   "read_weights", 
   "free_gates"
  ]

LEARNTHISOBJECTIVES = {
   "general_loss": True,
    "allocation_weights": False,
    "allocation_gate": False,
    "write_gate": False,
    "write_weights": False,
    "read_modes": False,
    "read_weights": False,
    "free_gates": False,
    "usage_vector": False,
    "write_weights": False,
}

class teachinternalobjectives:
  def __init__(self, **kwargs):
    self.bools = list()
    self.objectives = list()
    self.keycombs = list()
    self.objectivesdict = dict()
    self.booldict = dict()
    for k in ["allocation_weights", "allocation_gate", "write_gate", "write_weights", "read_modes", "read_weights", "free_gates", "usage_vector", "write_weights", "write_vector", "erase_vector"]:
      self.objectives.append(k)
      self.objectivesdict[k] = list()
      self.bools.append(k+"_bool")
      self.keycombs.append((k+"_bool", k))
      self.booldict[k+"_bool"] = False

  def resetLists(self):
    for obj in self.objectives:
      self.objectivesdict[obj] = list()

  def stack(self, dim=1):
    for key1, key2 in self.keycombs:
      if self.booldict[key1] and len(self.objectivesdict[key2]) > 0:
            self.objectivesdict[key2] = T.stack(self.objectivesdict[key2], dim=dim)

  def isTensor(self, key):
    return self.booldict[key+"_bool"] and isinstance(self.objectivesdict[key], T.Tensor)

  def enable(self, obj):
    self.booldict[obj+"_bool"] = True
  
  def disable(self, obj):
     self.booldict[obj+"_bool"] = False

  def addTensor(self, key, tensor):
    if self.booldict[key+"_bool"]:
      self.objectivesdict[key].append(tensor)
  
  def getTensorList(self, key):
    return self.objectivesdict[key]



RETURNOTHEROBJ = {"allocation_weights_bool":True, "allocation_weights": [], 
                   "allocation_gate_bool": True, "allocation_gate":[], 
                   "write_gate_bool": True, "write_gate": [],
                   "write_weights_bool": True, "write_weights": [],
                   "read_modes_bool": True, "read_modes": [],
                   "read_weights_bool": True, "read_weights": [],
                   "free_gates_bool": True, "free_gates": [],
                   "erase_vector_bool": True, "erase_vector": [],
                   "write_vector_bool": True, "write_vector": [],
                   "usage_vector_bool": True, "usage_vector": [],
                   "keycombs" : [("allocation_weights_bool", "allocation_weights"),
                                 ("allocation_gate_bool", "allocation_gate"),
                                 ("write_gate_bool", "write_gate"),
                                 ("read_modes_bool", "read_modes"),
                                 ("write_weights_bool", "write_weights"),
                                 ("read_weights_bool", "read_weights"),
                                 ("free_gates_bool", "free_gates"),
                                 ("erase_vector_bool", "erase_vector"),
                                 ("write_vector_bool", "write_vector"),
                                 ("usage_vector_bool", "usage_vector")],
                    "bools": [
                       "allocation_weights_bool", 
                       "allocation_gate_bool", 
                       "write_gate_bool", 
                       "write_weights_bool", 
                       "read_modes_bool", 
                       "read_weights_bool", 
                       "free_gates_bool", 
                       "erase_vector_bool",
                        "write_vector_bool",
                        "usage_vector_bool"
                       ]
                   }

class hashabledict(dict):
    def __hash__(self):
        return hash(tuple(sorted((k,str(v)) for k,v in self.items())))

class InputStorage:
  def __init__(self):
    self.usedinputData = {}
    self.usedInputDataSet = set()
  
  def hash(self, data, length=256): # hash function
    data = T.flatten(T.from_numpy(data))
    m = hashlib.sha256()
    m.update(bytes(str(data), 'utf-8'))
    return m.hexdigest()[:length]

  def saveInput(self, input_data, output=None, withoutIncrement=False, flag=None): # returns occurence of input
    flatinputstr = self.hash(input_data)
    if flatinputstr not in self.usedinputData:
        d = hashabledict()
        d["occ"] = 1
        d["input"] = input_data
        d["target"] = None
        d["err"] = 0.0
        d["flag"] = flag
        d["output"] = output
        self.usedinputData[flatinputstr] = d
        return 0
    elif not withoutIncrement:
        self.usedinputData[flatinputstr]["occ"] = self.usedinputData[flatinputstr]["occ"] +1
    self.usedInputDataSet = self.usedInputDataSet.union({flatinputstr})
    return self.usedinputData[flatinputstr]["occ"]
  
  def isSaved(self, input_data, flag=None): # returns occurence of input
    flatinputstr = self.hash(input_data)
    if flatinputstr not in self.usedInputDataSet:
       return False
    elif self.usedinputData[flatinputstr]["flag"] == flag:
        return True
    return False
  
  def getDataByFlag(self, flag=None):
    ret = []
    for key, value in self.usedinputData.items():
      if value["flag"] == flag:
        ret.append(value)
    return ret
  
  def getHashSet(self, flag=None):
    ret = set()
    for key, value in self.usedinputData.items():
      if value["flag"] == flag:
        ret.add(key)
    return ret

  
  def saveInputWithError(self, input_data, target, error, output=None, flag=None): # returns occurence and error of input
    flatinputstr = self.hash(input_data)
    if flatinputstr not in self.usedinputData:
        d = hashabledict()
        d["occ"] = 1
        d["input"] = input_data
        d["target"] = target
        d["err"] = error
        d["flag"] = flag
        d["output"] = output
        self.usedinputData[flatinputstr] = d
        self.usedInputDataSet = self.usedInputDataSet.union({flatinputstr})
      
        return self.usedinputData[flatinputstr]
    else:
        self.usedinputData[flatinputstr]["occ"] = self.usedinputData[flatinputstr]["occ"] +1
        self.usedinputData[flatinputstr]["err"] = error
    return self.usedinputData[flatinputstr]

  def removeInputWithError(self, input_data): # deletes input
    flatinputstr = self.hash(input_data)
    if flatinputstr in self.usedinputData:
      del self.usedinputData[flatinputstr]
      self.usedInputDataSet = self.usedInputDataSet.difference({flatinputstr})

  def getHighestErrorInputs(self, n=100, flag=None): # returns n highest error inputs
    ret = []
    i=0
    for key, value in sorted(self.usedinputData.items(), key=lambda item: item[1]["err"], reverse=True):
        if i >= n:
          break
        if value["flag"] == flag:
          ret.append(self.usedinputData[key])
          i=i+1
    return ret

def tensor2string(tensor): # converts tensor to string
    with np.printoptions(precision=2, suppress=True, linewidth=150, floatmode='fixed'):
      return str(tensor.detach().cpu().numpy())