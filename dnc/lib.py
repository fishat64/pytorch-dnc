import torch as T
import torch.nn.functional as F

def criterion(predictions, targets): # cross entropy loss
  return T.mean(
      -1 * F.logsigmoid(predictions) * (targets) - T.log(1 - F.sigmoid(predictions) + 1e-9) * (1 - targets)
  )
def mse(prediction, target): # mean squared error loss
  return T.mean((prediction - target)**2)

def huber_loss(prediction, target, delta=0.25): # huber loss
  residual = T.abs(prediction - target)
  loss = T.where(residual < delta, 0.5 * residual ** 2, delta * (residual - 0.5 * delta))
  return T.mean(loss)

def exp_loss(prediction, target): # exponential loss
  return T.mean(T.exp(T.abs(prediction - target)))-1

def createlossfn(lfn1, lfn2, alpha=0.5): # weighted loss function between two loss functions
  def lossfn(prediction, target):
    return alpha * lfn1(prediction, target) + (1 - alpha) * lfn2(prediction, target)
  return lossfn



class hashabledict(dict):
    def __hash__(self):
        return hash(tuple(sorted((k,str(v)) for k,v in self.items())))

class InputStorage:
  def __init__(self):
    self.usedinputData = {}

  def saveInput(self, input_data): # returns occurence of input
    flatinput = T.flatten(T.from_numpy(input_data))
    flatinputstr = ""
    for i in range(flatinput.shape[0]):
        flatinputstr += str(flatinput[i].item())
    if flatinputstr not in self.usedinputData:
        d = hashabledict()
        d["occ"] = 1
        d["input"] = input_data
        d["target"] = None
        d["err"] = 0.0
        self.usedinputData[flatinputstr] = d
        return 0
    else:
        self.usedinputData[flatinputstr]["occ"] = self.usedinputData[flatinputstr]["occ"] +1
    return self.usedinputData[flatinputstr]["occ"]
  
  def saveInputWithError(self, input_data, target, error): # returns occurence and error of input
    flatinput = T.flatten(T.from_numpy(input_data))
    flatinputstr = ""
    for i in range(flatinput.shape[0]):
        flatinputstr += str(flatinput[i].item())
        flatinputstr += ","
    if flatinputstr not in self.usedinputData:
        d = hashabledict()
        d["occ"] = 1
        d["input"] = input_data
        d["target"] = target
        d["err"] = error
        self.usedinputData[flatinputstr] = d
        return self.usedinputData[flatinputstr]
    else:
        self.usedinputData[flatinputstr]["occ"] = self.usedinputData[flatinputstr]["occ"] +1
        self.usedinputData[flatinputstr]["err"] = error
    return self.usedinputData[flatinputstr]

  def removeInputWithError(self, input_data): # deletes input
    flatinput = T.flatten(T.from_numpy(input_data))
    flatinputstr = ""
    for i in range(flatinput.shape[0]):
      flatinputstr += str(flatinput[i].item())
      flatinputstr += ","
    if flatinputstr in self.usedinputData:
      del self.usedinputData[flatinputstr]

  def getHighestErrorInputs(self, n=100): # returns n highest error inputs
    ret = []
    i=0
    for key, value in sorted(self.usedinputData.items(), key=lambda item: item[1]["err"], reverse=True):
        if i >= n:
          break
        ret.append(self.usedinputData[key])
        i=i+1
    return ret
