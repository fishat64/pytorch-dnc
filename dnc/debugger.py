
class CustomDataCollector():
    def __init__(self):
        self.list = []
        self.objLists = { "write_key": [], "write_vector": [], "erase_vector": [], "free_gates": [], "read_keys": [], "read_strengths": [], "write_strength": [], "write_gate": [], "allocation_gate": [], "read_modes": [], }
        self.currentobject = { "i": 0, }


    def save(self, key, value):
        if key in self.currentobject:
            newi = self.currentobject["i"] + 1
            self.list.append(self.currentobject)
            self.currentobject = { "i": newi, }
        self.currentobject[key] = value
        self.objLists[key].append(value)

    def print(self, key, n=3):
        print("Printing", key)
        for i in range(len(self.objLists[key])):
            print(self.maximalIndex(self.objLists[key][i], n))



    def maximalIndex(self, tensor, n, dim=2):
        newtensor = tensor.argsort(dim=dim, descending=True)[:n]
        retDict = {}
        for i in range(n):
            retDict[newtensor[i]] = tensor[newtensor[i]]
        return retDict
