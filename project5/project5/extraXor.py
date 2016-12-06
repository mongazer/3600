from Testing import average, stDeviation
import random
from NeuralNet import buildNeuralNet
from NeuralNetUtil import getList
def getXorData(fileString ="xordata.txt", limit=100000 ):
    """
    returns limit # of examples from file passed as string
    """
    examples=[]
    attrValues={}
    data = open(fileString)
    attrs = ['a','b']
    attr_values = [['0','1'],
                 ['0','1']]
    
    attrNNList = [('a', {'0' : getList(1,2), '1' : getList(2,2)}),
                 ('b',{'0' : getList(1,2), '1' : getList(2,2)})]

    classNNList = {'1':[0,1], '0':[1,0]}
    
    for index in range(len(attrs)):
        attrValues[attrs[index]]=attrNNList[index][1]

    lineNum = 0
    for line in data:
        inVec = []
        outVec = []
        count=0
        for val in line.split(','):
            if count==2:
                outVec = classNNList[val[:val.find('\n')]]
            else:
                inVec.append(attrValues[attrs[count]][val])
            count+=1
        examples.append((inVec,outVec))
        lineNum += 1
        if (lineNum >= limit):
            break
    random.shuffle(examples)
    return examples

def buildXorData(size=4):

    Data =  getXorData()
    DataTrainList = []
    for cdRec in Data:
        tmpInVec = []
        for cdInRec in cdRec[0] :
            for val in cdInRec :
                tmpInVec.append(val)
        #print "in :" + str(cdRec) + " in vec : " + str(tmpInVec)
        tmpList = (tmpInVec, cdRec[1])
        DataTrainList.append(tmpList)
    #print "car data list : " + str(carDataList)
   
    DataTestList = DataTrainList
    return DataTrainList, DataTestList

xorData = buildXorData() 
def testXorData(hiddenLayers = [0]):
    return buildNeuralNet(xorData,maxItr = 200, hiddenLayerList =  hiddenLayers)

def test5xor(hiddenLayers = [0]):
    result = []
    print("test xor set for 5 times\n")
    print "num of hidden layer", hiddenLayers
    for i in range(5):
        nnet,acu=testXorData(hiddenLayers)
        result.append(acu)
    print("---------Result------- ")
    print(result)
    print("------ Avg ----------")
    avg = average(result)
    print(avg)
    print("--------- Std ----------")
    print(stDeviation(result))    
    return avg



#test5xor()
def output():
    f = open("xorlayer","w")
    for i in range(90):
        
        acc = test5xor([i])
        f.write(str(acc))
        f.write("\n")
        if (acc == 1.0):
            break
    f.close()

output()
