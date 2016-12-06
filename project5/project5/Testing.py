from NeuralNetUtil import buildExamplesFromCarData,buildExamplesFromPenData,buildExamplesFromExtraData
from NeuralNet import buildNeuralNet
import cPickle 
from math import pow, sqrt

def average(argList):
    return sum(argList)/float(len(argList))

def stDeviation(argList):
    mean = average(argList)
    diffSq = [pow((val-mean),2) for val in argList]
    return sqrt(sum(diffSq)/len(argList))

penData = buildExamplesFromPenData() 
def testPenData(hiddenLayers = [24]):
    return buildNeuralNet(penData,maxItr = 200, hiddenLayerList =  hiddenLayers)

carData = buildExamplesFromCarData()
def testCarData(hiddenLayers = [16]):
    return buildNeuralNet(carData,maxItr = 200,hiddenLayerList =  hiddenLayers)


extraData = buildExamplesFromExtraData()
def testExtraData(hiddenLayers = [10]):
    return buildNeuralNet(extraData,maxItr = 200,hiddenLayerList =  hiddenLayers)


def test5extra(hiddenLayers = [10]):
	result = []
	print("test extra set for 5 times")
	print "hidden layer", hiddenLayers
	for i in range(5):
		nnet,acu=testExtraData()
		result.append(acu)
	print("---------Result---------")
	print(result)
	print("--------- Max ----------")
	print(max(result))
	print("----------Avg ----------")
	print(average(result))
	print("--------- Std ----------")
	print(stDeviation(result))



def test5pen(hiddenLayers = [24]):
	result = []
	print("test pen set for 5 times/n")
	print "hidden layer", hiddenLayers
	for i in range(5):
		nnet,acu=testPenData()
		result.append(acu)
	print("---------Result---------")
	print(result)
	print("--------- Max ----------")
	print(max(result))
	print("----------Avg ----------")
	print(average(result))
	print("--------- Std ----------")
	print(stDeviation(result))



def test5car(hiddenLayers = [16]):
	result = []
	print("test car set for 5 times")
	print "hidden layer", hiddenLayers
	for i in range(5):
		nnet,acu=testCarData()
		result.append(acu)
	print("---------Result---------")
	print(result)
	print("--------- Max ----------")
	print(max(result))
	print("----------Avg ----------")
	print(average(result))
	print("--------- Std ----------")
	print(stDeviation(result))


test5pen()
test5car()

def testpenlayer():
	avglist=[]
	stdlist=[]
	f = open("penlayer","w")
	f1 = open("penavg","w")
	f2 = open("penstd", "w")
	f3 = open("penmax", "w")
	for i in range(0,45,5):
		aculist=[]
		f.write("layer = "+ str(i)+ "\n")
		for j in range(5):
			nnet, acu = buildNeuralNet(penData,maxItr = 200,hiddenLayerList =  [i])
			aculist.append(acu)
		curavg=average(aculist)
		curstd=stDeviation(aculist)
		curmax=max(aculist)
		f.write("accuracy list: ")
		for item in aculist:
			f.write(str(item)+" ")
		print("---------Result------- ")
		print(aculist)
		f.write("\n Avg: "+ str(curavg))
		f1.write(str(curavg))
		f1.write("\n")
		print("------ Avg ----------")
		print(curavg)
		f.write("\n Std: "+ str(curstd))
		print("--------- Std ----------")
		print(curstd)
		f2.write(str(curstd))
		f2.write("\n")
		f.write("\n Max: "+ str(curmax))
		print("--------- Max ----------")
		print(curmax)
		f3.write(str(curmax))
		f3.write("\n")	

		avglist.append(curavg)
		stdlist.append(curstd)
	print("++++++++++++++++++++++++++++++++++")
	print "Overall result"
	print(avglist)
	print(stdlist)
	f.write("\n ++++++++++++++++++++++++++++++ \n")
	f.write("avg result collection: ")
	for item in avglist:
			f.write(str(item)+" ")
	f.write("\n std result collection: ")
	for item in stdlist:
			f.write(str(item)+" ")
	f.write("\n\n\n\n\n Done")
	f.close()
	f1.close()
	f2.close()
	f3.close()

def testcarlayer():
	avglist=[]
	stdlist=[]
	f = open("carlayer","w")
	f1 = open("caravg","w")
	f2 = open("carstd", "w")
	f3 = open("carmax", "w")
	for i in range(0,45,5):
		aculist=[]
		f.write("layer = "+ str(i)+ "\n")
		for j in range(5):
			nnet, acu = buildNeuralNet(carData,maxItr = 200,hiddenLayerList =  [i])
			aculist.append(acu)
		curavg=average(aculist)
		curstd=stDeviation(aculist)
		curmax=max(aculist)
		f.write("accuracy list: ")
		for item in aculist:
			f.write(str(item)+" ")
		print("---------Result------- ")
		print(aculist)
		f.write("\n Avg: "+ str(curavg))
		f1.write(str(curavg))
		f1.write("\n")
		print("------ Avg ----------")
		print(curavg)
		f.write("\n Std: "+ str(curstd))
		print("--------- Std ----------")
		print(curstd)
		f2.write(str(curstd))
		f2.write("\n")
		f.write("\n Max: "+ str(curmax))
		print("--------- Max ----------")
		print(curmax)
		f3.write(str(curmax))
		f3.write("\n")	

		avglist.append(curavg)
		stdlist.append(curstd)
	print("++++++++++++++++++++++++++++++++++")
	print "Overall result"
	print(avglist)
	print(stdlist)
	f.write("\n ++++++++++++++++++++++++++++++ \n")
	f.write("avg result collection: ")
	for item in avglist:
			f.write(str(item)+" ")
	f.write("\n std result collection: ")
	for item in stdlist:
			f.write(str(item)+" ")
	f.write("\n\n\n\n\n Done")
	f.close()
	f1.close()
	f2.close()
	f3.close()


# testpenlayer()
# testcarlayer()

# test5extra()