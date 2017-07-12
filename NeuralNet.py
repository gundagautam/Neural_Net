'''
Created on Feb 28, 2017

@author: sangeeta
'''
from datashape.coretypes import float64

def makeWeightList(hLayer,hList, inputLen):
    weightMatrix = defaultdict(list)
    j = 1
    for i in hList:     
        weightMatrix[j].append(np.random.uniform(-1,1,(i,inputLen)))
        inputLen = i+1
        j+=1
    weightMatrix[j].append(np.random.uniform(-1,1,(1,inputLen)))
    return weightMatrix

def sigmoid(val):
    try:
        val = (1 / (1 + math.exp(-val)))
    except OverflowError:
        if val<0:
            val = 1
        else: 
            val = 0
    return val
  
def forwardPass(weightMatrix,inputLayerVals):
    netMatrix = defaultdict(list)
    for i in weightMatrix.keys():
        netMatrix[i].extend(np.matmul(inputLayerVals,np.transpose(weightMatrix[i][0])))
        netMatrix1= list()
        for element in netMatrix[i]:
            element = sigmoid(element)
            netMatrix1.append(element)
        netMatrix[i]=netMatrix1
        inputLayerVals = [1]+netMatrix[i]
    return(netMatrix)     
        
def calcOutput(netMatrix): 
    return (netMatrix[len(netMatrix.keys())])

def backwardPass(output, target,netMatrix,weightMatrix,inputLayer):
    learningRate = 0.1
    deltaMatrix = defaultdict(list)
    delta = 1
    xij = 1
    i = len(weightMatrix.keys())
      
    while i >= 1:
        if i == len(weightMatrix.keys()):
            deltaMatrix.get(i)[0]=deltaMatrix.get(i)[0][0]
            i-=1        
        else:
            deltaMatrix = backwardPassHiddenUnit(output, target,weightMatrix,netMatrix,deltaMatrix)
            i-=1
    k=len(netMatrix.keys())-1
    while k >0:
        netMatrix[k-1]=[1]+netMatrix[k-1]
        if k==1:
            weightMatrix[k][0] += np.outer(deltaMatrix[k][0],inputLayer)*learningRate
        else:
            weightMatrix[k][0] += np.outer(deltaMatrix[k][0],netMatrix[k-1])*learningRate
        k-=1
    return weightMatrix

def backwardPassOpUnit(output, target,weightMatrix):
    deltaMatrix = defaultdict(list)
    deltaOp = list()
    deltaMatrix[len(weightMatrix.keys())].append(deltaOp)
    return deltaMatrix   

def backwardPassHiddenUnit(output, target,weightMatrix,netMatrix,deltaPrev):
    deltaMatrix = defaultdict(list)
    tot = 0
    i = len(weightMatrix.keys())
    while i>=1:
        temp=np.matmul(np.transpose(deltaPrev.get(i)[0]),weightMatrix.get(i)[0])
        j=len(netMatrix[i-1])-1
        temp=temp[1:]
        while j>=0:
            temp[j]=netMatrix[i-1][j]*(1-netMatrix[i-1][j])*temp[j]
            j-=1
        deltaPrev[i-1].append(temp)
        i-=1       
    return deltaPrev
    
inputFile = sys.argv[1]
trainingPercent = sys.argv[2]
errorTolerance = sys.argv[3]
hiddenLayers = sys.argv[4]
hiddenList =  list()
i = 1
while i <= int(hiddenLayers):
    hiddenList.append(int(sys.argv[4+i]))
    i +=1
 
#print(hiddenList)

inputLayerIn = np.loadtxt(inputFile,dtype=np.float64,delimiter=',')
inputLayerIn = np.around(inputLayerIn, decimals=2)
i = 0
x0List = [1]*len(inputLayerIn)
inputLayerIn = np.column_stack((x0List,inputLayerIn))
n = 0

doneList = random.sample(range(0,len(inputLayerIn)), len(inputLayerIn))

m=0
n = 0
inputLayer= list()
testLayer = list()

for i in doneList:
    if m < int((len(inputLayerIn)*(float(trainingPercent)/100))):
        inputLayer.append(inputLayerIn[i])
    else:
        testLayer.append(inputLayerIn[i])
    m+=1
inputLayer = np.array(inputLayer)
testLayer = np.array(testLayer)

inputLayer1 = inputLayer[:,:len(inputLayer[0])-1]
classLabel = inputLayer[:,len(inputLayer[0])-1:]

testLayer1 = testLayer[:,:len(testLayer[0])-1]
classLabelTest = testLayer[:,len(testLayer[0])-1:]

weightMatrix = defaultdict(list)
weightMatrix = makeWeightList(int(hiddenLayers),hiddenList,len(inputLayer1[0]))
error = 999999
killCount = 2000
while error>float(errorTolerance) and killCount>0:
    error = 0
    k = 0
    for inputLayerVals in inputLayer1:
        netMatrix = defaultdict(list)
        netMatrix = forwardPass(weightMatrix,inputLayerVals)
        outputVal = calcOutput(netMatrix)[0]
        weightMatrix = backwardPass(outputVal,classLabel[k],netMatrix,weightMatrix,inputLayerVals)
        error += pow((classLabel[k][0]-outputVal),2)
        k+=1
    error = error/(2*int((len(inputLayerIn)*(float(trainingPercent)/100))))
    #print(error)
    killCount-=1 

totalError=error    
#print(weightMatrix)
error=0
k = 0
for testLayerVals in testLayer1:
    netMatrix = defaultdict(list)
    netMatrix = forwardPass(weightMatrix,testLayerVals)
    outputVal = calcOutput(netMatrix)[0]
    error += pow((classLabelTest[k][0]-outputVal),2)
    k+=1
error = error/(2*int((len(inputLayerIn)*(float(100-int(trainingPercent))/100))))
#print(error)
testError = error
print()
for i in weightMatrix.keys():
    if i != len(weightMatrix):
        print("Hidden Layer "+str(i))
    else:
        print("Output Layer:")
        #print(weightMatrix[i][0])
    k1=1
    for k in weightMatrix[i][0]:
        print("Neuron"+str(k1)+" weights:  "+'  '.join(str(e) for e in k))
        k1+=1
    print()

print("Total Training Error: "+str(totalError))
print("Total Test Error: "+str(testError))