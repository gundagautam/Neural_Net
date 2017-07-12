'''
Created on Feb 27, 2017

@author: sangeeta
'''

inputFile = sys.argv[1]
outputFile = sys.argv[2]

# the open keyword opens a file in read-only mode by default
from numpy import dtype, float_, float16
from sqlalchemy.sql.expression import false
f = open(inputFile)
# read all the lines in the file and return them in a list
lines = f.readlines()
f.close()
mainList = list()
mainList1 = list()
for line in lines:
    if(line.find('?')!=-1):
        continue
    if (line[0:1]==' '):
        line = line[1:]
    lineList = re.split(r'[ ,|;"]+',line)
    if ('' in lineList):
        continue
    mainList.append(lineList)
#print(mainList)  
mainList1 = list()
listKeyVal = defaultdict(list)
listVals = list()
length = 0
tempList=list()
for list1 in mainList:
    i = 0
    while i<len(list1):
        try:
            list1[i] = float(list1[i])
        except ValueError:
            tempList = listKeyVal[i]
            if list1[i] not in tempList:
                if(i == len(list1)-1):
                    length+=1
                list1[i] = len(listKeyVal[i])-1
            else:
                list1[i] = listKeyVal[i].index(list1[i])
        i = i+1

if(length==0):
    length=(np.amax(mainList, axis=0)[len(mainList[0])-1])-(np.amin(mainList, axis=0)[len(mainList[0])-1])

#print(listKeyVal)
listOfIndexes = list(range(0, len(mainList[0])))
listOfStrIndexes = listKeyVal.keys()
#print(len(listKeyVal.keys()))

finalPreList = [x for x in listOfIndexes if x not in listOfStrIndexes] 
mainList = np.array(mainList)
#print(len(mainList[0]))
mainList1 = mainList[:,finalPreList]
mainList2 = mainList[:,len(mainList[0])-1:]
print(listOfStrIndexes)
mainList2 = mainList2/length

print(mainList2)
mainList1 = (mainList1 - np.mean(mainList1, axis=0))/np.std(mainList1, axis=0)
mainList1 = np.around(mainList1, decimals=2)
#print(mainList1.shape)
#print(mainList[:,[1]].shape)
for i in listOfStrIndexes:
    mainList1 = np.hstack((mainList1[:,:i], mainList[:,[i]], mainList1[:,i:]))
mainList1 = np.hstack((mainList1[:,:len(mainList[0])-1],mainList2))
print (mainList1)

mainList1=mainList1.tolist()
with open(outputFile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(mainList1)







