from pandas import read_csv
from pandas import DataFrame
from nltk import word_tokenize, pos_tag
from tensorflow import nn
from tensorflow import keras
import numpy as np

TAGS = ["$", "''", "(", ")", ",", "--", ".", ":", "CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNP", "NNPS", "NNS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB", "``", "#"]

SIZE = 300

def findIndex(pos, offset):
    index = 0
    counter = 0
    while(counter < offset):
        counter += len(pos[index][0])
        if(pos[index][0][-1] not in ['.', ',', '?', '!', '(', ')', '-']):
            counter += 1
        index += 1
    return index
    
def getLongest(poss):
    return max([len(a) for a in poss])
    
def padPoss(file, poss, maxlen):
    for i in range(len(poss)):
        #x = findIndex(poss[i], file['Pronoun-offset'][i])
        #poss[i] = poss[i][:x+1]
        poss[i] += [('.','.')] * (maxlen - len(poss[i]))
        
def getTags(pos):
    tags = []
    for pair in pos:
        #tags.append(TAGS.index(pair[1]))
        if(pair[1] == "NNP" or pair[1] == "NNPS"):
            tags.append(1)
        elif(pair[1] == "PRP" or pair[1] == "PRP$"):
            tags.append(2)
        else:
            tags.append(0)
    return tags
        
def createInput(file, poss):
    input = []
    for i in range(file.shape[0]):
        entry = []
        #entry.append(findIndex(poss[i], file['A-offset'][i]))
        #entry.append(findIndex(poss[i], file['B-offset'][i]))
        entry += getTags(poss[i])
        input.append(entry)
    return np.array(input)
    
def createOutput(file):
    output = []
    for i in range(file.shape[0]):
        a = file['A-coref'][i]
        b = file['B-coref'][i]
        if(a == True and b == False):
            output.append(0)
        elif(a == False and b == True):
            output.append(1)
        else:
            output.append(2)
    return np.array(output)
    
def createDataSet(path):
    print("Reading file")
    file = read_csv(path, sep='\t')

    poss = []
    print("Creating tagging")
    for i in range(file.shape[0]):
        poss.append(pos_tag(word_tokenize(file['Text'][i])))
        
    maxlen = getLongest(poss)
    print("Padding")
    padPoss(file, poss, SIZE)

    print("Generating input")
    input = createInput(file, poss)
    print("Generating output")
    output = createOutput(file)

    return input, output
    
def createPredictionInput(path):
    file = read_csv(path, sep='\t')
    
    poss = []
    for i in range(file.shape[0]):
        poss.append(pos_tag(word_tokenize(file['Text'][i])))
        
    maxlen = getLongest(poss)
    padPoss(file, poss, SIZE)
    
    input = createInput(file, poss)
    
    return input
    
def normalizePredictions(predictions):
    sums = np.sum(predictions, 1)
    
    new_predictions = []
    for i in range(len(predictions)):
        new_predictions.append(predictions[i]/sums[i])
        
    return new_predictions
    
def saveCSV(path, predictions, savePath):
    file = read_csv(path, sep='\t')

    predictions = np.array(predictions)
    
    id = file["ID"]
    a = predictions[:,0]
    b = predictions[:,1]
    neither = predictions[:,2]
    
    data = {
    'ID': id,
    'A': a,
    'B': b,
    'NEITHER': neither
    }
    
    df = DataFrame(data, columns=['ID', 'A', 'B', 'NEITHER'])
    df.to_csv(savePath, index=None, header=True)


input, output = createDataSet('gap-test.tsv')    

print("Creating model")
model = keras.Sequential()
model.add(keras.layers.Dense(32, activation=nn.relu, input_shape=(SIZE,)))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(3, activation=nn.sigmoid))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
              
print("Training model")
model.fit(input, output, epochs=25)

print("Loading test dataset")
inputTest = createPredictionInput('test_stage_1.tsv')

print("Predicting")
predictions = model.predict(inputTest)
print("Normalizing")
predictions = normalizePredictions(predictions)

print("Saving")
saveCSV('test_stage_1.tsv', predictions, 'submission.csv')

print("Finished!")

