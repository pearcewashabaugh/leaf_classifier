import numpy, pandas, os, skimage
import matplotlib.pyplot as plt
from skimage import io
from skimage import data
from sklearn import svm
from skimage.transform import resize
from IPython.display import display, HTML

train = pandas.read_csv('Kaggle_Data/train.csv', header=0).sort_values(['species','id']) #Load the training data
test = pandas.read_csv('Kaggle_Data/test.csv', header=0) #Load the testing data

numTrain = len(train['id'])
numTest = len(test['id'])
numPics = numTrain + numTest
pic_id = numpy.empty(numPics, dtype=object)   #Empty array of length numPics
pic_id[:numTrain] = train['id'] #Copy Train's ids to the front
pic_id[numTrain:] = test['id'] #Copy Test's ids to the end


#NOTE: The images have Average Width = 693.3 and Average Height = 493.1

pics = numpy.array([ numpy.array(io.imread('Kaggle_Data/images/%d.jpg' % (pic_id[i]))) for i in range(numPics) ]) #Load each image into a 2D array

pics_resized = numpy.array([ numpy.array(skimage.transform.resize(pics[i],(100,100))) for i in range(numPics) ])  #Resize each image

pics_resized_flat = numpy.array([ pics_resized[i].flatten() for i in range(numPics) ])  #Flatten each image into a 1D array

#NOTE: From here on out, it's best to pretend that the ids of the training (resp. testing) images had been 0, ..., numTrain - 1 (resp. numTrain, ..., numPics - 1) all along.

clf = svm.SVC(gamma = 0.001, C = 100.0)
clf.fit( pics_resized_flat[:numTrain], range(numTrain) )

species = numpy.empty(numPics, dtype=object) #Create an empty array of length numPics
species[:numTrain] = train['species'] #Copy Train's species
closest_train_ind = clf.predict(pics_resized_flat[numTrain:]) #For each test leaf, get the index of training leaf that is closest to this test leaf.
for i in range(numTrain,numPics):
    species[i] = species[closest_train_ind[i-numTrain]]




#Given myId, we now display all images of myId's species
myId = 0
myIdSpecies = species[myId]
display("The leaf indexed by %d, pictured below, is of the species %s." % (myId, myIdSpecies))
plt.imshow(pics[myId])
plt.show()

display('Below are all the TEST leaves of the species ' + myIdSpecies + ':')
for i in range(numPics):
    if i != myId and species[i] == myIdSpecies and i >= numTrain:
        display("index i = %d,  picture id = %d" % (i, pic_id[i]))
        plt.imshow(pics[i])
        plt.show()

