# import paket yang diperlukan
import numpy, os
import cv2
import matplotlib.pyplot as plt

from io import BytesIO
from PIL import Image
from time import time
from IPython.display import display
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC

#Path to the root image directory containing sub-directories of images
path="dataset_svm/"

data_slice = [70,195,78,172] # [ ymin, ymax, xmin, xmax]
# to extract the ‘interesting’ part of the image files 
# and avoid use statistical correlation from the background 

# resize ratio to reduce sample dimention
resize_ratio = 2.5

h = int((data_slice[1] - data_slice[0])/resize_ratio) #ymax - ymin slice, Height of image in float
w = int((data_slice[3] - data_slice[2])/resize_ratio) #xmax - xmin slice, Width of image in float 
# print("Image dimension after resize (h,w) :", h, w)

n_sample = 0 #Initial sample count
label_count = 0 #Initial label count
n_classes = 0 #Initial class count

#PCA Component 
n_components = 8

#Flat image Feature Vector
X=[]
#Int array of Label Vector
Y=[]

target_names = [] #Array to store the names of the persons

for directory in os.listdir(path):
    for file in os.listdir(path+directory):
        img=cv2.imread(path+directory+"/"+file)[data_slice[0]:data_slice[1], data_slice[2]:data_slice[3]]
        img=cv2.resize(img, (w,h))
        img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        featurevector=numpy.array(img).flatten()
        X.append(featurevector)
        Y.append(label_count)
        n_sample = n_sample + 1
    target_names.append(directory)
    label_count=label_count+1

# print("Samples :", n_sample)
# print("Class :", target_names)
n_classes = len(target_names)

###############################################################################
# Split into a training set and a test set using a stratified k fold

# split into a training and teststing set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

###############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction

# print("\nExtracting the top %d eigenfaces from %d faces" % (n_components, len(X_train)))

t0 = time()
pca = PCA(n_components=n_components, whiten=True).fit(X_train)

# print("done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, h, w))

# print("\n")
# print("Projecting the input data on the eigenfaces orthonormal basis")

t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# print("done in %0.3fs" % (time() - t0))
# print("\n")

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_components):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i], cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
        
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)



def svm_model(personName):

    #Path to the root image directory containing sub-directories of images
    path="dataset_svm/"

    data_slice = [70,195,78,172] # [ ymin, ymax, xmin, xmax]
    # to extract the ‘interesting’ part of the image files 
    # and avoid use statistical correlation from the background 

    # resize ratio to reduce sample dimention
    resize_ratio = 2.5

    h = int((data_slice[1] - data_slice[0])/resize_ratio) #ymax - ymin slice, Height of image in float
    w = int((data_slice[3] - data_slice[2])/resize_ratio) #xmax - xmin slice, Width of image in float 
    # print("Image dimension after resize (h,w) :", h, w)

    n_sample = 0 #Initial sample count
    label_count = 0 #Initial label count
    n_classes = 0 #Initial class count

    #PCA Component 
    n_components = 8

    #Flat image Feature Vector
    X=[]
    #Int array of Label Vector
    Y=[]

    target_names = [] #Array to store the names of the persons

    for directory in os.listdir(path):
        for file in os.listdir(path+directory):
            img=cv2.imread(path+directory+"/"+file)[data_slice[0]:data_slice[1], data_slice[2]:data_slice[3]]
            img=cv2.resize(img, (w,h))
            img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            featurevector=numpy.array(img).flatten()
            X.append(featurevector)
            Y.append(label_count)
            n_sample = n_sample + 1
        target_names.append(directory)
        label_count=label_count+1

    # print("Samples :", n_sample)
    # print("Class :", target_names)
    n_classes = len(target_names)

    ###############################################################################
    # Split into a training set and a test set using a stratified k fold

    # split into a training and teststing set
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

    ###############################################################################
    # Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
    # dataset): unsupervised feature extraction / dimensionality reduction

    # print("\nExtracting the top %d eigenfaces from %d faces" % (n_components, len(X_train)))

    t0 = time()
    pca = PCA(n_components=n_components, whiten=True).fit(X_train)

    # print("done in %0.3fs" % (time() - t0))

    eigenfaces = pca.components_.reshape((n_components, h, w))

    # print("\n")
    # print("Projecting the input data on the eigenfaces orthonormal basis")

    t0 = time()
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    ###############################################################################
    # Train a SVM classification model
    
    # print("\n")
    # print("Fitting the classifier to the training set")
    t0 = time()
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }

    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid, verbose=8)
    clf = clf.fit(X_train_pca, y_train)

    t0 = time()
    y_pred = clf.predict(X_test_pca)

    # calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("\nModel accuracy is: ", accuracy)

    print("\nClassification Report: ")
    print(classification_report(y_test, y_pred, target_names=target_names))

    print("\nConfusion Matrix : ")
    print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
    
    ###############################################################################
    # Prediction of user based on the model

    test = []
    #testImage = "test/Tiger_Woods_0010.jpg"
    testImage = "images/" + personName

    testImage=cv2.imread(testImage)[data_slice[0]:data_slice[1], data_slice[2]:data_slice[3]]
    testImage=cv2.resize(testImage, (w,h))
    testImage=cv2.cvtColor(testImage, cv2.COLOR_BGR2GRAY)
    testImageFeatureVector=numpy.array(testImage).flatten()
    test.append(testImageFeatureVector)
    testImagePCA = pca.transform(test)
    testImagePredict=clf.predict(testImagePCA)
    
    print('\n----------------------------')
    
    result = {
        "state": True,
        "message": "Hasil klasifikasi SVM",
        "data": [
            {
                "name": target_names[testImagePredict[0]],
                "accuracy": str(clf.score(X_test_pca,y_test)),
            }
        ]
    }
    
    print(f'Prediksi Nama yaitu: {target_names[testImagePredict[0]]}')
    print('----------------------------\n')
    
    return result

def is_jpg(filename):
    try:
        img = Image.open(filename)
        print(img.format)
        
        return img.format == 'JPG'
    except IOError:
        print("Not JPG\n")
        
        return False