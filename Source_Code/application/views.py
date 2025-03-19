from django.shortcuts import render,HttpResponse


from django.shortcuts import render,redirect
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import login ,logout,authenticate
from django.core.files.storage import default_storage
import tempfile
global AERFC,encoder
autoencoder=None
sc=None
# Create your views here.
global X_train_scaled,X_test_scaled
def home(request):
    return render(request,'Home.html')

def register(request):
    if request.method == 'POST':
        First_Name = request.POST['name']
        Email = request.POST['email']
        username = request.POST['username']
        password = request.POST['password']
        confirmation_password = request.POST['cnfm_password']
        select_user=request.POST['role']
        if select_user=='admin':
            admin=True
        else:
            admin=False
        if password == confirmation_password:
            if User.objects.filter(username=username).exists():
                messages.error(request, 'Username already exists, please choose a different one.')
                return redirect('register')
            else:
                if User.objects.filter(email=Email).exists():
                    messages.error(request, 'Email already exists, please choose a different one.')
                    return redirect('register')
                else:
                    user = User.objects.create_user(
                        username=username,
                        password=password,
                        email=Email,
                        first_name=First_Name,
                        is_staff=admin
                    )
                    user.save()
                    return redirect('login')
        else:
            messages.error(request, 'Passwords do not match.')
        return render(request, 'register.html')
    return render(request, 'register.html')

def login_view(request):
    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')
        if User.objects.filter(username=username).exists():
            user=User.objects.get(username=username)
            if user.check_password(password):
                user = authenticate(username=username,password=password)
                if user is not None:
                    login(request,user)
                    messages.success(request,'login successfull')
                    return redirect('/')
                else:
                   messages.error(request,'please check the Password Properly')
                   return redirect('login')
            else:
                messages.error(request,"please check the Password Properly")  
                return redirect('login') 
        else:
            messages.error(request,"username doesn't exist")
            return redirect('login')
    return render(request,'login.html')
# Load and preprocess the dataset
def logout_view(request):
    logout(request)
    return redirect('login')


import seaborn as sns
import os
import cv2
import joblib
import pickle
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from PIL import Image

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout, Flatten
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.models import model_from_json

from skimage.transform import resize
from skimage.io import imread
from skimage import io, transform
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

global X_train,X_test,y_train,y_test
path = r"Dataset"
model_folder = "model"
categories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    

X_file = os.path.join(model_folder, "X.txt.npy")
Y_file = os.path.join(model_folder, "Y.txt.npy")
if os.path.exists(X_file) and os.path.exists(Y_file):
    X = np.load(X_file)
    Y = np.load(Y_file)
    print("X and Y arrays loaded successfully.")
else:
    X = [] # input array
    Y = [] # output array
    for root, dirs, directory in os.walk(path):
        for j in range(len(directory)):
            name = os.path.basename(root)
            print(f'Loading category: {dirs}')
            print(name+" "+root+"/"+directory[j])
            if 'Thumbs.db' not in directory[j]:
                img_array = cv2.imread(root+"/"+directory[j])
                img_resized = resize(img_array, (64, 64, 3))
                # Append the input image array to X
                X.append(img_resized.flatten())
                    # Append the index of the category in categories list to Y
                Y.append(categories.index(name))
X = np.array(X)
Y = np.array(Y)
np.save(X_file, X)
np.save(Y_file, Y)
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.20,random_state=42)
labels=['Fraud','NotFraud']
precision = []
recall = []
fscore = []
accuracy = []
def calculateMetrics(image,algorithm, predict, y_test):
    global categories
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    conf_matrix = confusion_matrix(y_test, predict)
    total = sum(sum(conf_matrix))
    se = conf_matrix[0,0]/(conf_matrix[0,0]+conf_matrix[0,1])
    se = se* 100
    sp = conf_matrix[1,1]/(conf_matrix[1,0]+conf_matrix[1,1])
    sp = sp* 100
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    accuracy.append(a)
    CR = classification_report(y_test, predict,target_names=categories)
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = categories, yticklabels = categories, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(categories)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class')       
    plt.legend()
    plt.savefig(image)
import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import json
import matplotlib.pyplot as plt
def extratree():
    '''
    global x_train, x_test, y_train, y_test, X, Y

    # Flatten the data for the Extra Trees Classifier
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)

    # Load or train the Extra Trees Classifier model
    model_filename = os.path.join(model_folder, "ETC_model.pkl")
    if os.path.exists(model_filename):
        mlmodel = joblib.load(model_filename)
    else:
        mlmodel = ExtraTreesClassifier(n_estimators=10, max_depth=3)
        mlmodel.fit(x_train_flat, y_train)
        joblib.dump(mlmodel, model_filename)
        print(f'Model saved to {model_filename}')
    y_pred1 = mlmodel.predict(x_test_flat)

    # Performance metrics for Extra Trees Classifier
    image1 = 'static/images/etc.jpg'
    calculateMetrics(image1, 'Existing Extra Trees Classifier', y_pred1, y_test)
    '''
def cnnModel():
    global X, Y, x_train, x_test, y_train, y_test, model_folder, categories, model, history

    # Shuffle and split the data
    indices_file = os.path.join(model_folder, "shuffled_indices.npy")
    if os.path.exists(indices_file):
        indices = np.load(indices_file)
        X = X[indices]
        Y = Y[indices]
    else:
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        np.save(indices_file, indices)
        X = X[indices]
        Y = Y[indices]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

    # Reshape and preprocess the data for CNN
    x_train = x_train.reshape((-1, 64, 64, 3))  # Assuming 64x64 RGB images
    x_test = x_test.reshape((-1, 64, 64, 3))
    y_train = to_categorical(y_train, num_classes=len(categories))
    y_test = to_categorical(y_test, num_classes=len(categories))

    # File paths
    Model_file = os.path.join(model_folder, "DLmodel.json")
    Model_weights = os.path.join(model_folder, "DLmodel_weights.h5")
    Model_history = os.path.join(model_folder, "history.pckl")
    num_classes = len(categories)

    # Load or train the CNN model
    if os.path.exists(Model_file):
        with open(Model_file, "r") as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
        model.load_weights(Model_weights)
        print(model.summary())
        with open(Model_history, 'rb') as f:
            history = pickle.load(f)
            acc = history['accuracy']
    else:
        # Define the CNN model
        model = Sequential()
        model.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(units=256, activation='relu'))
        model.add(Dense(units=num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())

        # Train the model
        hist = model.fit(x_train, y_train, batch_size=16, epochs=25, validation_data=(x_test, y_test), shuffle=True, verbose=2)

        # Save the model and training history
        model.save_weights(Model_weights)
        with open(Model_file, "w") as json_file:
            json_file.write(model.to_json())
        with open(Model_history, 'wb') as f:
            pickle.dump(hist.history, f)
        acc = hist.history['accuracy']

    # Evaluate the CNN model
    Y_pred = model.predict(x_test)
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    # Calculate performance metrics
    image = 'static/images/cnn.png'
    calculateMetrics(image, "custom CNN", Y_pred_classes, y_test_classes)

    # Retrieve the last two metrics from the list
def modelevaluate(request):
    if len(accuracy)==0:
        '''
        extratree()
        '''
        cnnModel()
    return render(request, 'modelevaluation.html', {
        'algorithm': 'Custom CNN',
        'image': 'static/images/cnn.png',
        'accuracy': accuracy[0],
        'precision':precision[0],
        'recall': recall[0],
        'fscore': fscore[0],
    })
    
'''
        
        'algorithm1': 'Existing Extra Trees',
        'image1': 'static/images/etc.jpg',
        'accuracy1': metrics['accuracy'][0],
        'precision1': metrics['precision'][0],
        'recall1': metrics['recall'][0],
        'fscore1': metrics['fscore'][0]
'''
def prediction_view(request):
    import joblib
    Model_file = os.path.join(model_folder, "DLmodel.json")
    Model_weights = os.path.join(model_folder, "DLmodel_weights.h5")
    Model_history = os.path.join(model_folder, "history.pckl")
    num_classes = len(categories)
    with open(Model_file, "r") as json_file:
        loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
        model.load_weights(Model_weights)
        print(model.summary())
    with open(Model_history, 'rb') as f:
        history = pickle.load(f)
        acc = history['accuracy']
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        filename = default_storage.save(uploaded_file.name, uploaded_file)
        img = cv2.imread(filename)
        img = cv2.resize(img, (64,64))
        im2arr = np.array(img)
        im2arr = im2arr.reshape(-1,64,64,3)
        test = np.asarray(im2arr)
        test = test.astype('float32')
        test = test/255
        X_test_features = model.predict(test)
        predict = np.argmax(X_test_features)
        img = cv2.imread(filename)
        img = cv2.resize(img, (500,500))
        cv2.putText(img, 'Classified as : '+categories[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 255), 2)
        cv2.imshow('Classified as : '+categories[predict], img)
        cv2.waitKey(0)
        default_storage.delete(filename)
        msg=categories[predict]
        messages.success(request,msg)
        return redirect('prediction')
    return render(request,'predict.html')
def about(request):
    return render(request,'about.html')
def chart(request):
    return render(request,'chart.html')
