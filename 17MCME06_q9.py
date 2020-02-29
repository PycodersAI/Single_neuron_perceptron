import numpy as np
import matplotlib.pyplot as plt

# sigmoid function used as activation function for neuron
def sigmoid(x):
  return (1/ (1 + np.exp(-x)))

# function to train the neural network when given input, target and number of epochs
def train(M,S,C,epoch):
  W = [0.1,0.4] # randomly chosen weights
  T = 0.3  # randomly chosen threshold
  lr = 0.01  # learning rate or eeta
  error = []  # an array that stores error value during every epoch
  for j in range(0,epoch):
    err = 0
    for i in range(0,len(M)):
      out = sigmoid((W[0]*M[i]) + (W[1]*S[i]) - T)
      err = err + abs(C[i] - out)
      W[0] = W[0] + (lr*M[i]*(C[i]-out))
      W[1] = W[1] + (lr*S[i]*(C[i]-out))
      T = T - (lr*(C[i]-out))
      print("new weight0 = "+str(W[0]))
      print("new weight1 = "+str(W[1]))
      print("new Threshold = "+str(T))
      print("\n")
    error.append(err)
  return W[0],W[1],T,error

def predict(M,S,C,epoch):
  error = []
  epox = np.arange(0,epoch,1)
  w1,w2,T,error = train(M,S,C,epoch)
  #print("latest w1 = "+str(w1))
  #print("latest w2 = "+str(w2))
  #print("latest T = "+str(T))
  while (True):
    print('Enter 1 for prediction')
    print('Enter 2 to show abs error vs epochs graph')
    print('Enter 3 to quit')
    choice = int(input('Select choice='))
    if (choice == 1):
      m = float(input('Enter mass of the plane:'))
      s = float(input('Enter speed of the plane:'))
      sigma = (m*w1) + (s*w2)
      s1 = sigmoid(sigma - T)
      print("\n")
      # s1 closer to 1 is bomber and closer to 0 is fighter
      if (sigma >= T):
        print("Prediction: Bomber")
        print("Confidence="+str(s1)+"/1.000\n")
      else:
        print("Prediction: Fighter")
        print("Confidence="+str(1-s1)+"/1.000\n") 
    elif (choice == 3):
      print('Bye!!')
      return
  # for plotting the abs(error) vs epochs
    elif (choice == 2):
      plt.title('Absolute Error vs Epochs')
      plt.xlabel('epoch')
      plt.ylabel('error')
      plt.plot(epox,error)
      plt.grid()
      plt.show()
      print("\n")
    else:
      print('select a valid choice\n')
      continue

# input values and target values 
M = [1.0,2.0,0.1,2.0,0.2,3.0,0.1,1.5,0.5,1.6] # mass
S = [0.1,0.2,0.3,0.3,0.4,0.4,0.5,0.5,0.6,0.7] # speed
C = [1,1,0,1,0,1,0,1,0,0]  # target 1 is bomber and 0 is fighter

# No. of epochs 
# Suggestion: 50000 - 100000 epochs ensure better accuracy
print("Suggestion: 50000 - 100000 epochs ensure better accuracy")
epoch = int(input('Enter number of epochs = '))
predict(M,S,C,epoch)

# K Pranav Bharadwaj (17MCME06)
# IM.tech 6th sem
