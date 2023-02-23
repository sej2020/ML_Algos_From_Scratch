import pandas as pd
import numpy as np

#---------------------------------------------------------------------------------------
# MLP
#---------------------------------------------------------------------------------------

class MLP:

    def __init__(self, n_hidden, n_input_feat, n_classes, learning_rate, epochs):
        self.n_hidden = n_hidden #4
        self.n_input = n_input_feat #13
        self.n_output = n_classes #3
        self.lr = learning_rate #0.01
        self.epochs = epochs #40

        self.v = np.random.uniform(size=((self.n_input+1)*self.n_hidden,1))
        self.w = np.random.uniform(size=((self.n_hidden+1)*self.n_output,1))


    def hidden_layer(self, x, v):
        a = np.zeros(shape=(self.n_hidden,1))
        for node in range(self.n_hidden): #0,1,2,3
            a_x = np.zeros(shape=(self.n_input+1,1)) 
            for index in range(len(v)): #0,1,2,3,...,54,55
                if index % self.n_hidden == node:
                    a_x[index//self.n_hidden] = v[index]
            a[node] = np.dot(a_x.T,x)
        return a
        
    def sigmoid(self, vector):
        for entry in range(len(vector)):
            vector[entry] = (lambda z: 1/(1 + np.exp(-z)))(vector[entry])
        return vector
    
    def output_layer(self, a, w):
        y = np.zeros(shape=(self.n_output,1))
        for node in range(self.n_output): #0,1,2
            y_a = np.zeros(shape=(self.n_hidden+1,1)) 
            for index in range(len(w)): #0,1,2,3,...,14
                if index % self.n_output == node:
                    y_a[index//self.n_output] = w[index]
            y[node] = np.dot(y_a.T,a)
        return y
        
    def softmax(self, vector):
        e_vector = np.exp(vector)
        sum = np.sum(e_vector)
        return e_vector*(1/sum)
        
    def forward(self, input):
        input = np.array(input).reshape(self.n_input,1)
        x = np.append(np.array([[1]]), input, axis=0)
        a = self.sigmoid(self.hidden_layer(x, self.v))
        a = np.array(a).reshape(self.n_hidden,1)
        a = np.append(np.array([[1]]), a, axis=0)
        a1 = np.ndarray.copy(a)
        y = self.softmax(self.output_layer(a1, self.w))
        return (x,a,y)
     
    def delta_output(self, prediction, label):
        return (prediction-label)*prediction*(1-prediction)
    
    def delta_hidden(self, a, delt_out):
        delt_hidd = np.zeros(shape=np.shape(a))
        for j in range(len(delt_hidd)):
            loc_sum = 0
            for index in range(len(self.w)):
                which_a = index // self.n_output
                which_out_node = index % self.n_output
                if which_a == j:
                    loc_sum += delt_out[which_out_node]*self.w[index]
            delt_hidd[j] = a[j]*(1-a[j])*loc_sum
        return delt_hidd

    def predict(self, attributes):
        prediction = np.zeros(shape=(len(attributes.index),1))
        for i in range(len(attributes.index)):
            x, a, y = self.forward(attributes.iloc[i])
            for j in range(len(y)):
                if y[j] == np.amax(y):
                    prediction[i] = j+1
        return prediction


    def fit(self, train_attrib, train_labels, val_attrib, val_labels):

        train_results = []
        val_results = []

        for epoch in range(self.epochs):

            train_missed, val_missed = 0., 0.

            for i in range(len(train_attrib)):

                tr_label =  np.array(train_labels[i]).reshape(self.n_output,1)

                #forward pass for training data
                x, a, y  = self.forward(train_attrib.iloc[i])

                #updating error term
                tr_pred_one_hot = np.zeros(shape=np.shape(y))
                for i in range(len(y)):
                    if y[i] == np.amax(y):
                        tr_pred_one_hot[i] = 1
                if not np.array_equal(tr_pred_one_hot, tr_label):
                    train_missed += 1

                #calculating gradients
                delt_out = self.delta_output(y,tr_label) #tensor of shape len(self.n_output)
                delt_hidd = self.delta_hidden(a, delt_out) #tensor of shape len(self.n_hidden)

                #calculating second layer weights updates
                new_w = np.zeros(shape=np.shape(self.w))
                for index in range(len(self.w)):
                    which_a = index // self.n_output
                    which_out_node = index % self.n_output
                    new_w[index] = self.w[index] - self.lr*delt_out[which_out_node]*a[which_a]
                
                #calculating first layer weights updates
                new_v = np.zeros(shape=np.shape(self.v))
                for index in range(len(self.v)):
                    which_in_node = index // self.n_hidden
                    which_hidd_node = index % self.n_hidden
                    new_v[index] = self.v[index] - self.lr*delt_hidd[which_hidd_node]*x[which_in_node]

                #updating weights   
                self.w = new_w
                self.v = new_v
            
            #forward pass for validation set
            for j in range(len(val_attrib)):
                val_label =  np.array(val_labels[j]).reshape(self.n_output,1)
                val_x, val_a, val_y  = self.forward(val_attrib.iloc[j])

            #updating error term for validation set
                val_pred_one_hot = np.zeros(shape=np.shape(val_y))
                for i in range(len(val_y)):
                    if val_y[i] == np.amax(val_y):
                        val_pred_one_hot[i] = 1
                if not np.array_equal(val_pred_one_hot, val_label):
                    val_missed += 1

            #calculating accuracy  
            train_acc = 1 - train_missed/len(train_attrib)
            val_acc = 1 - val_missed/len(val_attrib)
            train_results.append(train_acc)
            val_results.append(val_acc)

            if epoch == self.epochs - 1:
                print(f'Final Epoch ({epoch}) - Training Accuracy: {train_acc}, Validation Accuracy: {val_acc}')
        
        return train_results, val_results
