import pandas as pd
import numpy as np
import os


#------------------------------------------------------------------------
# DATA PREPROCESSING
#------------------------------------------------------------------------
def load_data(datapath):
    csv_path = os.path.abspath(datapath)
    return pd.read_csv(csv_path, header=None)

def data_split(dataset):
    #shuffling dataset
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    
    #splitting train/test
    train = dataset.sample(frac=0.8)
    val = dataset.drop(train.index)

    #splitting attributes/labels
    train_y = train.iloc[:,0]
    train_x = train.drop(0, axis=1)
    val_y = val.iloc[:,0]
    val_x = val.drop(0, axis=1)
    return train_x, train_y, val_x, val_y

def standardize(df):
    df_stand = df.copy()
    for column in df_stand.columns:
        df_stand[column] = (df_stand[column] - df_stand[column].mean()) / df_stand[column].std()    
    return df_stand

def onehotencode(series):
    vector = np.array(series)
    encoded = np.zeros(shape=(len(vector),3))
    for i in range(len(vector)):
        if vector[i] == 1:
            encoded[i] = np.array([1,0,0])
        if vector[i] == 2:
            encoded[i] = np.array([0,1,0])
        if vector[i] == 3:
            encoded[i] = np.array([0,0,1])
    return encoded

#---------------------------------------------------------------------------------------
# MLP
#---------------------------------------------------------------------------------------

class MLP:

    def __init__(self, n_hidden, n_input_feat, n_classes, learning_rate, epochs):
        self.n_hidden = n_hidden #4
        self.n_input = n_input_feat #13
        self.n_output = n_classes #3
        self.lr = learning_rate #0.0001
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
        # print(f'This is y: {y}')
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

    # def error(self, predictions, labels):
    #     sse = 0
    #     for i, j in zip(predictions, labels):
    #         sse += (i - j)**2
    #         print(f'i:{i}, j:{j}, new sse: {sse}')
    #     return sse*0.5

    def fit(self, train_attrib, train_labels, val_attrib, val_labels):

        for epoch in range(self.epochs):
            train_err, val_err = 0., 0.

            for i in range(len(train_attrib)):

                tr_label =  np.array(train_labels[i]).reshape(self.n_output,1)

                #forward pass for training data and updating error term
                x, a, y  = self.forward(train_attrib.iloc[i])
                train_err += np.mean((y-tr_label)**2).item()

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
            
            for j in range(len(val_attrib)):
                val_label =  np.array(val_labels[j]).reshape(self.n_output,1)
                val_x, val_a, val_y  = self.forward(val_attrib.iloc[j])
                val_err += np.mean((val_y-val_label)**2).item()
                    
            train_sse = train_err*0.5/len(train_attrib)
            val_sse = val_err*0.5/len(val_attrib)

            print(f'Epoch {epoch} - Training SSE: {train_sse}, Validation SSE: {val_sse}')
            pass


def main(datapath):
    #splitting dataset
    train_x, train_y, val_x, val_y = data_split(load_data(datapath))

    #standardizing attributes
    train_x_stand, val_x_stand = standardize(train_x), standardize(val_x)

    #one-hot-encoding labels
    train_y_hot = onehotencode(train_y)
    val_y_hot = onehotencode(val_y)
    myMLP = MLP(4,13,3,0.01,1000)
    myMLP.fit(train_x_stand, train_y_hot, val_x_stand, val_y_hot)
    pass
    # return prediction


# deriv of sigmoid: lambda z: (1/(1 + np.exp(-z)))*(1-(1/(1 + np.exp(-z))))

main('MLP_Classifier\wine.data')

#try shifted softmax if i can get this to work

    # def one_hot(self, vector):
    #     for entry in range(len(vector)):
    #         vector[entry] = 1 if vector[entry] == max(vector) else 0
    #     return np.reshape(vector,(1,3))
