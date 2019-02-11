from __future__ import print_function
import torch.autograd as autograd
from torch.autograd import Variable
import numpy as np
import pickle
import torch
import torch.nn as nn 
import torch.nn.functional as functionalnn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import pandas
import random

def preprocess(x):
    for ix in range(0, min(4, x.shape[2])):
        m, s = x[:,ix,:][(x[:,ix,:]>0)].mean(), x[:,ix,:][(x[:,ix,:]>0)].std()
        x[:,ix,:][((x[:,ix,:] - m ) > 3*s) ] = 0
        print(ix, m, s)

def load_data(TimeIn=[1*12,3*12+1], Timeout=[5*12-3, 5*12+3], outcomeIx=0):
    #(data, data_percentile, datakeys, datagenders)
    TimeIn = [int(i) for i in TimeIn]
    Timeout = [int(i) for i in Timeout]
    (d, dp, dk, dg, dethn, drace) = pickle.load(open('timeseries_data20170711-154021.pkl','rb'))
    vitals = ['BMI', 'HC', 'Ht Percentile', 'Wt Percentile'] #['BMI', 'HC', 'Ht', 'Wt']
    #dimension of data is N x |vitals| x |time=18*12mnths|

    gender_streched = np.repeat(np.array(dg).reshape(len(dg),1,1), d.shape[2], axis=2)
    ethn_dummy, race_dummy = np.array(pandas.get_dummies(dethn)), np.array(pandas.get_dummies(drace))
    ethn_streched = np.repeat(np.array(ethn_dummy).reshape(len(dethn), ethn_dummy.shape[1],1), d.shape[2], axis=2)
    race_streched = np.repeat(np.array(race_dummy).reshape(len(drace), race_dummy.shape[1],1), d.shape[2], axis=2)

    d = np.concatenate([d, gender_streched, ethn_streched, race_streched], axis=1)
    preprocess(d)

    print ('total num of ppl with any of the vitals measured at age '+ str(TimeIn[0]) + ' to '+str(TimeIn[1])+'(mnth):', ((d[:,0:len(vitals),TimeIn[0]:TimeIn[1]].sum(axis=2)>0).sum(axis=1)>0).sum())
    print ('total num of ppl with BMI measured at age '+ str(Timeout[0]) + ' to '+str(Timeout[1])+'(mnth):', ((d[:, outcomeIx, Timeout[0]:Timeout[1]].sum(axis=1)>0)).sum() )
    print ('total num of ppl with both of above consitions:', (((d[:, outcomeIx, Timeout[0]:Timeout[1]].sum(axis=1)>0)) & ((d[:,0:len(vitals), TimeIn[0]:TimeIn[1]].sum(axis=2)>0).sum(axis=1)>0)).sum() ) 

    ix_selected_cohort = (((d[:, outcomeIx, Timeout[0]:Timeout[1]].sum(axis=1)>0)) & ((d[:, 0:len(vitals), TimeIn[0]:TimeIn[1]].sum(axis=2)>0).sum(axis=1)>0))
    dInput = d[ix_selected_cohort,:,TimeIn[0]:TimeIn[1]]
    dOutput = d[ix_selected_cohort, outcomeIx, Timeout[0]:Timeout[1]]
    dInputPerc = dp[ix_selected_cohort,:,TimeIn[0]:TimeIn[1]]
    dOutputPerc = dp[ix_selected_cohort, outcomeIx, Timeout[0]:Timeout[1]]
    dkselected = np.array(dk)[ix_selected_cohort]
    dgselected = np.array(dg)[ix_selected_cohort]
    dethenselected = np.array(dethn)[ix_selected_cohort]
    draceselected = np.array(drace)[ix_selected_cohort]

    print('input shape:',dInput.shape, 'output shape:', dOutput.shape)
    return dInput, dOutput, dkselected, dgselected, dethenselected, draceselected

def split_train_valid_test(dInput, dOutput, dkselected=None, dgselected=None, ratioTest=0.25, ratioValid = 0.50):
    random.seed(0)
    assert dInput.shape[0] == dOutput.shape[0]
    ix = list(range(0,dInput.shape[0]))
    random.shuffle(ix)
    ix_test = ix[0:int(len(ix)*ratioTest)]
    ix_valid = ix[int(len(ix)*ratioTest):int(len(ix)*ratioValid)]
    ix_train = ix[int(len(ix)*ratioValid):]

    dInTrain, dOutTrain = dInput[ix_train,:,:], dOutput[ix_train,:]
    dInValid, dOutValid = dInput[ix_valid,:,:], dOutput[ix_valid,:]
    dInTest, dOutTest = dInput[ix_test,:,:], dOutput[ix_test,:]
    return dInTrain,dOutTrain, dInValid, dOutValid, dInTest, dOutTest

def augment_date(batchInput):
    error = np.random.normal(0, 0.1, batchInput.numpy().shape)
    error[(batchInput.numpy()==0)] = 0
    batchInput += torch.from_numpy(error).float()

def impute(batchInputNumpy):
    kernel = C(1.0, (1e-3, 1e3)) * RBF(20, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel)
    for dataix in range(0, batchInputNumpy.shape[1]):
        for dim in range(0, 4):
            X = np.nonzero(batchInputNumpy[:,dataix, dim])[0]
            if len(X) == 0:
                continue
            y = batchInputNumpy[:,dataix, dim][X]
            y_normed = (y - y.mean())
            gp.fit(X.reshape(-1, 1), y_normed.reshape(-1, 1))
            xpred = np.array(list(range(0, batchInputNumpy.shape[0]))).reshape(-1, 1)
            batchInputNumpy[:,dataix, dim] = gp.predict(xpred).ravel() + y.mean()
            # return(X,y,gp.predict(xpred).ravel() + y.mean() )
 
def plot_progress(trainloss, trainepoch, validloss, validepoch):
    plt.plot(trainepoch, trainloss)
    plt.plot(validepoch, validloss)
    plt.draw()
    plt.pause(1)

class LSTMPredictor_singleclassifier(nn.Module):
    def __init__(self, hidden_dim, input_dim, num_layers, dout, bfirst, target_size, minibatch_size, time_dim, bidirectional):
        super(LSTMPredictor_singleclassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.minibatch_size = minibatch_size
        self.time_dim = time_dim
        self.num_layers = num_layers
        self.directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dout, batch_first=bfirst, bidirectional=bidirectional)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2output = nn.Linear(hidden_dim * self.directions * self.time_dim , target_size)
        self.hidden = self.init_hidden(minibatch_size)

    def init_hidden(self,minibatch_size): # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros( self.num_layers * self.directions, self.minibatch_size, self.hidden_dim)),
                autograd.Variable(torch.zeros( self.num_layers * self.directions, self.minibatch_size, self.hidden_dim)))
    
    def forward(self, input):        
        lstm_out, self.hidden = self.lstm(input)
        # lstm_out_trans = (torch.transpose(lstm_out.contiguous(), 0, 1))
        # print(lstm_out.size())
        net_out1 = self.hidden2output(lstm_out[:,:,:].contiguous().view(self.minibatch_size, self.hidden_dim * self.time_dim * self.directions))
        # net_softmaxout = functionalnn.log_softmax(net_out)
        return net_out1

class LSTMPredictor_multiclassifier(nn.Module):
    def __init__(self, hidden_dim, input_dim, num_layers, dout, bfirst, target_size, minibatch_size, time_dim, bidirectional):
        super(LSTMPredictor_multiclassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.minibatch_size = minibatch_size
        self.time_dim = time_dim
        self.num_layers = num_layers
        self.directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dout, batch_first=bfirst, bidirectional=bidirectional)

    def init_hidden(self, minibatch_size): # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(self.minibatch_size, self.num_layers * self.directions,  self.hidden_dim)),
                autograd.Variable(torch.zeros(self.minibatch_size, self.num_layers * self.directions,  self.hidden_dim)))
    
    def forward(self, input):        
        lstm_out, self.hidden = self.lstm(input)
        return lstm_out[-1]

class MLPPredictor(nn.Module):
    def __init__(self, hidden_dim, input_dim, num_layers, dout, target_size, time_dim, minibatch_size):
        super(MLPPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.num_layers = num_layers
        self.minibatch_size = minibatch_size
        self.mlp = []
        current_size =  time_dim * input_dim
        for i in range(0, self.num_layers - 1):
            self.mlp.append(nn.Linear(current_size, hidden_dim))
            self.mlp.append(nn.ReLU())
            current_size = hidden_dim
        self.mlp.append(nn.Linear(current_size, target_size))
        print(self.mlp)
        self.parameters = []
        for m in self.mlp:
            self.parameters.append(m.parameters())
            print(self.parameters)
        self.parameters = nn.ParameterList(self.parameters)

    def forward(self, input):
        print(out.size())
        out = self.mlp[0](input.view(self.minibatch_size, self.input_dim*self.time_dim))
        for m in self.mlp[1:]:
            print(out.size())
            out = m(out)
        return out


def build_train_dnn(dIn, dOut, dInValid, dOutValid, dInTest,dOutTest, model_file='xxx.pyth_pkl', 
    hidden_dim=100, dropout=0.5, batch_size=16, num_layers=3, gap=0, bidirectional=False, batch_first=False, model_type='mlp'):
    # set ramdom seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    avgTrainLoss, avgValidLoss, trainepoch, validepoch = [], [], [], []
    plt.ion()
    plt.show()
    # load data and make training set
    print('input data:', dIn.shape, dIn.__class__)
    if model_type in ['lstm1', 'lstm2']:
        dInputTransposed = dIn.transpose((2,0,1)).copy()
        dOutputTransposed = dOut.transpose().max(axis=0)    
        print('output data:', dOut.shape, dOut.__class__)
        dInValidtrans, dOutValidtrans = dInValid.transpose((2,0,1)).copy(), dOutValid.transpose().max(axis=0)
        dInTesttrans, dOutTesttrans = dInTest.transpose((2,0,1)).copy(), dOutTest.transpose().max(axis=0)
    elif model_type in ['mlp']:
        dInputTransposed = dIn.copy()
        dOutputTransposed = dOut.transpose().max(axis=0)    
        print('output data:', dOut.shape, dOut.__class__)
        dInValidtrans, dOutValidtrans = dInValid.copy(), dOutValid.transpose().max(axis=0)
        dInTesttrans, dOutTesttrans = dInTest.copy(), dOutTest.transpose().max(axis=0)
    seq_size = dIn.shape[2] - gap
    input_dim = dIn.shape[1]
    target_dim = 1
    totalbatches = int(dIn.shape[0]/batch_size)

    if model_type == 'lstm1':
        model = LSTMPredictor_singleclassifier(hidden_dim, input_dim, num_layers, dropout, batch_first, target_dim, batch_size, seq_size, bidirectional)
    elif model_type == 'lstm2':
        model = LSTMPredictor_multiclassifier(hidden_dim, input_dim, num_layers, dropout, batch_first, target_dim, batch_size, seq_size, bidirectional)
    elif model_type == 'mlp':
        model = MLPPredictor(hidden_dim, input_dim, num_layers, dropout, target_dim, seq_size, batch_size)

    loss_function = nn.MSELoss() #if classification nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01) #optim.Adam is faster
    
    try:
        model = torch.load(model_file)
        print('loaded model:', model_file)
        skip_train = True
    except:
        skip_train = False

    for epoch in range(3000):
        if skip_train == True:
            break
        total_loss = 0
        total_cnt = 0
        ix_shuffle = range(0,dIn.shape[0])
        random.shuffle(list(ix_shuffle))
        dInputTransposed_shuffled = dInputTransposed[:,ix_shuffle,:]
        dOutputTransposed_shuffled = dOutputTransposed[ix_shuffle]
        for batchIx in range(0, totalbatches):
            
            if model_type in ['lstm1', 'lstm2']:
                batchInputNumpy = dInputTransposed_shuffled[0:seq_size-gap, (batchIx*batch_size):(batchIx*batch_size) + batch_size, 0:input_dim]
            elif model_type in ['mlp']:
                batchInputNumpy = dInputTransposed_shuffled[(batchIx*batch_size):(batchIx*batch_size) + batch_size, 0:input_dim, 0:seq_size-gap,]
            
            batchInput = torch.from_numpy(batchInputNumpy).float()
            batchTarget1 = torch.from_numpy(dOutputTransposed_shuffled[(batchIx*batch_size):(batchIx*batch_size) + batch_size]).float()
            batchTarget2 = torch.from_numpy(dInputTransposed_shuffled[gap:seq_size, (batchIx*batch_size):(batchIx*batch_size) + batch_size, 0:input_dim]).float()
            if model_type in ['lstm1', 'mlp']:
                batchTarget = batchTarget1
            elif model_type in ['lstm2']:
                batchTarget = batchTarget2

            model.zero_grad()
            if model_type in ['lstm1', 'mlp']:
                model.hidden = model.init_hidden(batch_size)
            predictions = model(Variable(batchInput))
            loss = loss_function(predictions, Variable(batchTarget))
            total_loss += loss.data.numpy()[0]
            total_cnt += 1
            loss.backward()
            optimizer.step()

        print('average Train mse loss at epoch:', epoch, ' is:',total_loss/total_cnt)
        avgTrainLoss.append(total_loss/total_cnt)
        trainepoch.append(epoch)
        plot_progress(avgTrainLoss, trainepoch, avgValidLoss, validepoch)

        if (epoch % 10) == 0 :
            valid_loss = 0
            total_cnt_valid = 0
            for ixvalidBatch in range(0,int(len(dOutValid)/batch_size)):
                if model_type in ['lstm1', 'lstm2']:
                    validBatchIn = torch.from_numpy(dInValidtrans[0:seq_size-gap, (ixvalidBatch*batch_size):(ixvalidBatch*batch_size) + batch_size, :]).float()
                elif model_type in ['mlp']:
                    validBatchIn = torch.from_numpy(dInValidtrans[(ixvalidBatch*batch_size):(ixvalidBatch*batch_size) + batch_size, :, 0:seq_size-gap]).float()
                validbatchOut1 = torch.from_numpy(dOutValidtrans[(ixvalidBatch*batch_size):(ixvalidBatch*batch_size) + batch_size]).float()
                validbatchOut2 = torch.from_numpy(dInValidtrans[gap:seq_size, (ixvalidBatch*batch_size):(ixvalidBatch*batch_size) + batch_size, :]).float()
                if model_type in ['lstm1', 'mlp']:
                    validbatchOut = validbatchOut1
                elif model_type in ['lstm2']:
                    validbatchOut = validbatchOut2

                validPred = model(Variable(validBatchIn))
                loss = loss_function(validPred, Variable(validbatchOut))
                valid_loss += loss.data.numpy()[0]
                total_cnt_valid += 1
                timestr = time.strftime("%Y%m%d-%H%M%S")
                # torch.save(model, 'obesityat5'+timestr+'.pyth_lstm')
            print('   average Valid mse loss at epoch:', epoch, ' is:',valid_loss/total_cnt_valid)
            avgValidLoss.append(valid_loss/total_cnt_valid)
            validepoch.append(epoch)

    test_pred_all = np.zeros((dOutTesttrans.shape[0]),dtype=float)
    test_loss = 0
    total_cnt_test = 0
    for ixtestBatch in range(0,int(len(dOutTest)/batch_size)):
        testBatchIn = torch.from_numpy(dInTesttrans[0:seq_size, (ixtestBatch*batch_size):(ixtestBatch*batch_size) + batch_size, :]).float()
        testBatchOut = torch.from_numpy(dOutTesttrans[(ixtestBatch*batch_size):(ixtestBatch*batch_size) + batch_size]).float()
        testPred = model(Variable(testBatchIn))
        test_pred_all[(ixtestBatch*batch_size):(ixtestBatch*batch_size) + batch_size] = testPred.data.numpy().ravel().copy()
        loss = loss_function(testPred, Variable(testBatchOut))
        test_loss += loss.data.numpy()[0]
        total_cnt_test += 1

    print('average Test mse loss at epoch:', epoch, ' is:',test_loss/total_cnt_test)
    return test_pred_all, dOutTesttrans

