##################################################################################
# Configuration of all the Datasets and Networks
# Author: Javier Fañanás Anaya
# Email: javierfa@unizar.es
##################################################################################
import os

import libs.metrics as metrics

class Config:
    def __init__(self, dataset_name):

        self.dataset_name = dataset_name
        self.dataset_dir = os.path.join('datasets',dataset_name,'series') 
        self.scalers_dir = os.path.join('datasets',dataset_name,'scalers')
        self.networks_dir = os.path.join('datasets',dataset_name,'networks')
        self.losses_dir = os.path.join('datasets',dataset_name,'losses')

        #If there is no GPU available, use '/cpu:0'
        self.device = '/gpu:0'

        # Wiener Hammerstein Benchmark:
        # https://www.nonlinearbenchmark.org/benchmarks/wiener-hammerstein-process-noise
        if dataset_name == 'wiener_hammerstein':

            #DATA CONFIG:
            #Validation and Test size
            self.val_size = 0.05
            self.test_size = 0.0026
            #Shuffle all the series
            self.shuffle = False

            #Batch lenght (Prediction horizon during training/validation)
            #With length = -1 AND batch_size == 1, each data series will have its maximum length
            self.length =  2048
            self.batch_size = 16 

            #Buffer/Historical
            self.buffer = 10

            #Input: u
            self.input_size = 1
            self.input_labels = ["u"]
            #State: y
            self.state_size = 1
            self.state_labels = ["y"]

            #Data scale
            self.scale_min = -1
            self.scale_max = 1

            #NETWORK CONFIG
            #Multihead-attention
            self.multihead_attention = True
            self.num_heads = 4
            self.key_dim = 8

            #Lstm layers
            self.lstm_layers = 1
            self.lstm_units = 64
            self.lstm_activation = 'tanh'
            self.lstm_recurrent_activation = 'sigmoid'

            #Dense layers
            self.dense_layers = 2
            self.dense_units = 128
            self.dense_activation = 'relu'

            self.dropout = 0
            self.l2_alpha = 0

            #TRAINING CONFIG
            #optimizer = Adam
            self.loss = metrics.mse
            self.loss_name = 'mse'
            self.only_notf = True #Training ONLY without Teacher-Forcing (TF)

            self.epochs_tf = 50 #Max epochs during TF training
            self.lr_tf = 0.001 #Learning Rate used during TF training
            self.early_stop_tf = 3 #Early stop during TF Training

            self.epochs_notf = 200 #Max epochs during no-TF training
            self.lr_notf  = [0.001, 0.0001, 0.00001] #Learning Rate used during no-TF training
            self.early_stop_notf = 3 #Early stop during no-TF Training

        # Industrial Robot Benchmark:
        # https://www.nonlinearbenchmark.org/benchmarks/industrial-robot
        elif dataset_name == 'robot':

            #DATA CONFIG:
            #Validation and Test size
            self.val_size = 0.072
            self.test_size = 0.082
            #Shuffle all the series
            self.shuffle = True

            #Batch lenght (Prediction horizon during training/validation)
            #With length = -1 AND batch_size == 1, each data series will have its maximum length
            self.length =  600
            self.batch_size = 6 

            #Buffer/Historical
            self.buffer = 10

            #Input: u (Nm)
            self.input_size = 6
            self.input_labels = ["u_1","u_2","u_3","u_4","u_5","u_6"]
            #State: y (deg)
            self.state_size = 6
            self.state_labels = ["y_1","y_2","y_3","y_4","y_5","y_6"]

            #Data scale
            self.scale_min = -1
            self.scale_max = 1

            #NETWORK CONFIG
            #Multihead-attention
            self.multihead_attention = True
            self.num_heads = 8
            #self.key_dim = (self.input_size + self.state_size)*2
            self.key_dim = 24
            
            #Lstm layers
            self.lstm_layers = 1
            self.lstm_units = 128
            self.lstm_activation = 'tanh'
            self.lstm_recurrent_activation = 'sigmoid'

            #Dense layers
            self.dense_layers = 2
            self.dense_units = 512
            self.dense_activation = 'relu'

            self.dropout = 0
            self.l2_alpha = 0

            #TRAINING CONFIG
            #optimizer = Adam
            self.loss = metrics.r2
            self.loss_name = 'r2'
            self.only_notf = True #Training ONLY without Teacher-Forcing (TF)

            self.epochs_tf = 50 #Max epochs during TF training
            self.lr_tf = 0.001 #Learning Rate used during TF training
            self.early_stop_tf = 3 #Early stop during TF Training

            self.epochs_notf = 200 #Max epochs during no-TF training
            self.lr_notf  = [0.001, 0.0001, 0.00001] #Learning Rate used during no-TF training
            self.early_stop_notf = 5 #Early stop during no-TF Training

        else:
            print("Dataset ["+ dataset_name +"] not found")
            exit()