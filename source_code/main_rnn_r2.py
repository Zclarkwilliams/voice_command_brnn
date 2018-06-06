#! Python3

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from librosa.core import load
from librosa.feature import mfcc
from librosa.display import specshow
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

#------ Variables
global n_mfccs
n_mfccs         = 40
global n_input
n_input         = 26 # --> TODO: Figure this out programatically through input files
global numcontext
numcontext      = 10

global n_hidden
n_hidden        = 2048

global n_hidden_in_1
n_hidden_in_1   = n_hidden
global n_hidden_out_1
n_hidden_out_1  = n_hidden
global default_std_dev
default_std_dev = 0.046875 # Got this from DeepSpeech.py a mozilla deep birnn model
global n_cell_dim
n_cell_dim      = n_hidden # same size as hidden cell depth
global relu_clip
relu_clip       = 20.0 # RELU clipping for non-reccurent layers

global dropout
dropout         = [0.05, -1.01 -1.0, 0.0, 0.0, -1.0]
global random_seed
random_seed     = 4567

def ReadAudioFile(file_array, numfile):
    i = 0
    a_array = []
    audio_matrix = []
    a_files_array = []

    for i in numfile: 
        #   Read the audio file
        audio, sample_rate = load(file_array[i])
                
        #   Generate an array of the audio file loaded
        audio_array = np.array(afile, dtype=float)
        audio_matrix.append(audio_array)
        
        #   Set an array containing the sample_rate 
        fs_matrix.append(sample_rate)

        a_array = ConvertAudioToInputArray(audio, sample_rate, n_mfccs, numcontext)

        a_files_array.append(a_array)
    
    return (a_files_array)

def ConvertAudioToInputArray(audio, sr, num_mfccs, numcontext):
    #   Get the average mel function of the audio files
    mfcc_a = mfcc(y=audio, sr=sr, n_mfcc=num_mfccs)
    
    #   BiRNN stride = 2
    mfcc_a = mfcc_a[::2]

    #   one stride per timestep in the input
    num_strides = len(mfcc_a)

    #   add empty initial and final contexts
    empty_context = np.zeros((numcontext, num_mfccs), dtype=mfcc_a.dtype)
    mfcc_a = np.concatenate((empty_context, mfcc_a, empty_context))

    #   create a view into the array with overlapping strides of size
    #   numcontext (past) + 1 (present) + numcontext (future)
    window_size = 2*numcontext+1
    train_inputs = np.lib.stride_tricks.as_strided(
        mfcc_a, 
        (num_strides, window_size, num_mfccs), 
        (mfcc_a.stires[0], mfcc_a.stires[0], mfcc_a.stires[1]), 
        writeable=False)
    
    #   Flatten the second and third dimensions
    train_inputs = np.reshape(train_inputs, [num_strides, -1])

    #   Whiten inputs
    #   copy the strided array so we can write to it safely
    train_inputs = np.copy(train_inputs)
    train_inputs = (train_inputs - np.mean(train_inputs))/np.std(train_inputs)

    #   Return the training data
    return train_inputs

def AudioFileLoad(file_dir):
    audio_matrix = []
    numfile = 0
    
    if os.path.isdir(file_dir) == True:
        #   Valid directory, go to load files into a list
        print("Directory validated.")
        print("Loading audio files from " + str(file_dir) + "...")
        for file in os.listdir(file_dir):
			#    Decode the filename from the file system
            if file.endswith('.wav'):   #   Valid wav file type 
                #   Attach proper audio file name to file path
                wav_file_path = os.path.join(file_dir,file)
                file_array.append(wav_file_path)
                numfile += 1
            else:
                print("ERROR: Incorrect File Type. Expecting .wav File. File Loaded: " + str(filename))   
        print("%i Audio file loaded successfully." % numfile)
    else:
        print("ERROR: Invalid file path for provided audio files. File Path: " + str(file_dir))
        exit()

    return ReadAudioFile(file_array, numfile)

def BiRNN(batch_x, seq_length, dropout):
    #   Get the shape of the input batch
    batch_x_shape = tf.shape(batch_x)
    #   Reshape and prepare for first layer of BiRNN
    batch_x = tf.transpose(batch_x, [1, 0, 2])
    batch_x - tf.reshape(batch_x, [-1, n_input + 2*n_input*numcontext])

    #   Pass inputs through N hidden layerswith RELU clipped activation and dropout

    #   Input Layer 1
    bias_in_1   = tf.get_variable(name="bias_in_1", 
                                  shape=[n_hidden_in_1], 
                                  initializer=(tf.random_normal_initializer(stddev=default_std_dev)))
    weight_in_1 = tf.get_variable(name="weight_in_1", 
                                  shape=[n_input + 2*n_input*num_context, n_hidden_in_1], 
                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    layer_in_1  = tf.minimum(tf.nn.relu(tf.add(tf.matmul(batch_x, weight_in_1), bias_in_1)), relu_clip)
    layer_in_1  = tf.nn.dropout(layer_in_1, (1.0 - dropout[0]))

    #   Create LTSM cells for feed-forward and feed-backward layers of BiRNN
    #   Feed Forward Cells:
    ltsm_fw_cell = tf.contrib.rnn.BasicLTSMCell(n_cell_dim,  
                                                forget_bias=1.0, 
                                                state_is_truple=True, 
                                                reuse=tf.get_variable_scope().reuse)
    ltsm_fw_cell = tf.contrib.rnn.DropoutWrapper(ltsm_fw_cell,
                                                 input_keep_prob=1.0 - dropout[3],
                                                 output_keep_prob=1.0 - dropout[3],
                                                 seed=random_seed)
    #   Feed Backward Cells:
    ltsm_bw_cell = tf.contrib.rnn.BasicLTSMCell(n_cell_dim,  
                                                forget_bias=1.0, 
                                                state_is_truple=True, 
                                                reuse=tf.get_variable_scope().reuse)
    ltsm_bw_cell = tf.contrib.rnn.DropoutWrapper(ltsm_fw_cell,
                                                 input_keep_prob=1.0 - dropout[4],
                                                 output_keep_prob=1.0 - dropout[4],
                                                 seed=random_seed)
    
    #   Reshap last input layer shapped [n_steps, batch_size, n_cell_dim]
    #   to single tensorof shape [n_steps*batch_size, 2*n_cell_dim]
    layer_in_1 = tf.reshape(layer_in_1, shape=[-1, batch_x_shape[0], n_hidden_in_1])

    #   Feed input layer that was just reshapped into the BiRNN
    outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=ltsm_fw_cell,
                                                             cell_bw=ltsm_bw_cell,
                                                             inputs=layer_in_1,
                                                             dtype=tf.float32,
                                                             time_major=True,
                                                             sequence_length=seq_length)
    
    #   Reshap outputs shapped [n_steps, batch_size, n_cell_dim]
    #   to single tensorof shape [n_steps*batch_size, 2*n_cell_dim]
    outputs = tf.concat(outputs, 2)
    outputs = tf.reshape(outputs, [-1, 2*n_cell_dim])

    #   Feed outputs to the output layer with clipped RELU activation and dropout
    bias_out_1   = tf.get_variable(name="bias_out_1", 
                                   shape=[n_hidden_out_1], 
                                   initializer=(tf.random_normal_initializer(stddev=default_std_dev)))
    weight_out_1 = tf.get_variable(name="weight_out_1", 
                                   shape=[(2*n_cell_dim), n_hidden_out_1], 
                                   initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    layer_out_1 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(outputs, weight_out_1), bias_out_1)), relu_clip)
    layer_out_1 = tf.nn.dropout(layer_out_1, (1.0 - dropout[5]))

    #   Reshap output layer_N shapped [n_steps, batch_size, n_cell_dim]
    #   to single tensorof shape [n_steps*batch_size, 2*n_cell_dim]
    output_layer = tf.reshape(layer_out_1, [-1, batch_x_shape[0], n_hidden_out_1], name="logits")
    
    return output_layer