#! Python 3


'''

TODO:
   Resize all loaded files to be the same.
        ERROR REC. -  Value Error: Dimension 0 in both shapes must be equal....
        Proposed - get length of largest loaded file, zero pad smaller files to match

    Verify that slice up all files is happening correctly 


'''

import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
import matplotlib.pyplot as plt

from librosa.display import specshow
from librosa.feature import mfcc
from librosa.core import load


'''^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Database of words which the voice commands can be.
    Words: yes, no, up, down, left, right, on, off

    These will be expanded later to:
        CMD: "Trun" (right or left) (degree value 1 to 180) "degrees"
        CMD: "Go Forward" (value 0 to 180) "centimeters"
        CMD: "Go Backwards" (value 0 to 180) "centimeters"
        CMD: "Turn Off"
    Maybe more to come...
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^'''
SIZE_OF_DATABASE = 100

global words_in_database
words_in_database = [SIZE_OF_DATABASE]
words_in_database = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off']

#for words in words_in_database:
#    print(words)

'''^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
audiofile(file_dir)
    Load all the files from the provided folder 
    inputs
        file directory -> string type path variable
            file type to load -> .wav
    output
        audio_array -> multi-dimensional matrix of audio_array files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^'''
def AudioFileLoad(file_dir):
    audio_matrix = []
    file_array = []
    fs_matrix = []
    mfccs = []
    
    elem_length = 0
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
               
                #   Read the audio file
                afile, sample_rate = load(wav_file_path)
                
                #audio_bin = tf.read_file(wav_file_path)
                #desired_channels = 2
                #wav_form = contrib_audio.decode_wav(audio_bin, desired_channels=desired_channels)

                #with tf.Session() as sess:
                #    sample_rate, audio = sess.run([wav_form.sample_rate, wav_form.audio])

                #   Generate an array of the audio file loaded
                audio_array = np.array(afile, dtype=float)
                #audio_array = np.array(audio, dtype=float)
                audio_matrix.append(audio_array)
                
                #   Set an array containing the sample_rate 
                fs_matrix.append(sample_rate)
                #   Get the average mel function of the audio files
                mfcc_a = mfcc(y=audio_array, sr=sample_rate)#, n_mfcc=40)
                mfcc_mean = np.mean(mfcc_a.T, axis=0)
                mfccs.append(mfcc_mean)

                ''' (OPTIONAL) Ploting the loaded audio file mfccs

                plt.figure()
                specshow(mfcc_a, x_axis='time')
                plt.colorbar()
                plt.title('MFCC')
                plt.tight_layout()
                plt.show()
                
                print("File " + str(file) + " loaded successfully")
                print(audio_matrix[numfile])
                '''

                numfile += 1
            else:
                print("ERROR: Incorrect File Type. Expecting .wav File. File Loaded: " + str(filename))
                
        print("%i Audio file loaded successfully." % numfile)
    else:
        print("ERROR: Invalid file path for provided audio files. File Path: " + str(file_dir))
        exit()

    return (audio_matrix, fs_matrix, mfccs)


'''^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
GetTensor(load_audio_matrix)
    This function takes the loaded audio array matrix and converts it into a 
	tensor array matrix.
    inputs
        Load_audio_matrix -> loaded audio array matrix raw data
    output
        tensor_matrix -> multi-dimensional matrix of audio tensor file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^'''
def GetTensor(loaded_audio_matrix):
    tensor_matrix = []
    sliced_matrix = []
    numtensor = 0
    
    if loaded_audio_matrix is None:
        print("ERROR: Audio Matrix is Empty. (GetTensor())")
    else:
        print("Converting loaded audio file to a tensor matrix...")
        for audio_array in loaded_audio_matrix:
            if audio_array is not None:
                tensor = tf.convert_to_tensor(audio_array, dtype=tf.int32)

                tensor_matrix.append(tensor)
                #print("\n", tensor_matrix)
                
                #tensor_slices = tf.data.Dataset.from_tensor_slices(raw_tensor)
                #sliced_matrix.append(tensor_slices)
                #print(tensor_slices)

                #print("Length of tensor element %i is %i " % (numtensor, len(audio_array)))
                numtensor += 1
            else:
                print("ERROR: Audio Matrix Position Empty.(GetTensor())")
    
    print("Successfully converted %i audio matrix to tensor matix." % numtensor)
    
    return (tensor_matrix)#, sliced_matrix)

'''^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    rnn()
        This is the recurrent neural network funciton. 
        Voice input - pre-recorded to start, later will be altered to take 
        continuous audio input from raspberry pi audio jack with mic plugged 
        in. Mic will probably be a lab maded mic with a passive anaolg low 
        pass filter or bandpass filter.

        Input:

        Output:    

RNN paramaters 
    batch_size
    time_step
    learn_rate
    ets...
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^'''

global batch_size
batch_size = 256
global time_steps
time_steps = 5
global learning_rate
learning_rate = 0.0001
global training_iters
training_iters = 300000

def birnn(audio_file_matrix, atensor_matrix, fs_matrix):
    print("Processing through BiRNN...")

    '''
    Params
    '''
    learning_rate = 0.0001
    training_iters = 100000
    batch_size = 256
    display_step = 10

    n_step = 30
    n_hidden = 256
    n_inputs = 256
    n_classes = 8 # 8 elements on command list -> see words_in_database
    n_cell_dim = 100

    #define weights
    weights = {
        #Hidden layer weights => 2*n_hidden because fw & fb cells
        'out': tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    with tf.name_scope('weights'):
        for j in range(len(atensor_matrix)):
            #   Reshape to (n_steps*batch_size, n_input)
            #num_row, num_cols = size(atensor_matrix[j])#map(lambda i: i.value, tf.shape(atensor_matrix[j]))
            #print("\n\n r: %i c: %i" % (num_row, num_cols))
            print(atensor_matrix)
            inputs_x = tf.shape(tf.expand_dims(atensor_matrix, 2))
            inputs_x = tf.reshape(atensor_matrix[j], [-1, tf.size(inputs_x)])
            print(tf.shape(inputs_x))
            
            #   Split to get a list of 'n-steps' tensors of shape (batch_size, n_input)
            inputs_x = tf.split(inputs_x, n_step)
            print(tf.shape(inputs_x))

        weights_out1 = tf.Variable(tf.truncated_normal([2, n_hidden], stddev=np.sqrt(1./n_hidden)), name='weights')
        biases_out1  = weights_out1 = tf.Variable(tf.zeros(n_hidden), name='biases')
        #weights_out2 = tf.Variable(tf.truncated_normal([2, n_hidden], stddev=np.sqrt(1./n_hidden)), name='weights')
        #biases_out2  = weights_out1 = tf.Variable(tf.zeros(n_hidden), name='biases')

    with tf.name_scope('LTSM'):
        #   Forward directional cell
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        #   Backward directional cell
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    '''
    with tf.name_scope("dynamic_rnn"):
        rnn_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
        initial_state = rnn_cell.zero_state(batch_size, dtype=tf.int32)
        outputs, output_states = tf.nn.dynamic_rnn( cell=rnn_cell,
                                                    inputs=inputs_x,
                                                    initial_state=initial_state,
                                                    dtype=tf.int32)
        tf.summary.histogram("activations", outputs)
    
    '''
    with tf.name_scope('BiRNN'):
        X = tf.placeholder(tf.int32, shape=[None, n_step, n_inputs])
        X_seqs = tf.unstack(tf.transpose(X, perm=[1, 0, 2]))
        inputs_series = tf.split(seq_length, truncated_backprop_length, axis=1)
        basic_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=n_neurons)
        output_seqs, states = tf.nn.static_rnn(basic_cell, X_seqs, dtype=tf.float32)
        #exit()
        #   Feed 'layer 3' (layer output from fw to bw to birnn) into the LSTM BRNN 
        #   cell and get the LSTM BRNN output
        #outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
        #                                                         cell_bw=lstm_bw_cell,
        #                                                         #  Input is the previous fully 
        #                                                         #  connected layer before the LTSM
        #                                                         inputs=inputs_x,
        #                                                         dtype=tf.float32,
        #                                                         time_major=False),
        #                                                         sequence_length=seq_length)
        tf.summary.histogram("activations", outputs)

    exit()

    # Create a placeholder for the summary statistics
    with tf.name_scope("accuracy"):
        # Compute the edit (Levenshtein) distance of the top path
        distance = tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.targets)
    
        # Compute the label error rate (accuracy)
        self.ler = tf.reduce_mean(distance, name='label_error_rate')
        self.ler_placeholder = tf.placeholder(dtype=tf.float32, shape=[])
        self.train_ler_op = tf.summary.scalar("train_label_error_rate", self.ler_placeholder)
        self.dev_ler_op = tf.summary.scalar("validation_label_error_rate", self.ler_placeholder)
        self.test_ler_op = tf.summary.scalar("test_label_error_rate", self.ler_placeholder)


'''^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    MAIN Function call and script run section
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^'''
#   Set up and point to the recorded voice .wav files in /voice_recs folder
voice_recs_fldr = "voice_recs"
test_rec_folder = "test_recs"

#   Get the directory path to the current location specific to the computer running on
src_path = os.path.abspath(__file__)    #   /path/to/dir/foobar.py

#   Step back from the .py file to the general folder 
src_dir = r"C:\Users\Zachary\Documents\A_School\ECE578-9_IntelligentRobotics_1-2\Voice_Rec_DNN\voice_rnn\voice_command_brnn\voice_recs" 
#src_dir = os.path.split(src_path)[0]    #   i.e. /path/to/dir/

#   Join the desired voice_recs folder to the path to the project
#abs_file_path = os.path.join(src_dir,voice_recs_fldr)
abs_file_path = os.path.join(src_dir,test_rec_folder)

#	Load the audio files into a matrix or array of arrays or multidimensional array
audio_matrix, fs_matrix, mfccs = AudioFileLoad(abs_file_path) # output => audiomatrix ; type - Matrix ; length - specified by file

#	Convert the loaded audio array files into tensor arrays
atensor_matrix= GetTensor(audio_matrix)#audio_matrix) # output => atensormatrix ; type - Matrix of tensors ; length - audiomatrix
#, atensor_sliced 
#   Set up the Bidirectional Recursive Neural Net
birnn(audio_matrix, atensor_matrix, fs_matrix)