#! Python3

import os
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from librosa.core import load


def parser():
    audio_matrix = []
    file_array = []
    fs_matrix = []
    mfccs = []

    elem_length = 0
    numfile = 0
    #   Get the directory path to the current location specific to the computer running on
    #   Step back from the .py file to the general folder 
    test_rec_folder = "test_recs"
    src_dir = r"C:\Users\Zachary\Documents\A_School\ECE578-9_IntelligentRobotics_1-2\Voice_Rec_DNN\voice_rnn\voice_command_brnn\voice_recs" 

    #   Join the desired voice_recs folder to the path to the project
    file_dir = os.path.join(src_dir,test_rec_folder)
    
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
                try:   
                    #   Read the audio file
                    afile, sample_rate = load(wav_file_path)
                    #   Get the average mel function of the audio files
                    mfcc_a = mfcc(y=audio_array, sr=sample_rate, n_mfcc=40)
                    mfcc_mean = np.mean(mfcc_a.T, axis=0)
                except Exception as e: 
                    print("Error encountered while parsing file: " , file_dir)
                    return None, None

                feature = mfcc_mean
                label = str(os.path.splitext(file_dir)[0])
                
                return [feature, label]


parser()

temp = train.apply(parser, axis=1)
temp.columns = ['feature', 'label']

X = np.array(temp.feature.to_list())
y = np.array(temp.feature.to_list())

lb = LabelEncoder()

y = np_utils.to_categorical(lb.fit_tranform(y))


