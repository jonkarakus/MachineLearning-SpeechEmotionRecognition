    """
Emotion Recognition Using Machine Learning Techniques

This project involves the analysis of audio samples to recognize emotional states. The data samples were
processed using a machine learning model that employs Support Vector Machines (SVM) for training.
Cross-validation techniques were utilized to tune the model, ensuring robustness and accuracy.

Author: Jon Karakus
Credits: Please credit Jon Karakus if you use or reference this code in your projects. Thank you for respecting
the efforts involved in developing this application.

Note: If you are viewing this code on GitHub, please note that paths and sensitive information have been 
omitted to protect privacy.

Enjoy exploring this emotion recognition model!
"""

#----------------------------------------------------------------------------------------------------------------


import os
import librosa
import numpy as np
import sounddevice as sd 

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight
from scipy.io.wavfile import write
from scipy.signal import butter, lfilter
from sklearn.decomposition import PCA


import tempfile 



audio_path = 'C:\\Users\\jkara\\Desktop\\Audio'


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5*fs
    normal_cutoff = cutoff/nyq
    b, a = butter(order, normal_cutoff, btype ='high', analog=False)
    
    return b,a


def apply_highpass(audio, cutoff_freq, fs, order=5):
    b,a = butter_highpass(cutoff_freq, fs, order=order)
    filtered_audio = lfilter(b, a, audio)
    
    return filtered_audio



def feature_extraction(file_path):
    try:
        #loading audio file
        audio, sample_rate = librosa.load(file_path, res_type = 'kaiser_fast')

        
        #aduio padding 
        if len(audio) < 512:
            audio = np.pad(audio, (0, 512 - len(audio)), 'constant')

        #Apply highpass filter
        audio = apply_highpass(audio, 80, sample_rate) 

        #feature extraction
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_fft=512, n_mfcc = 40, hop_length = 512)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate, n_fft=512, hop_length = 512)
        mel = librosa.feature.melspectrogram(y=audio, sr = sample_rate, n_fft=512, hop_length = 512)
        contrast = librosa.feature.spectral_contrast(y=audio, sr = sample_rate, n_fft=512, hop_length = 512)
        tonnetz = librosa.feature.tonnetz(y = librosa.effects.harmonic(audio), sr=sample_rate, hop_length = 512)
    
        #feature combination 
        aggregated_features = np.hstack([np.mean(mfccs, axis=1),
                                        np.mean(chroma, axis =1),
                                        np.mean(mel, axis=1),
                                        np.mean(contrast, axis=1),
                                        np.mean(tonnetz, axis=1)])
        return aggregated_features
    except Exception as e:
        print(f"Error enconutered parsing file: {file_path}, error: {e}")
        
        return None



def process_audio_files(parent_folder):
    feature_labels = []

    for subdir, dirs, files in os.walk(parent_folder):
        for file in files:
            if file.lower().endswith('.wav'):
                emotion_label = os.path.basename(subdir)
                full_path = os.path.join(subdir, file)
                print(f'processing {full_path} ...')

                #Extract
                file_features = feature_extraction(full_path)

                if file_features is not None:
                    feature_labels.append((file_features, emotion_label))
                    
    return feature_labels



def record_live(duration=5, fs = 44300):
    
    #records live audio from the microphone for given duration and sample rate

    print("Recording In Progess .....")
    recording = sd.rec(int(duration*fs), samplerate = fs, channels=2, dtype='float32')
    sd.wait()   #Allowing recording to finish
    print("Recording Stopped")

    #Saving to a temp file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix = '.wav')
    write(temp_file.name, fs, recording)

    return temp_file.name



def predict_live_emotion(model, duration=5):

    audio_file = record_live(duration=duration)

    #Extract features from live audio
    features = feature_extraction(audio_file)

    if features is not None:
        features = np.expand_dims(features, axis=0) #reshape for single prediction
        prediction = model.predict(features)
        print(f"Predicted Emotion: {prediction[0]}")
    else:
        print("Could not extract features from live audio")

    # Cleanup
    os.remove(audio_file)   #deletes the tempfile
    
    
    

def main():
    feature_labels = process_audio_files(audio_path)
    print(f"Extracted features and labels from {len(feature_labels)} files.")

    #unpack features and labels
    features, labels = zip(*feature_labels)

    
    #converting to numpy array for SVM
    features = np.array(features)
    labels = np.array(labels)


    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(probability=True))])
    

    grid_parameters = {
        'svc__C':[0.1,1,10,100],                                #Regularization parameter
        'svc__gamma': ['scale', 'auto'],                        #Kernel coefficients
        'svc__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],    #Kernel types (Find which gave best results hint: rbf, sigmoid)
        'svc__class_weight': [None, 'balanced'],    
        'svc__degree': [2,3,4],
        'svc__max_iter': [400, 1100]
        }

    


    #Grid Search
    grid_search = GridSearchCV(pipeline, grid_parameters, cv = 5, n_jobs=-1, verbose =2)
    
    grid_search.fit(features_train, labels_train)
    
    #Pipeline to run through
    best_pipeline = grid_search.best_estimator_

    labels_pred = best_pipeline.predict(features_test)

    print(classification_report(labels_test, labels_pred))
    
    print("Best parameters found:", grid_search.best_params_)


    
    print("\nReady to Predict Your Emotion Live! Press ENTER to Start Recording ...")
    input()     # wait for the enter key

        
    predict_live_emotion(best_pipeline)

               
    
        

if __name__ == "__main__":
    main()


        
