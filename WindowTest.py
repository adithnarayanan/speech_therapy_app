from kivy.app import App
from kivy.lang import Builder
from kivy.properties import NumericProperty
from kivy.properties import ObjectProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock

import librosa
import numpy as np
import os
import tensorflow as tf
import sounddevice as sd
from scipy.io.wavfile import write

new_model = tf.keras.models.load_model('D:\chell\Documents\Adith\PyCharmProjects\KivyProject_Windows\\venv\my_model')  # loads model from 'my_model' file

Builder.load_string('''
<AudioInterface>:
    orientation: 'vertical'
    padding: '50dp'
    spacing: '20dp'
    Label:
        id: state_label
        size_hint_y: None
        height: sp(40)
        text: '[u]Speech Therapy App[/u]'
        font_size: '24sp'
        markup: 'True'
    Label:
        id: state_label
        size_hint_y: None
        height: sp(40)
        halign: 'center'
        text: 'Instructions: Press button below to begin recording.\\nSay the sound /a/ aloud for 3 seconds.\\nThe recording will stop automatically and analyze'
    Button:
        id: record_button
        text: 'Start Recording'
        on_release: root.start_recording()
    Label:
        id: results_label
        size_hint_y: None
        height: sp(40)
        text: '(Results Will Be Display Here)'
''')


class AudioInterface(BoxLayout):
    '''Root Widget.'''

  #  isRecording = False;
  #  isProcessing = False;
   # record_button = ids['record_button']
    #record_button.disabled = False

    def start_recording(self):
      #  self.isRecording = True;
       # self.update_labels()
        record_button = self.ids['record_button']
        state_label = self.ids['state_label']
        record_button.disabled = True
        state_label.text = 'Recording...'

        fs = 22050  # Sample rate
        seconds = 3  # Duration of recording
        buffer = int(fs * 0.5) #500 ms buffer before and after (makes recording 2 seconds in length)

        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)

        record_button.disabled = True
        state_label.text = 'Recording...'
        sd.wait()
        write('output.wav', fs, myrecording)
      #  self.isRecording = False;
      #  self.isProcessing = True
        state_label.text = 'Processing...'
        #input_recording = myrecording[buffer:-buffer]
        self.predict()

    # def update_labels(self):
    #     record_button = self.ids['record_button']
    #     state_label = self.ids['state_label']
    #     isRecording = self.isRecording
    #     isProcessing = self.isProcessing
    #     if isRecording and not isProcessing:
    #         record_button.disabled = True
    #         state_label.text = 'Recording...'
    #     elif not isRecording and isProcessing:
    #         record_button.disabled = True
    #         state_label.text = 'Processing...'
    #     else:
    #         record_button.disabled = False
    #         state_label.text = 'Ready to Record'

    def predict(self):
        max_pad_len = 7223
        audio, sample_rate = librosa.load('output.wav', res_type='kaiser_fast')
        buffer = int(sample_rate*0.5)
        mfccs = librosa.feature.mfcc(y=audio[buffer:-buffer], sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        print (mfccs.shape)
        x_test = mfccs.reshape(1, 40, 7223, 1)
        a = new_model.predict(x_test)
        self.displayResults(a[0])

    def displayResults(self, output_array):
        results_label = self.ids['results_label']
        record_button = self.ids['record_button']
        state_label = self.ids['state_label']
        if output_array[0] > 0.5:
            results_label.text = 'Predicted: Healthy Voice' + str(output_array)
        else:
            results_label.text = 'Predicted: Pathological Voice' + str(output_array)
       # self.isProcessing = False;
        record_button.disabled = False
        state_label.text = 'Ready to Record'





class AudioApp(App):

    def build(self):
        return AudioInterface()

    def on_pause(self):
        return True


if __name__ == "__main__":
    AudioApp().run()