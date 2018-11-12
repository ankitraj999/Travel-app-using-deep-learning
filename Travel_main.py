import pandas as pd
import numpy as np
import os
from keras.models import Model
from keras.models import load_model
from keras.regularizers import l2
from keras.layers import Input, Dense, LSTM, Dropout, Conv1D, MaxPool1D, Flatten, Reshape
from keras.layers.wrappers import TimeDistributed
from keras.layers.merge import concatenate
from keras.utils import plot_model
import vlc
import time
import speech_recognition as sr

df = pd.read_csv('data.txt',sep="|",header=None,names=['sentence','label','intent'])
data_arr = df.values
s_list, l_list, i_list = [],[],[]
for row in data_arr:
    s_list.append(row[0])
    l_list.append(row[1])
    i_list.append(row[2])
  # length of sentence list ,label list and intent list
print([len(x) for x in [s_list,l_list,i_list]])
  # encode l_list
  # encode index = 0 for l_list, to be used for padding
  # prepare dictionary

l_dict = {}
l_dict['_unknown'] = 0
for index, label in enumerate(set(x for y in l_list for x in y.split()),1):
    l_dict[label] = index
    
print(l_dict)
  # pad all the labels to a length of 50

encoded_l = np.zeros((len(l_list),50,len(l_dict)))

  # iterate through the l_list

for row_index, row in enumerate(l_list):
    
    l = row.split()
    
    for num in range(50):
        if num < len(l):
            encoded_l[row_index][num][l_dict[l[num]]] = 1
        else:
            encoded_l[row_index][num][0] = 1
  # encode i_list
  # prepare dictionary

i_dict = {}
for index, label in enumerate(set(x for y in i_list for x in y.split())):
    i_dict[label] = index
    
print(i_dict)
  
encoded_i = np.zeros((len(l_list),len(i_dict)))

  # iterate through the i_list

for intent_index, intent in enumerate(i_list):
    encoded_i[intent_index][i_dict[intent]] = 1
from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format('glove-wiki-gigaword-300.txt')
embedded_s = np.zeros((len(s_list),50,300))

  # iterate through the s_list

for s_index, s in enumerate(s_list):
    words = s.split()
    
    for num in range(50):
        if num < len(words):
            try: 
                embedded_s[s_index][num] = model.wv.get_vector(words[num].lower())
            except KeyError:
                
                print(words[num].lower(),'not in vocab')
                embedded_s[s_index][num] = model.wv.get_vector('unk')
intent_target = np.zeros((len(i_list),3))
  # 0 = Travel, 1 = Meal, 2 = Travel,Meal

for index, i in enumerate(i_list):
    if i == 'Travel':
        intent_target[index] = [1,0,0]
    elif i == 'Meal':
        intent_target[index] = [0,1,0]
    else:
        intent_target[index] = [0,0,1]
#reversed_l_dict = dict((v,k) for k,v in l_dict.items())
#reversed_l_dict[0] = ''
reversed_l_dict={0:'',1:'veg',2:'destination',3:'source',4:'Non-veg',5:'O',6:'day'}
intent_dict = {0:'Travel',1:'Meal',2:'Travel,Meal'}

#keras model defination for intent
def intent():
    visible = Input(shape=(50,300))
    hidden1 = Dropout(0.2)(visible)
    hidden2 = LSTM(100,return_sequences=True)(hidden1)

    # CNN layers 
    c_hidden1 = Conv1D(filters=50,kernel_size=5)(hidden2)
    c_hidden2 = MaxPool1D(pool_size=5)(c_hidden1)

    c_extract = Reshape((50,9))(c_hidden2)


    # CNN layers 2

    c2_hidden1 = Conv1D(filters=50,kernel_size=2)(hidden2)
    c2_extract = Reshape((50,49))(c2_hidden1)

    # Time distributed layers for labelling
    t_hidden1 = Dense(64,activation='relu',kernel_regularizer=l2(0.01))(hidden2)
    t_hidden2 = Dropout(0.2)(t_hidden1)
    t_hidden3 = concatenate([c_extract,c2_extract,t_hidden2])
    t_hidden4 = Flatten()(t_hidden3)
    t_hidden5 = Dense(16, activation='relu')(t_hidden4)
    t_output = Dense(3,activation='softmax')(t_hidden5)

    pred_model = Model(inputs=visible,outputs=t_output)

    return pred_model

#keras model defination for label
def label():
    visible = Input(shape=(50,300))
    hidden1 = Dropout(0.2)(visible)
    hidden2 = LSTM(100,return_sequences=True)(hidden1)

    # CNN layers for classification
    c_hidden1 = Conv1D(filters=50,kernel_size=5)(hidden2)
    c_hidden2 = MaxPool1D(pool_size=5)(c_hidden1)

    c_extract = Reshape((50,9))(c_hidden2)

    # Time distributed layers for labelling
    t_hidden1 = Dense(64,activation='relu')(hidden2)
    t_hidden2 = Dropout(0.2)(t_hidden1)
    t_hidden3 = concatenate([c_extract,t_hidden2])
    t_output = TimeDistributed(Dense(7,activation='softmax'))(t_hidden3)

    pred_model = Model(inputs=visible,outputs=t_output)
    

    
    return pred_model


label_model=label()
intent_model=intent()

#play file
def play(text,p='ankit_gtts_test.mp3'):
    import gtts
    import os
    import vlc
    tts=gtts.gTTS(text,lang="en")
    tts.save("/home/raj/Downloads/Google-API-key&gcd sdk/"+p)
    vlc_ins=vlc.MediaPlayer('/home/raj/Downloads/Google-API-key&gcd sdk/'+p)
    vlc_ins.play()

# google api speech to text
def recognise_voice():
    r = sr.Recognizer()
    m = sr.Microphone()
    value='Error'
    try:
    
        with m as source:
            r.adjust_for_ambient_noise(source)
        
        
        print("Say something!")

        with m as source:
            audio = r.listen(source)

        print("listened")
           
        try:
              # recognize speech using Google Speech Recognition
                value = r.recognize_google(audio)
                print("Now to recognize it...")
                #play("Got it!")

            
                print("You said: {}".format(value))
        except sr.UnknownValueError as ev:
                
                value='Error'

                play("Sorry! some Unknown Error. Please say again... ")  
                print("Unknow Error")
        except sr.RequestError as e:
                
                value='Error'
                play("Sorry! some Connection Request Error. Please say again...")
                
                print('Request Error')
            
    except KeyboardInterrupt:
        pass
    return value


# loading/saving/training function for lable model 
def input_to_label(reconized_speech):
   # x = input().strip().lower()
    x=reconized_speech
    
    words = x.split()
    
    embedded_input = np.zeros((1,50,300))
    
    for index, word in enumerate(words):
        try:
            embedded_input[0][index] = model.get_vector(word)
        except KeyError:
            embedded_input[0][index] = model.get_vector('unk')
    print("printing embedded input")         
    print(np.mean(embedded_input))
    
    #output = label_model.predict(embedded_input)
    if not os.path.exists('lstm_label.h5'):
        #label_model=label()
        label_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        label_model.fit(x=embedded_s,y=encoded_l,epochs=4,batch_size=64,validation_split=0.3)
        #saving label
        label_model.save('lstm_label.h5')
        label_model.save_weights('lstm_label_weights.h5')
        output = label_model.predict(embedded_input)
        
    else:
        loadM=load_model('lstm_label.h5')
        loadM.load_weights('lstm_label_weights.h5')
        output = loadM.predict(embedded_input)
       # print("evaluation model")
       # scores = loadM.evaluate(s100_list,l100_list , verbose=0)
       # print("%s: %.2f%%" % (loadM.metrics_names[1], scores[1]*100))
    
    
    
    print("printing predicted output")
    print(np.mean(output))
    label_list = []
    
    for index, label in enumerate(output[0]):
        a=reversed_l_dict[np.argmax(label)]
        label_list.append(a)

        print(np.argmax(label), end="")
    print("\n predicted label", label_list)
    ak_dict=dict()
    n=len(words)
    for i in range(n):
        if label_list[i]!='O'and label_list[i]!='':
            print(label_list[i])
            if label_list[i] in ak_dict:
                ak_dict[label_list[i]].append(words[i])
            else:
                ak_dict[label_list[i]]=[words[i]]
    
    
    
    return ak_dict
# loding/training/saving function for intent_model
def input_to_intent(reconized_speech):
    global embedded_s
    global encoded_l
    global intent_target
    x=reconized_speech
    words = x.split()
    
    embedded_input = np.zeros((1,50,300))
    
    for index, word in enumerate(words):
        try:
            embedded_input[0][index] = model.get_vector(word)
        except KeyError:
            embedded_input[0][index] = model.get_vector('unk')
    
    if not os.path.exists('lstm_intent.h5'):
        #intent_model=intent()
        intent_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        intent_model.fit(x=embedded_s,y=intent_target,epochs=30,batch_size=64,validation_split=0.3)
        #saving intent model
        intent_model.save('lstm_intent.h5')
        intent_model.save_weights('lstm_intent_weights.h5')
        outputs = intent_model.predict(embedded_input)
        intent = intent_dict[np.argmax(outputs[0])]
    else:
        loadM=load_model('lstm_intent.h5')
        loadM.load_weights('lstm_intent_weights.h5')
        outputs = loadM.predict(embedded_input)
        intent = intent_dict[np.argmax(outputs[0])]
    
    return intent

# listening function
def detect_fun():
    while True:
        
        
        print('type start to start listening')
        st_listen=input()
        if st_listen=='start':
            reconized_speech=recognise_voice()
            if reconized_speech!='Error':
            
                play('captured voice '+reconized_speech)
                break
    return reconized_speech

ret=''
def start_fun():
 play("Welcome to Travel app  ")
 time.sleep(0.5)
 play("where do you want to travel?")
 reconized_speech=detect_fun()
 print('want to continue: type yes or no or exit')
 x=input()
 
 if x=='no':
   global ret
   ret=x
   return
 if x=='exit':
   return
 single_dict={'Meal':['veg','Non-veg'],'Travel':['source','destination','day'],'Travel,Meal':['veg','Non-veg','source','destination','day']}
 out=input_to_intent(reconized_speech)
 print(out)
 label_dict=input_to_label(reconized_speech)
 print('printing label dictionary')
 print(label_dict)
 b_val=single_dict[out]
 print("printing final dictionary intent key value")
 print(b_val)
 m=len(b_val)
 final_list=[]
 day_list=['Sunday','Monday','Tuesday','Wednesday','Thrusday','Friday','Saturday']
 context_dict={'day':'Tell me the day of travelling','source':'Tell me your departure place','destination':'Tell me your destination','veg':'Tell me veg meal','Non-veg':'Tell me Non-veg meal'}
 for i in range(m):
    if b_val[i] in label_dict.keys():
        print(b_val[i],label_dict[b_val[i]])
        final_list.append(b_val[i])
        final_list.append("is")
        fl=' and '.join(label_dict[b_val[i]])
        final_list.append(fl)
        final_list.append('.')
    else:
        
        if b_val[i] in context_dict:
            play(context_dict[b_val[i]])
            detect_v=detect_fun()

            if b_val[i]=='day':
             while True:
              if detect_v in day_list:
                break
              else:
                play(context_dict[b_val[i]])
                detect_v=detect_fun()
            if b_val[i]=='source':
             while True:
                if 'destination' in label_dict.keys():
                    if label_dict['destination']!=detect_v:
                        break
                    else:
                        play(context_dict[b_val[i]])
                        detect_v=detect_fun() 
                        
            		
            if b_val[i]=='destination':
             while True:
                if 'source' in label_dict.keys():
                    if label_dict['source']!=detect_v:
                        break
                    else:
                        play(context_dict[b_val[i]])
                        detect_v=detect_fun()
            final_list.append(b_val[i])
            final_list.append('is')
            final_list.append(detect_v)
            print(b_val[i],detect_v)
       
            
 final_st=" ".join(final_list)
 time.sleep(.5)
 print(final_st)
 play(final_st)
 time.sleep(3)
 play("Your ticket is booked. Thank you! for being with us")

start_fun()
while ret =='no':
  start_fun()
