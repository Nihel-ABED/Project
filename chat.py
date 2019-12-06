import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import numpy as np
import tflearn
import tensorflow
import random
import json
import pickle
import pandas as pd

with open("intents.json") as file:
    data = json.load(file)
    
loadedmodel=pickle.load(open('trainedbot.sav','rb'))

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for question in intent["question"]:
        wrds = nltk.word_tokenize(question)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tags"])

    if intent["tags"] not in labels:
            labels.append(intent["tags"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)
training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1
     
    training.append(bag)
    output.append(output_row)


training = numpy.array(training)
output = numpy.array(output)

with open("data.pickle", "wb") as f:
    pickle.dump((words, labels, training, output), f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

# try:
#   model.load("model.tflearn")
# except:
model.fit(training, output, n_epoch=10, batch_size=8, show_metric=True)
model.save("model.tflearn")
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)

def conversation():
  inp = input("You: ")
  results = model.predict([bag_of_words(inp, words)])
  results_index = numpy.argmax(results)
  tags = labels[results_index]
  return (inp, tags)



def chat():
  while True:
    
    inp, tags=conversation()
    if inp.lower() == "quit":
        break
    ###########################################
    if tags == "thanks":
      test = pd.read_csv("H_H_test.csv")
      #test=test.values.tolist()
      t_c=test.columns
      t_v=test.values[0]

      for i in range(len(t_v)):
          if t_v[i]==" ":
            
            print(t_c[i] )
            inp = input("You: ")
            t_v[i]=float(inp)

      test=pd.DataFrame([t_v],columns=t_c)

      filename='Heart_Health.sav'
      ymodel =pickle.load(open(filename, 'rb'))
        
      result=ymodel.predict(test)
      if result==0:
        answer=["You are safe my friend, you can rest and then you will be ok."]
      else:
        answer=["Maybe you have some trouble, I advice you to check your doctor. Would you let me make an appointment?"]
    ###############################################
    elif tags == "goodbye":
      test = pd.read_csv("H_F_test.csv")
      #test=test.values.tolist()
      t_c=test.columns
      t_v=test.values[0]

      for i in range(len(t_v)):
          if t_v[i]==" ":
            print("Can you answer this question: ", t_c[i] )
            inp = input("You: ")
            t_v[i]=float(inp)

      test=pd.DataFrame([t_v],columns=t_c)

      filename='Heart_failure.sav'
      ymodel =pickle.load(open(filename, 'rb'))
        
      result=ymodel.predict(test)
      if result==0:
        answer=["I just checked your medical file, you can rest and then you will be ok."]
      else:
        answer=["Now you should take your drugs. Can I make an appointment with your doctor for you as soon as possible?"]

    ###########################################
    elif tags == "greeting":
      test = pd.read_csv("D_P_test.csv")
      #test=test.values.tolist()
      t_c=test.columns
      t_v=test.values[0]

      for i in range(len(t_v)):
          if t_v[i]==" ":
            print("Can you answer this question: ", t_c[i] )
            inp = input("You: ")
            t_v[i]=float(inp)

      test=pd.DataFrame([t_v],columns=t_c)

      filename='Diabetes-Prediction.sav'
      ymodel =pickle.load(open(filename, 'rb'))
        
      result=ymodel.predict(test)
      if result==0:
        answer=["You are all safe, take a cup of water and stay away from fats."]
      else:
        answer=["You should consume more water my friend. Also you have to check your doctor if this comes again."]
    #############################################
    else:
      for tg in data["intents"]:
          if tg['tags'] == tags:
              answer = tg['answer']

    print(random.choice(answer))
chat()




