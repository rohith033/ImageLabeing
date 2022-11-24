# -*- coding: utf-8 -*-
"""app.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZjvjjSw0Xf6trajmt3j_Q4C7TpTpJfve
"""

#/default_exp app

!pip install nbdev

!pip install fastai

!pip install gradio

#/export
from fastai.vision.all import *
import gradio as gr

#/export
learner = load_learner("model.pkl")

#/export
learner=load_learner("./model.pkl")

learner.predict("./panda.jpeg")

learner.predict("/content/black.jpeg")

learner.predict("/content/brown.jpeg")

#/export
cat =("black bear","brown bear","Griggly Panda","panda")
def classify(img):
  pred,_,probs=learner.predict(img)
  return dict(zip(cat,map(float,probs)))

classify("./panda.jpeg")

#/export
img = gr.inputs.Image(shape=(192,192))
label = gr.outputs.Label()
exp = ["/content/panda.jpeg","./black.jpeg","/content/brown.jpeg"]
interface = gr.Interface(fn=classify,inputs=img,outputs=label,examples=exp)
interface.launch(inline=False,debug=True,share=True)

import nbdev
nbdev.export.nb_export('app.ipynb', './')
print('Export successful')

