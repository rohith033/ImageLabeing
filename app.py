import pip
failed = pip.main(["install", "fastai"])
import fastai
from fastai.vision.all import *
import gradio as gr
learner = load_learner("./model.pkl")
cat =("black bear","brown bear","panda")
def classify(img):
  pred,_,probs=learner.predict(img)
  return dict(zip(cat,map(float,probs)))
img = gr.inputs.Image(shape=(192,192))
label = gr.outputs.Label()
exp = ["./black.jpeg","./brown.jpeg","./panda.jpeg"]
interface = gr.Interface(fn=classify,inputs=img,outputs=label,examples=exp)
interface.launch(inline=False,debug=True,share=True)
