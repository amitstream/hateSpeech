#!/usr/local/bin/python3.9

import streamlit as st
import requests
import json

import datetime
print(datetime.datetime.now(),"Program start.")

def get_prediction_aiclub(s):
  data={"Tweet_Text":s}
  url = 'https://askai.aiclub.world/bb38c320-72aa-4397-be1c-77bc95787625'
  r = requests.post(url, data=json.dumps(data))
  response = getattr(r,'_content').decode("utf-8")
  print("Response:",response)
  b1=json.loads(response)["body"]
  j2=json.loads(b1)
  p=j2["predicted_label"]
  confidence=j2["confidence_score"]
  s=sorted(confidence.items(), key=lambda x: x[1], reverse=True)
  return p,s

st.title("Hate Speech Detector")
sentence=st.text_input('Sentence to analyze')

labels=['Homophobe', 'Sexist', 'OtherHate', 'NotHate', 'Religion', 'Racist']
msgs={
      'Homophobe': 'Determined this to be a homophobic comment',
      'Sexist':'AI says that this message is sexist', 
      'OtherHate':'AI believes that this is a hateful message', 
      'NotHate':'AI determines that this is not a hateful message', 
      'Religion':'AI determined this comment to be hateful to religious people', 
      'Racist':' AI says that this message is racist'
     }

paras={
      'Homophobe': 'Paragrapg for homophobic comments. \nSecond line for homophobic comment',
      'Sexist':'Para for AI says that this message is sexist', 
      'OtherHate':'Para for other hateful message', 
      'NotHate':'Para for not a hateful message', 
      'Religion':'Para for hateful to religious people', 
      'Racist':'Para for racist'
     }

if sentence:
  print("*************\nSentence:",sentence)
  (pr,s)=get_prediction_aiclub(sentence)
  cv=s[0][1]*100
  print(f"Saw {pr} with {s} for {cv}")
  c=f" with confidence: {cv:.1f}%"
  m=msgs[pr]
  p=paras[pr]
  
  print(p+c+m)
  st.title(m+c)
  st.text(p)
  print(datetime.datetime.now(),"Program end.")
