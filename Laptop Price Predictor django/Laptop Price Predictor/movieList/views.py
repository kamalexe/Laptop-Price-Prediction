from django.shortcuts import render
from django.http import HttpResponse
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from django.templatetags.static import static


import streamlit as st
import pickle


def home(request):
    pipe = pickle.load(open('pickle.pkl', 'rb'))
    df = pickle.load(open('df.pkl', 'rb'))
    #df = df.to_dict()
    labels =[]
    for i in df:
        labels.append(i)

    Company=df['Company'].unique()
    TypeName = df['TypeName'].unique()
    Weight = df['Weight'].unique()
    Ram = df['Ram'].unique()
    Touchscreen = df['Touchscreen'].unique()
    Ips = df['Ips'].unique()
    ppi = df['ppi'].unique()
    Cpu_brand = df['Cpu brand'].unique()
    HDD = df['HDD'].unique()
    SSD = df['SSD'].unique()
    gpu_Brand = df['Gpu Brand'].unique()
    os = df['os'].unique()
    pars = {'df':df,'labels':labels,'Company':Company,'TypeName':TypeName,'Ram':Ram,'Weight':Weight,'Touchscreen':Touchscreen,'Ips':Ips,'ppi':ppi,'Cpu_brand':Cpu_brand,'HDD':HDD,'SSD':SSD,'gpu_Brand':gpu_Brand,'os':os}
    return render(request, "home.html",pars)

def result(request):
    pipe = pickle.load(open('pickle.pkl', 'rb'))
    Company = request.GET.get('company')
    TypeName = request.GET.get('TypeName')
    Ram = int(request.GET.get('Ram'))
    Weight = float(request.GET.get('Weight'))
    Touchscreen = int(request.GET.get('Touchscreen'))
    Ips = int(request.GET.get('Ips'))
    ppi = request.GET.get('ppi')
    Cpu_brand = request.GET.get('Cpu_brand')
    HDD = int(request.GET.get('HDD'))
    SSD = int(request.GET.get('SSD'))
    Gpu_Brand = request.GET.get('gpu_Brand')
    os = request.GET.get('os')

    # print(pipe)
    print(Company)
    print(TypeName)
    print(Ram)
    print(Weight)
    print(Touchscreen)
    print(Ips)
    print(ppi)
    print(Cpu_brand)
    print(HDD)
    print(SSD)
    print(Gpu_Brand)
    print(os)

    query = np.array([Company,TypeName,Ram,Weight,Touchscreen,Ips,ppi,Cpu_brand,HDD,SSD,Gpu_Brand,os])
    query=query.reshape(1,12)
    final_price = int(np.exp(pipe.predict(query)))
    pars = {'final_price':final_price}
    return render(request, "result.html", pars)
