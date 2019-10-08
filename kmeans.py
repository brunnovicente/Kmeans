# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 13:14:36 2019

@author: brunn
"""
import random as rd
import numpy as np
import pandas as pd
from scipy.spatial import distance

class KMeans:
        
    def __init__(self, g = 3):
        self.g = g
    
    def agrupar(self, X):
        centros = self.selecionar_centroides_iniciais(X)
        self.centroides = centros.copy()
        
        cont = 1
        while True:        
            print('Executando ', cont)
            grupos = []
            for x in X:
                grupos.append(self.calcular_grupo(x, centros))
            
            dados = pd.DataFrame(X)
            dados['grupo'] = grupos
            centros = self.recalcular_centroides(dados)
            
            #Condição de parada
            if np.array_equal(self.centroides, centros):
                break
            else:
                self.centroides = centros.copy()
            cont += 1
                    
        return np.array(grupos)
    
    def selecionar_centroides_iniciais(self, X):
        centroides = []
        for i in np.arange(self.g):
            a = rd.randint(0, np.size(X, 0))
            centroides.append(X[a,:])
        return np.array(centroides)
    
    def recalcular_centroides(self, dados):
        centros = []
        for i in np.arange(self.g):
            grupo = dados.loc[dados['grupo']==i]
            grupo = grupo.drop(['grupo'], axis=1)
            
            if np.size(grupo,0) == 0:
                centros.append(self.centroides[i,:])
            else:
                centros.append(grupo.mean().values)
        return np.array(centros)
            
            
    
    def calcular_grupo(self, x, centroides):
        dis = distance.euclidean(x, centroides[0,:])
        grupo = 0
        for i in np.arange(1, self.g):
            d = distance.euclidean(x, centroides[i,:])
            if d < dis:
                grupo = i
                dis = d
        return grupo
        