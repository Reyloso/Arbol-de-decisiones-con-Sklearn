""" ejemplo arbol de decisiones con sklearn"""

#import bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import os
# para exportar en imagen
import pydot
import pydotplus
from IPython.display import Image

# import de informacion de archivo csv
data  =  pd.read_csv('car.csv', header = None)

# nombre de las columnas
data.columns = ['buying','maint','doors','persons','lug_boot','safety', 'class']

# convertir a informacion a datos numericos cuantificables

data.buying.replace(('vhigh', 'high', 'med', 'low'), (4, 3, 2, 1), inplace = True)
data.maint.replace(('vhigh', 'high', 'med', 'low'), (4, 3, 2, 1), inplace = True)
data.doors.replace(('2', '3', '4', '5more'), (1, 2, 3, 4), inplace = True)
data.persons.replace(('2', '4', 'more'), (1, 2, 3), inplace = True)
data.lug_boot.replace(('small', 'med', 'big'), (1, 2, 3), inplace = True)
data.safety.replace(('low', 'med', 'high'), (1, 2, 3), inplace = True)

# clase a la que pertenece cada vehiculo
data['class'].replace(('unacc', 'acc', 'vgood','good'), ('regular', 'normal', 'muy bueno', 'bueno'), inplace = True)

# se obtienen los valores de el array
dataset = data.values
x = dataset[:,0:6] # columna donde termina cada registro

# toma la columna que da la desicion 
# para este caso la 6 (la clase)  nos dice si debemos comprarlo o no
y = np.asarray(dataset[:,6], dtype = 'S6')

# libreria de inteligencia artificial y analisis de datos sklearn
# como queremos saber si comprar el auto o no, importamos el arbol de decisiones
from sklearn import tree 

# primero se importa para dividir informacion y segundo el valor que vamos poner a prueba
from sklearn.model_selection import train_test_split, cross_val_score
# y luego con lo que vamos a evaluar si nuestro algoritmo esta aprendiendo
from sklearn import metrics

#para exportar el arbol y poderlo colocar en la imagen
from sklearn.tree import export_graphviz
from io import StringIO



# dividir data 
# es recomendable dividir 80% aprendizaje y 20% pruebas
# se dividen los datos entre dato de entrenamiento y datos de prueba, la data de prueba es del 20 % la division aleatoria queda fija en cero
X_Train, X_Test, Y_Train, Y_Test = train_test_split(x, y, test_size=0.2, random_state=0 )

# determinamos la profundidad del arbol
tr =  tree.DecisionTreeClassifier(max_depth=10)

# se entrena el algoritmo
tr.fit(X_Train, Y_Train)

# se testea el arbol
y_pred = tr.predict(X_Test)
print(y_pred)

# probamos con unos datos propios
# y_pred = tr.predict([[1,1,4,3,3,3]])
# print("ejemplo propio ", y_pred)

# para medir el porcentaje de aprendizaje del arbol debe estar cercano al 100% pero nunca en el 100%
# si esto sucede indica que algo esta mal, en el aprendizaje del algoritmo
# igual si esta por debajo del 50%
score = tr.score(X_Test, Y_Test)
print("precición: %0.4f " % score)

# siguiente commit será la grafica