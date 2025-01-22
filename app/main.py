import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import decision_tree
from typing import List
import json

app = FastAPI()

# Definir el modelo para la matriz
class Matrix(BaseModel):
    matrix: List[List[float]]

# Definir el modelo para el vector
class VectorF(BaseModel):
    vector: List[int]
    

@app.post("/decision-tree")
def calculo(num_samples: int, num_features: int, max_depth: int):
    
    # Parámetros del conjunto de datos
    #num_samples = 10000  # Número de muestras
    #num_features = 10      # Número de características

    # Generar datos sintéticos
    np.random.seed(42)
    X_super_large = np.random.rand(num_samples, num_features) * 10  # Valores entre 0 y 10
    y_super_large = np.random.randint(0, 2, size=num_samples)       # Etiquetas 0 o 1

    # Convertir a listas (para compatibilidad con el módulo C++)
    X_super_large = X_super_large.tolist()
    y_super_large = y_super_large.tolist()

    # Crear y entrenar el árbol
    tree = decision_tree.DecisionTree()
    print("Entrenando el modelo con datos súper grandes...")
    tree.fit(X_super_large, y_super_large, max_depth)
    print("Entrenamiento completado.")

    # Predecir algunas muestras
    print("Realizando predicciones...")
    predictions = tree.predict(X_super_large[:100])  # Solo predecir para las primeras 100 muestras
    #print("Predicciones para las primeras 100 muestras:", predictions)

    j1 = {
        "Predicciones": predictions
    }
    jj = json.dumps(str(j1))

    return jj


    