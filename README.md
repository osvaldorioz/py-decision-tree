El algoritmo de **Decision Tree (Árbol de Decisión)** es un modelo de aprendizaje supervisado utilizado para tareas de clasificación y regresión. Representa decisiones en forma de un árbol jerárquico, donde:  

1. **Nodos internos**: Representan atributos o características de los datos, con divisiones basadas en una regla de decisión (como un umbral de valor).  
2. **Ramas**: Representan los posibles resultados de una decisión en un nodo.  
3. **Hojas**: Representan las salidas finales (etiquetas de clase en clasificación o valores continuos en regresión).  

### Funcionamiento:
1. **Construcción**: 
   - Selecciona el atributo que mejor divide los datos utilizando métricas como *Gini impurity*, *Entropía (ID3)*, o *Varianza Reducida*.
   - Divide los datos en subconjuntos según la regla de decisión.
   - Repite el proceso recursivamente en cada nodo hasta cumplir un criterio de parada (por ejemplo, máxima profundidad o mínimo número de datos por nodo).

2. **Predicción**:  
   - Para clasificar o predecir, un dato atraviesa las ramas del árbol según las reglas de decisión hasta llegar a una hoja.

### Ventajas:
- Fácil de interpretar y visualizar.
- Maneja datos categóricos y continuos.
- No requiere mucha preparación de los datos.

### Desventajas:
- Propenso al *overfitting* si no se limita su profundidad.
- Sensible a pequeños cambios en los datos, lo que puede generar árboles diferentes (inestabilidad).

El algoritmo puede mejorarse mediante métodos como **poda** o combinado en ensamblajes (e.g., Random Forest).
