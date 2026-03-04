# Clasificación: Machine Learning

## Descripción del Proyecto
Este proyecto tiene como objetivo principal analizar datos de una campaña de marketing para predecir si los clientes de un banco invertirán su dinero o no. La predicción se realizará mediante **machine learning**, abarcando desde la **lectura**, **análisis exploratorio**, **separación** y **transformación** de los datos, hasta el **ajuste**, **evaluación** y **comparación** de diferentes modelos de clasificación.

## Conjunto de Datos

### Principal
El conjunto de datos principal utilizado es `marketing_inversiones.csv`. Contiene información sobre clientes de un banco y sus respuestas a una campaña de marketing. Las columnas incluyen:
- `edad`: Edad del cliente.
- `estado_civil`: Estado civil del cliente (casado(a), divorciado(a), soltero(a)).
- `escolaridad`: Nivel de escolaridad (primaria, secundaria, superior).
- `default`: Si el cliente tiene un crédito en mora (sí/no).
- `saldo`: Saldo promedio anual.
- `prestatario`: Si el cliente tiene un préstamo (sí/no).
- `ultimo_contacto`: Duración del último contacto en segundos.
- `ct_contactos`: Número de contactos realizados durante la campaña.
- `adherencia_inversion`: **Variable Objetivo** que indica si el cliente invirtió o no (sí/no).

### Desafío (Churn)
Para el desafío, se utiliza el conjunto de datos `churn.csv`, que contiene información de clientes de un banco y su propensión a abandonar el servicio. La variable objetivo es `churn` (1 si el cliente abandonó, 0 si no).

## Metodología
El proyecto sigue una metodología estructurada para el desarrollo de modelos de Machine Learning:

### Análisis Exploratorio de Datos (EDA)
Se realizó un análisis exhaustivo para entender la distribución de las variables y detectar posibles inconsistencias o patrones. Se utilizaron gráficos de barras (`plotly.express.histogram`) para variables categóricas (como `adherencia_inversion`, `estado_civil`, `escolaridad`, `default`, `prestatario`) y gráficos de caja (`plotly.express.box`) para variables numéricas (como `edad`, `saldo`, `ultimo_contacto`, `ct_contactos`).

### Transformación de Datos
Dado que los algoritmos de Machine Learning requieren datos numéricos, las variables categóricas se transformaron:
- **Variables Explicativas (X):** Se aplicó `OneHotEncoder` de Scikit-learn a las columnas categóricas (`estado_civil`, `escolaridad`, `default`, `prestatario`) para convertirlas en un formato numérico binario. Se usó `drop='if_binary'` para evitar la multicolinealidad en variables con solo dos categorías.
- **Variable de Respuesta (y):** Se utilizó `LabelEncoder` de Scikit-learn para transformar la variable objetivo `adherencia_inversion` (y `churn` en el desafío) a un formato numérico (0 y 1).

### División de Datos
Los datos se dividieron en conjuntos de entrenamiento y prueba (`X_train`, `X_test`, `y_train`, `y_test`) utilizando `train_test_split` con una proporción estratificada (`stratify=y`) para asegurar una representación equitativa de las clases en ambos conjuntos.

### Entrenamiento y Evaluación de Modelos
Se entrenaron y evaluaron varios modelos de clasificación utilizando los datos preparados.

### Selección y Serialización de Modelos
Los modelos entrenados, junto con los transformadores (`OneHotEncoder`, `MinMaxScaler`), se serializaron utilizando la biblioteca `pickle` para su posterior uso en producción.

## Modelos Empleados
Se utilizaron los siguientes algoritmos de clasificación:
- **Modelo de Referencia (Baseline):** `DummyClassifier` (estrategia por defecto, predice la clase mayoritaria). Sirve como punto de comparación.
- **Árbol de Decisión:** `DecisionTreeClassifier`. Se ajustó el hiperparámetro `max_depth` para controlar el sobreajuste.
- **K-Nearest Neighbors (KNN):** `KNeighborsClassifier`. Requiere la normalización de datos debido a su dependencia de la distancia euclidiana.

## Resultados Clave

Para el conjunto de datos de `marketing_inversiones`:
- **DummyClassifier:** Exactitud de aproximadamente 0.6025 (60.25%).
- **DecisionTreeClassifier (sin `max_depth`):** Exactitud de 0.6593 en prueba y 1.0 en entrenamiento, indicando sobreajuste.
- **DecisionTreeClassifier (con `max_depth=3`):** Exactitud de **0.7256 (72.56%)** en prueba y 0.7361 en entrenamiento. Este fue el modelo campeón para este conjunto de datos.
- **KNeighborsClassifier (datos normalizados):** Exactitud de 0.7003 (70.03%).

El **Árbol de Decisión con `max_depth=3`** fue el modelo con mejor rendimiento en el conjunto principal.

## Uso de los Modelos
Los modelos serializados permiten predecir nuevos datos. Por ejemplo, para predecir un nuevo registro de cliente:
1. Cargar el `one_hot` encoder y el `model_tree` campeón usando `pickle.load()`.
2. Transformar los nuevos datos con el `one_hot` encoder.
3. Realizar la predicción con `model_tree.predict()`.

Ejemplo de predicción para un nuevo cliente:
```python
nuevo_dato = pd.DataFrame({
    'edad': [45],
    'estado_civil':['soltero (a)'],
    'escolaridad':['superior'],
    'default': ['no'],
    'saldo': [23040],
    'prestatario': ['no'],
    'ultimo_contacto': [800],
    'ct_contactos': [4]
})

one_hot_loaded = pd.read_pickle('/content/modelo_onehotencoder.pkl')
model_tree_loaded = pd.read_pickle('/content/modelo_champion.pkl')

nuevo_dato_transformed = one_hot_loaded.transform(nuevo_dato)
prediction = model_tree_loaded.predict(nuevo_dato_transformed)
# El resultado array([1]) indica 'sí' (adherencia a la inversión)
```

## Sección de Desafío: Predicción de Abandono (Churn)

### Descripción del Desafío
El desafío consistió en predecir el 'churn' (abandono de clientes) utilizando un nuevo conjunto de datos. El churn es una métrica crucial que indica clientes que cancelan el servicio en un período determinado.

### Pasos Realizados en el Desafío
1.  **Carga de Datos:** Se cargó el archivo `churn.csv`.
2.  **Limpieza Inicial:** Se eliminó la columna `id_cliente` por no ser relevante para el modelo.
3.  **Análisis Exploratorio:** Se realizaron gráficos de histograma para la variable objetivo `churn` y otras variables categóricas (`pais`, `sexo_biologico`, `miembro_activo`, `tiene_tarjeta_credito`), agrupando por `churn`. También se usaron boxplots para variables numéricas (`score_credito`, `edad`, `años_de_cliente`, `saldo`, `servicios_adquiridos`, `salario_estimado`).
4.  **Separación X e y:** Se definieron las variables explicativas `X` (todas excepto `churn`) y la variable objetivo `y` (`churn`).
5.  **Transformación de Variables Categóricas:** Se aplicó `OneHotEncoder` a `sexo_biologico`, `pais`, `miembro_activo` y `tiene_tarjeta_credito` en `X`, y `LabelEncoder` a `y`.
6.  **División de Datos:** Se dividió el conjunto de datos en entrenamiento y prueba de forma estratificada.
7.  **Entrenamiento y Evaluación de Modelos:**
    *   **DummyClassifier:** Se entrenó un modelo baseline.
    *   **DecisionTreeClassifier:** Se entrenó con `max_depth=4` para evitar el sobreajuste y se visualizó el árbol.
    *   **MinMaxScaler:** Se normalizaron los datos de entrenamiento y prueba para KNN.
    *   **KNeighborsClassifier:** Se entrenó con los datos normalizados.
8.  **Serialización:** Se guardaron el `one_hot` encoder y el modelo campeón (`DecisionTreeClassifier`) en archivos `.pkl`.

### Resultados Clave del Desafío
- **DummyClassifier:** Exactitud de 0.7964 (79.64%).
- **DecisionTreeClassifier (max_depth=4):** Exactitud de **0.8548 (85.48%)** en prueba y 0.8505 en entrenamiento. Este fue el modelo campeón para el conjunto de datos de churn.
- **KNeighborsClassifier (datos normalizados):** Exactitud de 0.8208 (82.08%).

El **Árbol de Decisión con `max_depth=4`** fue el modelo con mejor rendimiento para la predicción de churn.

Ejemplo de predicción para un nuevo cliente de churn:
```python
nuevo_dato_churn = pd.DataFrame({
    'score_credito': [850],
    'pais':['Francia'],
    'sexo_biologico':['Hombre'],
    'edad': [27],
    'años_de_cliente': [3],
    'saldo': [56000],
    'servicios_adquiridos': [1],
    'tiene_tarjeta_credito': [1],
    'miembro_activo': [1],
    'salario_estimado': [85270.00]
})

one_hot_churn_loaded = pd.read_pickle('/content/modelo_onehotencoder_desafio.pkl')
model_tree_churn_loaded = pd.read_pickle('/content/modelo_champion_desafio.pkl')

nuevo_dato_churn_transformed = one_hot_churn_loaded.transform(nuevo_dato_churn)
prediction_churn = model_tree_churn_loaded.predict(nuevo_dato_churn_transformed)
# El resultado array([0]) indica que el cliente 'no' dejará de utilizar los servicios.
```
