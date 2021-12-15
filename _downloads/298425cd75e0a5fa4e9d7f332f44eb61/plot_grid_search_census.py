# %%
# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""
=============================
GridSearch con datos de censo
=============================
"""
# %%
# Este notebook muestra cómo usar Fairlearn para generar predictores para el conjunto de datos del censo.
# Este conjunto de datos es un problema de clasificación: dado un rango de datos sobre 32.000 personas,
# predecir si sus ingresos anuales están por encima o por debajo de cincuenta mil dólares por año.
#
# Para los propósitos de este notebook, trataremos esto como un problema de decisión de préstamo.
# Fingiremos que la etiqueta indica si cada individuo pagó o no un préstamo en
# el pasado.
# Usaremos los datos para entrenar un predictor para predecir si individuos no vistos previamente
# pagará un préstamo o no.
# El supuesto es que las predicciones del modelo se utilizan para decidir si un individuo
# se le debe ofrecer un préstamo.
#
# Primero entrenaremos a un predictor inconsciente de la equidad y demostraremos que conduce a
# decisiones bajo una noción específica de equidad llamada *paridad demográfica*.
# Luego mitigamos la injusticia aplicando el :código:algoritmo `GridSearch` del
# Paquete Fairlearn.

# %%
# Cargar y preprocesar el conjunto de datos
# --------------------------------
# Descargamos el conjunto de datos usando la función `fetch_adult` en `fairlearn.datasets`.
# Empezamos importando los distintos módulos que vamos a utilizar:
#

from sklearn.model_selection import train_test_split
from fairlearn.reductions import GridSearch
from fairlearn.reductions import DemographicParity, ErrorRate
from fairlearn.metrics import MetricFrame, selection_rate, count
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics as skm
import pandas as pd
import matplotlib.pyplot as plt

# %%
# We can now load and inspect the data by using the `fairlearn.datasets` module:

from sklearn.datasets import fetch_openml

data = fetch_openml(data_id=1590, as_frame=True)
X_raw = data.data
Y = (data.target == '>50K') * 1

# %%
# Vamos a tratar el sexo de cada individuo como un sensible
# característica (donde 0 indica mujer y 1 indica hombre), y en
# En este caso particular, vamos a separar esta función y la eliminaremos.
# de los datos principales.
# Luego realizamos algunos pasos estándar de preprocesamiento de datos para convertir el
# datos en un formato adecuado para los algoritmos aprendizaje automático

A = X_raw["sex"]
X = X_raw.drop(labels=['sex'], axis=1)
X = pd.get_dummies(X)

sc = StandardScaler()
X_scaled = sc.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

le = LabelEncoder()
Y = le.fit_transform(Y)

# %%
# Finalmente, dividimos los datos en conjuntos de entrenamiento y prueba:

X_train, X_test, Y_train, Y_test, A_train, A_test = train_test_split(X_scaled,
                                                                     Y,
                                                                     A,
                                                                     test_size=0.2,
                                                                     random_state=0,
                                                                     stratify=Y)

# Work around indexing bug
X_train = X_train.reset_index(drop=True)
A_train = A_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
A_test = A_test.reset_index(drop=True)

# %%
# Entrenando a un predictor inconsciente de la equidad
# -------------------------------------
#
# Para mostrar el efecto de Fairlearn, primero entrenaremos un predictor  de aprendizaje automático estándar
# que no incorpora justicia.
# Para velocidad de demostración, usamos
# :class:`sklearn.linear_model.LogisticRegression` class:

unmitigated_predictor = LogisticRegression(solver='liblinear', fit_intercept=True)

unmitigated_predictor.fit(X_train, Y_train)

# %%
# Podemos comenzar a evaluar la equidad del predictor usando el `MetricFrame`:
metric_frame = MetricFrame(metrics={"accuracy": skm.accuracy_score,
                                    "selection_rate": selection_rate,
                                    "count": count},
                           sensitive_features=A_test,
                           y_true=Y_test,
                           y_pred=unmitigated_predictor.predict(X_test))
print(metric_frame.overall)
print(metric_frame.by_group)
metric_frame.by_group.plot.bar(
    subplots=True, layout=[3, 1], legend=False, figsize=[12, 8],
    title='Accuracy and selection rate by group')

# %%
# Al observar la disparidad en la precisión, vemos que los hombres tienen un error
# aproximadamente tres veces mayor que las mujeres.
# Más interesante es la disparidad de oportunidades: a los hombres se les ofrecen préstamos
# tres veces la tasa de mujeres.
#
# A pesar de que eliminamos la función de los datos de entrenamiento, nuestro
# predictor aún discrimina según el sexo.
# Esto demuestra que simplemente ignorar una característica sensible al instalar un
# predictor rara vez elimina la injusticia.
# En general, habrá suficientes otras características correlacionadas con la eliminación
# característica para generar un impacto dispar.

# %%
# Mitigación con GridSearch
# --------------------------
#
# La clase :class:`fairlearn.reductions.GridSearch` implementa una versión simplificada de
# reducción exponencial del gradiente de `Agarwal et al. 2018 <https://arxiv.org/abs/1803.02453>`_.
# El usuario proporciona un estimador de aprendizaje automático estándar, que se trata como una caja negra.
# `GridSearch` funciona generando una secuencia de reetiquetas y reponderaciones, y
# entrena un predictor para cada uno.
#
# Para este ejemplo, especificamos la paridad demográfica (en la característica sensible del sexo) como
# la métrica de equidad.
# La paridad demográfica requiere que se ofrezca la oportunidad a las personas (estén aprobadas
# para un préstamo en este ejemplo) independientemente de la membresía en la clase sensible (es decir, a mujeres
# y a hombres se les debe ofrecer préstamos a la misma tasa).
# Estamos usando esta métrica por simplicidad; en general, la equidad adecuada
# métrica no será obvia.

sweep = GridSearch(LogisticRegression(solver='liblinear', fit_intercept=True),
                   constraints=DemographicParity(),
                   grid_size=71)

# %%
# Nuestros algoritmos proporcionan métodos :code:`fit ()` y :code:`predict()`,
# por lo que se comportan de manera similar
# a otros paquetes ML en Python.
# Sin embargo, tenemos que especificar dos argumentos adicionales para: código: `fit ()` - la columna de sensibles
# etiquetas de características, y también la cantidad de predictores que se generarán en nuestro barrido.
#
# Después de que se complete :code:`fit ()`, extraemos el conjunto completo de predictores del objeto
# :class:`fairlearn.reductions.GridSearch`.

sweep.fit(X_train, Y_train,
          sensitive_features=A_train)

predictors = sweep.predictors_

# %%
# Podríamos trazar métricas de rendimiento y equidad de estos predictores ahora.
# Sin embargo, la gráfica sería algo confusa debido a la cantidad de modelos.
# En este caso, vamos a eliminar los predictores que están dominados en el
# espacio de error-disparidad por otros del barrido (tenga en cuenta que la disparidad solo será
# calculado para la característica sensible; otras características potencialmente sensibles
# no ser mitigado).
# En general, es posible que no desee hacer esto, ya que puede haber otras consideraciones
# más allá de la optimización estricta del error y la disparidad (de la característica sensible dada).

errors, disparities = [], []
for m in predictors:
    def classifier(X): return m.predict(X)


    error = ErrorRate()
    error.load_data(X_train, pd.Series(Y_train), sensitive_features=A_train)
    disparity = DemographicParity()
    disparity.load_data(X_train, pd.Series(Y_train), sensitive_features=A_train)

    errors.append(error.gamma(classifier)[0])
    disparities.append(disparity.gamma(classifier).max())

all_results = pd.DataFrame({"predictor": predictors, "error": errors, "disparity": disparities})

non_dominated = []
for row in all_results.itertuples():
    errors_for_lower_or_eq_disparity = all_results["error"][all_results["disparity"] <= row.disparity]
    if row.error <= errors_for_lower_or_eq_disparity.min():
        non_dominated.append(row.predictor)

# %%
# Finalmente, podemos evaluar los modelos dominantes junto con el modelo no mitigado.

predictions = {"unmitigated": unmitigated_predictor.predict(X_test)}
metric_frames = {"unmitigated": metric_frame}
for i in range(len(non_dominated)):
    key = "dominant_model_{0}".format(i)
    predictions[key] = non_dominated[i].predict(X_test)

    metric_frames[key] = MetricFrame(metrics={"accuracy": skm.accuracy_score,
                                              "selection_rate": selection_rate,
                                              "count": count},
                                     sensitive_features=A_test,
                                     y_true=Y_test,
                                     y_pred=predictions[key])


x = [metric_frame.overall['accuracy'] for metric_frame in metric_frames.values()]
y = [metric_frame.difference()['selection_rate'] for metric_frame in metric_frames.values()]
keys = list(metric_frames.keys())
plt.scatter(x, y)
for i in range(len(x)):
    plt.annotate(keys[i], (x[i] + 0.0003, y[i]))
plt.xlabel("accuracy")
plt.ylabel("selection rate difference")

# %%
# Vemos la formación de un frente de Pareto: el conjunto de predictores que representan compensaciones óptimas
# entre precisión y disparidad en las predicciones.
# En el caso ideal, tendríamos un predictor en (1,0) - perfectamente preciso y sin
# cualquier injusticia bajo paridad demográfica (con respecto a la característica sensible "sexo").
# El frente de Pareto representa lo más cerca que podemos llegar a este ideal según nuestros datos y
# elección de estimador.
# Tenga en cuenta el rango de los ejes: el eje de disparidad cubre más valores que la precisión,
# para que podamos reducir la disparidad sustancialmente por una pequeña pérdida de precisión.
# En un ejemplo real, elegiríamos el modelo que representara la mejor compensación
# entre precisión y disparidad dadas las limitaciones comerciales relevantes.
