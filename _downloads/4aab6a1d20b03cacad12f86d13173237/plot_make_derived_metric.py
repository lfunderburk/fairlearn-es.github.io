# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""
=============================
Obteniendo métricas derivadas
=============================
"""
# %%
# Este notebook demuestra el uso de la función :func:`fairlearn.metrics.make_derived_metric`.
# Muchos algoritmos de aprendizaje automático de orden superior (como los sintonizadores de hiperparámetros) hacen uso
# de métricas escalares al decidir cómo proceder.
# Mientras que :class:`fairlearn.metrics.MetricFrame` tiene la capacidad de producir tales
# escalares a través de sus funciones de agregación, su API no se ajusta a la que normalmente
# esperado por estos algoritmos.
# La función :func:`~ fairlearn.metrics.make_derived_metric` existe para solucionar este problema.
#
# Obteniendo los datos
# ====================
#
# * Esta sección se puede omitir. Simplemente crea un conjunto de datos para
# fines ilustrativos *
#
# Utilizaremos el conocido conjunto de datos UCI 'Adultos' como base de esta
# demostración. Esto no es para un escenario de préstamos, pero consideraremos
# como uno para los propósitos de este ejemplo. Usaremos las
# columnas 'raza' y 'sexo' (recortando la primera a tres valores únicos),
# y fabricaremos bandas de puntaje crediticio y tamaños de préstamos a partir de otras columnas.
# Comenzamos con algunas declaraciones de `importación`:

import functools
import numpy as np

import sklearn.metrics as skm
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_selector as selector
from sklearn.pipeline import Pipeline
from fairlearn.metrics import MetricFrame, make_derived_metric
from fairlearn.metrics import accuracy_score_group_min

# %%
# A continuación, importamos los datos, eliminando las filas a las que les faltan datos:

data = fetch_openml(data_id=1590, as_frame=True)
X_raw = data.data
y = (data.target == ">50K") * 1
A = X_raw[["race", "sex"]]

# %%
# Ahora vamos a preprocesar los datos. Antes de aplicar cualquier transformación,
# primero dividimos los datos en conjuntos de prueba y de entrenamiento. Todas las transformaciones
# que usemos se aplicarán en el conjunto datos de entrenamiento y luego aplicada al conjunto
# de prueba. Esto asegura que los datos no se filtren entre los dos conjuntos (esto es
# un serio pero sutil
# `problema en el aprendizaje automático <https://en.wikipedia.org/wiki/Leakage_(machine_learning)>`_).
# Primero dividimos los datos:

(X_train, X_test, y_train, y_test, A_train, A_test) = train_test_split(
    X_raw, y, A, test_size=0.3, random_state=12345, stratify=y
)

# Asegúrese de que los índices estén alineados entre X, y, A
# en las Series de conjuntos de prueba y entrenamiento.

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)
A_train = A_train.reset_index(drop=True)
A_test = A_test.reset_index(drop=True)

# %%
# A continuación, construimos dos objetos :class:`~ sklearn.pipeline.Pipeline`
# para procesar las columnas, una para datos numéricos y la otra
# para datos categóricos. Ambos imputan valores perdidos; la diferencia
# es si los datos están escalados (columnas numéricas) o tienen
# codificación one-hot (columnas categóricas). La imputación de datos no presentes
# generalmente deben hacerse con cuidado, ya que pueden
# introducirse prejuicios. Por supuesto, eliminar filas con
# los datos no presentes también puede causar problemas, si subgrupos de datos
# tienen datos de peor calidad.

numeric_transformer = Pipeline(
    steps=[
        ("impute", SimpleImputer()),
        ("scaler", StandardScaler()),
    ]
)
categorical_transformer = Pipeline(
    [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, selector(dtype_exclude="category")),
        ("cat", categorical_transformer, selector(dtype_include="category")),
    ]
)

# %%
# Con nuestro preprocesador definido, ahora podemos construir un
# nueva pipeline que incluye un Estimador:

unmitigated_predictor = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(solver="liblinear", fit_intercept=True),
        ),
    ]
)

# %%
# Con la pipeline completamente definida, primero podemos entrenarla
# con los datos de entrenamiento y luego generar predicciones
# de los datos de prueba.

unmitigated_predictor.fit(X_train, y_train)
y_pred = unmitigated_predictor.predict(X_test)

# %%
# Creando una métrica derivada
# ============================
#
# Suponga que nuestra métrica clave es la puntuación de precisión y lo que más nos interesa es
# asegurándose de que exceda algún límite ("threshold") para todos los subgrupos
# Podríamos usar :class:`~ fairlearn.metrics.MetricFrame` como
# sigue:

acc_frame = MetricFrame(
    metrics=skm.accuracy_score,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=A_test["sex"]
)
print("Minimum accuracy_score: ", acc_frame.group_min())

# %%
# Podemos crear una función para realizar esto en una sola llamada
# usando :func:`~ fairlearn.metrics.make_derived_metric`.
# Esto toma los siguientes argumentos (que siempre deben ser
# suministrados como argumentos de palabra clave):
#
# - :code:`metric =`, la función métrica base
# - :code:`transform =`, el nombre de la transformación de agregación
#  para realizar. Para esta demostración,
#  esto sería :code:`'group_min'`
# - :code:`sample_param_names =`, una lista de nombres de parámetros
# que debe tratarse como muestra
# parámetros. Esto es opcional y por defecto es
# :code:`['sample_weight']` que es apropiado para muchos
# métricas en `scikit-learn`.
#
# El resultado es una nueva función con la misma firma que el
# métrica base, que acepta dos argumentos adicionales:
#
# - :code:`sensitive_features =` para especificar las características sensibles
# que definen los subgrupos
# - :code:`método =` para ajustar cómo la transformación de agregación
# opera. Esto corresponde al mismo argumento en
# :meth: `fairlearn.metrics.MetricFrame.difference` y
#: meth: `fairlearn.metrics.MetricFrame.ratio`
#
# Para el caso actual, no necesitamos el argumento :code:`method =`,
# ya que estamos tomando el valor mínimo.

my_acc = make_derived_metric(metric=skm.accuracy_score, transform="group_min")
my_acc_min = my_acc(y_test, y_pred, sensitive_features=A_test["sex"])
print("Minimum accuracy_score: ", my_acc_min)

# %%
# Para mostrar que la función resultante también funciona con ponderaciones de muestra:

random_weights = np.random.rand(len(y_test))

acc_frame_sw = MetricFrame(
    metrics=skm.accuracy_score,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=A_test["sex"],
    sample_params={"sample_weight": random_weights},
)

from_frame = acc_frame_sw.group_min()
from_func = my_acc(
    y_test,
    y_pred,
    sensitive_features=A_test["sex"],
    sample_weight=random_weights,
)

print("From MetricFrame:", from_frame)
print("From function   :", from_func)

# %%
# La función devuelta también puede manejar parámetros que no son parámetros muestra.
# Considere :func:`sklearn.metrics.fbeta_score`, que
# tiene un argumento requerido :code:`beta =` (y supongamos que esta vez
# lo que más nos interesa es la diferencia máxima con el valor total).
# Primero evaluamos esto con :class:`fairlearn.metrics.MetricFrame`:

fbeta_03 = functools.partial(skm.fbeta_score, beta=0.3)
fbeta_03.__name__ = "fbeta_score__beta_0.3"

beta_frame = MetricFrame(
    metrics=fbeta_03,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=A_test["sex"],
    sample_params={"sample_weight": random_weights},
)
beta_from_frame = beta_frame.difference(method="to_overall")

print("From frame:", beta_from_frame)

# %%
# Y a continuación, creamos una función para evaluar lo mismo. Tenga en cuenta que
# no necesitamos usar la función :func:`functools.partial` para enlazar el argumento
# :code:`beta =`:

beta_func = make_derived_metric(metric=skm.fbeta_score, transform="difference")

beta_from_func = beta_func(
    y_test,
    y_pred,
    sensitive_features=A_test["sex"],
    beta=0.3,
    sample_weight=random_weights,
    method="to_overall",
)

print("From function:", beta_from_func)


# %%
# Métricas pregeneradas
# =====================
#
# Proporcionamos una serie de métricas pregeneradas para cubrir
# casos de uso comunes. Por ejemplo, proporcionamos la función
# :code:`precision_score_group_min ()` para
# encontrar la calificación de precisión mínima:


from_myacc = my_acc(y_test, y_pred, sensitive_features=A_test["race"])

from_pregen = accuracy_score_group_min(
    y_test, y_pred, sensitive_features=A_test["race"]
)

print("From my function :", from_myacc)
print("From pregenerated:", from_pregen)
assert from_myacc == from_pregen
