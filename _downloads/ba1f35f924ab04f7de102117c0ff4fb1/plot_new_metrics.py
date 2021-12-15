# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""
================================
Métricas con múltiples funciones
================================
"""
# %%
# Este notebook muestra la nueva API para métricas, que admite
# múltiples características sensibles y condicionales. Este ejemplo no
# contiene una discusión adecuada sobre cómo la justicia se relaciona con el conjunto de datos
# utilizado, aunque resalta problemas que los usuarios pueden querer
# considere al analizar sus conjuntos de datos.
#
# Vamos a considerar un escenario de préstamo de crédito, suponiendo que tengamos
# un modelo que predice si un cliente en particular
# va a reembolsar un préstamo. Esto podría utilizarse como base para decidir si
# o no ofrecer un préstamo a ese cliente. Con métricas tradicionales,
# evaluaríamos el modelo usando:
#
# - Los valores 'verdaderos' del conjunto de prueba
# - Las predicciones del modelo del conjunto de prueba
#
# Nuestras métricas de equidad calculan estadísticas de equidad basadas en grupos.
# Para usar estos, también necesitamos columnas categóricas del conjunto de prueba.
# Para este ejemplo, incluiremos:
#
# - El sexo de cada individuo (dos valores únicos)
# - La raza de cada individuo (tres valores únicos)
# - La categoría de puntaje crediticio de cada individuo (tres valores únicos)
# - Si el préstamo se considera 'grande' o 'pequeño'
#
# El sexo y la raza de una persona no deben afectar la decisión de un préstamo,
# pero sería legítimo considerar el puntaje crediticio de una persona
# y el tamaño relativo del préstamo que deseaban.
#
# Un escenario real será más complicado, pero esto servirá para
# ilustrar el uso de las nuevas métricas.
#
# Obteniendo los datos
# ====================
#
# *Esta sección se puede omitir. Simplemente crea un conjunto de datos para
# fines ilustrativos*
#
# Utilizaremos el conocido conjunto de datos UCI 'Adult' como base de este
# demostración. Esto no es para un escenario de préstamos, pero consideraremos
# como uno para los propósitos de este ejemplo. Usaremos el existente
# columnas 'raza' y 'sexo' (recortando la primera a tres valores únicos),
# y fabrique bandas de puntaje crediticio y tamaños de préstamos a partir de otras columnas.
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

from fairlearn.metrics import MetricFrame
from fairlearn.metrics import selection_rate, count


# %%
# A continuación, importamos los datos:

data = fetch_openml(data_id=1590, as_frame=True)
X_raw = data.data
y = (data.target == ">50K") * 1

# %%
# Para mayor claridad, consolidamos la columna 'raza' para tener
# tres valores únicos:


def race_transform(input_str):
    """Reduce values to White, Black and Other."""
    result = "Other"
    if input_str == "White" or input_str == "Black":
        result = input_str
    return result


X_raw["race"] = (
    X_raw["race"].map(race_transform).fillna("Other").astype("category")
)
print(np.unique(X_raw["race"]))

# %%
# Después, fabricamos las columnas para la banda de calificación crediticia y
# tamaño del préstamo solicitado. Estos están hipotéticos, y no
# parte del conjunto de datos real de alguna manera. Son simplemente para
# fines ilustrativos.


def marriage_transform(m_s_string):
    """Perform some simple manipulations."""
    result = "Low"
    if m_s_string.startswith("Married"):
        result = "Medium"
    elif m_s_string.startswith("Widowed"):
        result = "High"
    return result


def occupation_transform(occ_string):
    """Perform some simple manipulations."""
    result = "Small"
    if occ_string.startswith("Machine"):
        result = "Large"
    return result


col_credit = X_raw["marital-status"].map(marriage_transform).fillna("Low")
col_credit.name = "Credit Score"
col_loan_size = X_raw["occupation"].map(occupation_transform).fillna("Small")
col_loan_size.name = "Loan Size"

A = X_raw[["race", "sex"]]
A["Credit Score"] = col_credit
A["Loan Size"] = col_loan_size
A

# %%
# Ahora que hemos importado nuestro conjunto de datos y fabricado algunas funciones,
# podemos realizar un procesamiento más convencional. Para evitar el problema de
# `fuga de datos <https://en.wikipedia.org/wiki/Leakage_ (machine_learning)>`_,
# necesitamos dividir los datos en conjuntos de prueba y entrenamiento antes de aplicar
# cualquier transformación o escala:

(X_train, X_test, y_train, y_test, A_train, A_test) = train_test_split(
    X_raw, y, A, test_size=0.3, random_state=54321, stratify=y
)

# Asegúrese de que los índices estén alineados entre X, y, A,
# después de seleccionar y dividir el marco de datos en Series.

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
# es si los datos están escalados (columnas numéricas) o
# tienen codificación one-hot (columnas categóricas). Imputación de
# valores faltantes generalmente deben hacerse con cuidado, ya que esto podría
# introducir sesgos potencialmente. Por supuesto, eliminar filas con
# los datos faltantes también puede causar problemas, si subgrupos particulares
# tienen datos de peor calidad.

numeric_transformer = Pipeline(
    steps=[("impute", SimpleImputer()), ("scaler", StandardScaler())]
)
categorical_transformer = Pipeline(
    [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, selector(dtype_exclude="category")),
        ("cat", categorical_transformer, selector(dtype_include="category"))
    ]
)

# %%
# Con nuestro preprocesador definido, ahora podemos construir un
# nueva canalización que incluye un Estimador:

unmitigated_predictor = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(solver="liblinear", fit_intercept=True)
        )
    ]
)

# %%
# Con la pipeline (tubería) completamente definida, primero podemos entrenarla
# con los datos de entrenamiento y luego generar predicciones
# utilizando los datos de prueba.

unmitigated_predictor.fit(X_train, y_train)
y_pred = unmitigated_predictor.predict(X_test)


# %%
# Analizando el modelo con métricas
# ================================
#
# Después del formateo de datos y entrenamiento de modelos, tenemos lo siguiente
# de nuestro conjunto de prueba:
#
# - Un vector de valores verdaderos llamado ``y_test``
# - Un vector de predicciones del modelo llamado ``y_pred``
# - Un DataFrame (tabla de datos) con características categóricas relevantes para la equidad llamado ``A_test``
#
# Si fuésemos a utilizar un análisis de modelo tradicional, utilizaríamos algunas métricas
# que evalúan el conjunto de datos completo. Supongamos que en este caso,
# las métricas relevantes son :func:`fairlearn.metrics.selection_rate` y
# :func:`sklearn.metrics.fbeta_score` (con
# `beta = 0.6``).
# Podemos evaluar estas métricas directamente:

print("Selection Rate:", selection_rate(y_test, y_pred))
print("fbeta:", skm.fbeta_score(y_test, y_pred, beta=0.6))

# %%
# Sabemos que hay características sensibles en nuestros datos y queremos
# asegurarnos de no dañar a las personas debido a su membresía en
# estos grupos. Para este propósito, Fairlearn proporciona la clase
# :clase:`fairlearn.metrics.MetricFrame`. Construyamos una instancia de esta clase y luego miremos
# sus capacidades:

fbeta_06 = functools.partial(skm.fbeta_score, beta=0.6, zero_division=1)

metric_fns = {
    "selection_rate": selection_rate,
    "fbeta_06": fbeta_06,
    "count": count
}

grouped_on_sex = MetricFrame(
    metrics=metric_fns,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=A_test["sex"]
)

# %%
# La clase :class:`fairlearn.metrics.MetricFrame` requiere un
# mínimo de cuatro argumentos:
#
# 1. Las funciones métricas que se evaluarán
# 2. Los valores verdaderos
# 3. Los valores predichos
# 4. Los valores de las características sensibles
#
# Todos estos se pasan como argumentos al constructor. Si más de una métrica
# se requiere(como en este caso), entonces debemos
# proporcionarlos en un diccionario.
#
# Las métricas deben tener una firma ``fn (y_true, y_pred)``,
# entonces tenemos que usar :func:`functools.partial` en ``fbeta_score()`` para
# proporcionar ``beta = 0.6`` (mostraremos cómo pasar una lista con
# argumentos como ponderaciones de muestra en breve).
#
# Ahora echaremos un vistazo más de cerca a :class:`fairlearn.metrics.MetricFrame`.
# Primero, está la propiedad ``overall``, que contiene
# las métricas evaluadas en el conjunto de datos completo. Vemos que esto contiene el
# mismos valores calculados anteriormente:

assert grouped_on_sex.overall["selection_rate"] == selection_rate(
    y_test, y_pred
)
assert grouped_on_sex.overall["fbeta_06"] == skm.fbeta_score(
    y_test, y_pred, beta=0.6
)
print(grouped_on_sex.overall)

# %%
# La otra propiedad en :class:`fairlearn.metrics.MetricFrame`
# es ``by_group``, el cual contiene las métricas evaluadas en cada subgrupo definido
# por las categorías en el argumento ``sensitive_features =``. Tenga en cuenta que
# :func:`fairlearn.metrics.count` se puede usar para mostrar el número de
# puntos de datos en cada subgrupo. En este caso, tenemos resultados para hombres y mujeres:

grouped_on_sex.by_group

# %%
# Podemos ver inmediatamente una disparidad sustancial en la tasa de selección entre
# masculinos y femeninos.
#
# También podemos crear otro objeto :class:`fairlearn.metrics.MetricFrame`
# usando la raza como característica sensible:

grouped_on_race = MetricFrame(
    metrics=metric_fns,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=A_test["race"]
)

# %%
# La propiedad ``overall`` no cambia:

assert (grouped_on_sex.overall == grouped_on_race.overall).all()

# %%
# La propiedad ``by_group`` ahora contiene las métricas evaluadas según la columna 'raza':

grouped_on_race.by_group

# %%
# Vemos que también existe una disparidad significativa en las tasas de selección cuando
# agrupación por raza.

# %%
# Muestras de pesos y otras matrices
# ----------------------------------
#
# Observamos anteriormente que las funciones métricas subyacentes pasaron al
# constructor :class:`fairlearn.metrics.MetricFrame` debe ser de
# la forma ``fn (y_true, y_pred)`` - no admitimos argumentos escalares
# como ``pos_label =`` o ``beta =`` en el constructor. Dichos
# argumentos deben estar vinculados a una nueva función usando
# :func:`functools.partial`, junto con el resultado. Sin embargo, Fairlearn también apoya
# argumentos que tienen solo un elemento por cada muestra, con una matriz
# de pesos de muestra es el ejemplo más común. Estos están divididos
# en subgrupos junto con ``y_true`` y ``y_pred``, y se pasan
# a la métrica subyacente.
#
# Para usar estos argumentos, pasamos en un diccionario como `` sample_params =``
# argumento del constructor. Generemos algunos pesos aleatorios y
# pásales estos:

random_weights = np.random.rand(len(y_test))

example_sample_params = {
    "selection_rate": {"sample_weight": random_weights},
    "fbeta_06": {"sample_weight": random_weights}
}


grouped_with_weights = MetricFrame(
    metrics=metric_fns,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=A_test["sex"],
    sample_params=example_sample_params
)

# %%
# Podemos inspeccionar los valores generales y verificar que sean los esperados:

assert grouped_with_weights.overall["selection_rate"] == selection_rate(
    y_test, y_pred, sample_weight=random_weights
)
assert grouped_with_weights.overall["fbeta_06"] == skm.fbeta_score(
    y_test, y_pred, beta=0.6, sample_weight=random_weights
)
print(grouped_with_weights.overall)

# %%
# También podemos ver el efecto sobre la métrica que se evalúa en los subgrupos:

grouped_with_weights.by_group

# %%
# Cuantificación de disparidades
# ==============================
#
# Ahora sabemos que nuestro modelo está seleccionando individuos que son mujeres mucho menos
# a menudo que los hombres. Hay un efecto similar cuando
# examinando los resultados por raza, y los negros son seleccionados con mucha menos frecuencia que
# blancos (y los clasificados como 'otros'). Sin embargo, hay muchos casos en los que
# presentar todos estos números a la vez no será útil (por ejemplo, un
# tablero de alto nivel que monitorea el desempeño del modelo). Fairlearn ofrece
# varios medios de agregar métricas en los subgrupos, de modo que las disparidades
# pueden cuantificarse fácilmente.
#
# La más simple de estas agregaciones es ``group_min()``, que informa el
# valor mínimo visto para un subgrupo para cada métrica subyacente (también proporcionamos
# ``group_max()``). Esto es
# útil si hay un mandato de que "ningún subgrupo debe tener un ``fbeta_score()``
# de menos de 0.6". Podemos evaluar los valores mínimos fácilmente:

grouped_on_race.group_min()

# %%
# Como se señaló anteriormente, las tasas de selección varían mucho según la raza y el sexo.
# Esto se puede cuantificar en términos de una diferencia entre el subgrupo con
# el valor más alto de la métrica y el subgrupo con el valor más bajo.
# Para esto, proporcionamos el método ``difference(method ='between_groups)``:

grouped_on_race.difference(method="between_groups")

# %%
# También podemos evaluar la diferencia relativa que corresponde al
# valor total de la métrica. En este caso tomamos el valor absoluto, de modo que el
# el resultado es siempre positivo:

grouped_on_race.difference(method="to_overall")

# %%
# Hay situaciones en las que conocer los radios de las métricas evaluadas en
# los subgrupos es más útil. Para ello tenemos el método ``ratio()``.
# Podemos tomar las relaciones entre los valores mínimo y máximo de cada métrica:

grouped_on_race.ratio(method="between_groups")

# %%
# También podemos calcular los radios relativos al valor general de cada
# métrica. De manera análoga a las diferencias, las proporciones están siempre en el rango
# :math: `[0,1]`:

grouped_on_race.ratio(method="to_overall")

# %%
# Intersección de características
# =================================
#
# Hasta ahora, solo hemos considerado una característica sensible a la vez,
# y ya hemos encontrado algunos problemas graves en nuestros datos de ejemplo.
# Sin embargo, a veces se pueden esconder problemas graves en las intersecciones de
# características. Por ejemplo, el
# `Proyecto Gender Shades <https://www.media.mit.edu/projects/gender-shades/overview/>`_
# descubrió que los algoritmos de reconocimiento facial funcionaban peor para los negros
# que los blancos, y también peor para las mujeres que para los hombres (a pesar de la alta
# puntuación de precisión). Además, el rendimiento en mujeres negras fue *terrible*.
# Podemos examinar las intersecciones de características sensibles pasando
# varias columnas para el constructor :class:`fairlearn.metrics.MetricFrame`:

grouped_on_race_and_sex = MetricFrame(
    metrics=metric_fns,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=A_test[["race", "sex"]]
)

# %%
# Los valores generales no han cambiado, pero la tabla ``by_group`` ahora
# muestra las intersecciones entre subgrupos:

assert (grouped_on_race_and_sex.overall == grouped_on_race.overall).all()
grouped_on_race_and_sex.by_group

# %%
# Las agregaciones aún se realizan en todos los subgrupos para cada métrica,
# para que cada uno continúe reduciéndose a un solo valor. Si miramos
# `` group_min()``, vemos que violamos el mandato que especificamos para
# `` fbeta_score()`` sugerido arriba (para mujeres con una raza de 'Otro'):

grouped_on_race_and_sex.group_min()

# %%
# Mirando el método ``ratio()``, vemos que la disparidad es peor
# (específicamente entre hombres blancos y mujeres negras, si revisamos
# la tabla ``by_group``):

grouped_on_race_and_sex.ratio(method="between_groups")

# %%
# Funciones de control
# ====================
#
# Hay otra forma en que podemos dividir nuestros datos. Tenemos (*completamente
# inventadas*) características para los puntajes crediticios de las personas (en tres rangos)
# y también el tamaño del préstamo solicitado (grande o pequeño). En el escenario de préstamo,
# es aceptable que las personas con puntajes crediticios altos
# sean seleccionadas con más frecuencia que las personas con puntajes crediticios bajos.
# Sin embargo, dentro de cada rango de puntaje crediticio, no queremos una disparidad
# entre (digamos) mujeres negras y hombres blancos. Para ejemplificar estos casos,
# tenemos el concepto de *funciones de control*.
#
# Las funciones de control son introducidas por el argumento ``control_features =``
# del objeto :class:`fairlearn.metrics.MetricFrame`:

cond_credit_score = MetricFrame(
    metrics=metric_fns,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=A_test[["race", "sex"]],
    control_features=A_test["Credit Score"]
)

# %%
# Esto tiene un efecto inmediato en la propiedad ``overall``. En lugar
# de tener un valor para cada métrica, ahora tenemos un valor para cada
# valor único de la función de control:

cond_credit_score.overall

# %%
# La propiedad ``by_group`` es demostrada de manera similar:
cond_credit_score.by_group

# %%
# Los agregados de datos también se evalúan una vez para cada grupo identificado
# por la función de control:

cond_credit_score.group_min()

# %%
# Y:
cond_credit_score.ratio(method="between_groups")

# %%
# En nuestros datos, vemos que tenemos una escasez de resultados positivos
# para aquellos que no blancos y que tienen altos ingresos, lo que afecta significativamente
# los agregados de datos.
#
# Podemos seguir agregando más funciones de control:
cond_both = MetricFrame(
    metrics=metric_fns,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=A_test[["race", "sex"]],
    control_features=A_test[["Loan Size", "Credit Score"]]
)

# %%
# La propiedad ``overall`` se desglosa en más valores:
cond_both.overall

# %%
# Al igual que la propiedad ``by_groups``, donde los valores ``NaN``
# indica que no había muestras en la celda:

cond_both.by_group

# %%
# Los agregados de datos se comportan de manera similar. A estas alturas, estamos teniendo problemas importantes
# con intersecciones poco pobladas. Consideremos:


def member_counts(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return len(y_true)


counts = MetricFrame(
    metrics=member_counts,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=A_test[["race", "sex"]],
    control_features=A_test[["Loan Size", "Credit Score"]]
)

counts.by_group

# %%
# Recordemos que ``NaN`` indica que no hubo individuos
# en una celda - ``member_counts()`` ni siquiera habrá sido llamado.

# %%
# Exportando desde MetricFrame
# ============================
#
# A veces, necesitamos extraer nuestros datos para usarlos en otras herramientas.
# Para esto, podemos usar el método :py:meth:`pandas.DataFrame.to_csv`,
# ya que :py:meth:`~fairlearn.metrics.MetricFrame.by_group`
# será de tipo :class:`pandas.DataFrame` (o en algunos casos, será
# de tipo :class:`pandas.Series`, pero tiene un método similar
# :py:meth:`~ pandas.Series.to_csv`):

csv_output = cond_credit_score.by_group.to_csv()
print(csv_output)

# %%
# El método :py:meth:`pandas.DataFrame.to_csv` tiene una gran cantidad de
# argumentos para controlar el CSV exportado. Por ejemplo, puede escribir
# directamente a un archivo CSV, en lugar de devolver una cadena de caracteres (como se
# mostró anteriormente).
#
# La propiedad :meth:`~ fairlearn.metrics.MetricFrame.overall` puede
# manejarse de manera similar, en los casos en que no sea un escalar.
