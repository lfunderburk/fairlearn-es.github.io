# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

"""
=================================================
MetricFrame: más allá de la clasificación binaria
=================================================
"""

# %%
# Este notebook contiene ejemplos de uso :class:`~ fairlearn.metrics.MetricFrame`
# para tareas que van más allá de la simple clasificación binaria.

import sklearn.metrics as skm
import functools
from fairlearn.metrics import MetricFrame

# %%
# Resultados multiclase y no escalares
# ====================================
#
# Supongamos que tenemos un problema multiclase, con etiquetas :math:`\ in {0, 1, 2}`,
# y que deseamos generar matrices de confusión para cada subgrupo
# identificado por la característica sensible :math:`\ in {a, b, c, d}`.
# Esto es apoyado fácilmente por
# :class:`~ fairlearn.metrics.MetricFrame`, que no requiere
# el resultado de una métrica para ser un escalar.
#
# Primero, generemos algunos datos de entrada aleatorios:

import numpy as np

rng = np.random.default_rng(seed=96132)

n_rows = 1000
n_classes = 3
n_sensitive_features = 4

y_true = rng.integers(n_classes, size=n_rows)
y_pred = rng.integers(n_classes, size=n_rows)

temp = rng.integers(n_sensitive_features, size=n_rows)
s_f = [chr(ord('a')+x) for x in temp]

# %%
# Para usar :func:`~ sklearn.metrics.confusion_matrix`,
# es necesario enlazar previamente el argumento `labels` (etiquetas), ya que es posible
# que algunos de los subgrupos no contendrán todos
# las posibles etiquetas


conf_mat = functools.partial(skm.confusion_matrix,
                             labels=np.unique(y_true))

# %%
# Con esto ahora disponible, podemos crear nuestro objeto
# :class:`~ fairlearn.metrics.MetricFrame`:

mf = MetricFrame(metrics={'conf_mat': conf_mat},
                 y_true=y_true,
                 y_pred=y_pred,
                 sensitive_features=s_f)

# %%
# A partir de esto, podemos ver la matriz de confusión general:

mf.overall

# %%
# Y también las matrices de confusión para cada subgrupo:

mf.by_group

# %%
# Obviamente, los otros métodos como
# :meth:`~ fairlearn.metrics.MetricFrame.group_min`
# no funcionarán, ya que operaciones como 'less than' (menor que)
# no están bien definidos para matrices.

# %%
# Las funciones métricas con diferentes tipos de retorno también pueden
# mezclarse con :class:`~ fairlearn.metrics.MetricFrame`.
# Por ejemplo:

recall = functools.partial(skm.recall_score, average='macro')

mf2 = MetricFrame(metrics={'conf_mat': conf_mat,
                           'recall': recall
                           },
                  y_true=y_true,
                  y_pred=y_pred,
                  sensitive_features=s_f)

print("Overall values")
print(mf2.overall)
print("Values by group")
print(mf2.by_group)


# %%
# Argumentos no escalares
# =======================
#
# :class:`~ fairlearn.metrics.MetricFrame` no requiere
# que los argumentos sean escalares. Para demostrar esto,
# utilizaremos un ejemplo de reconocimiento de imágenes (proporcionado amablemente por
# Ferdane Bekmezci, Hamid Vaezi Joze y Samira Pouyanfar).
#
# Los algoritmos de reconocimiento de imágenes frecuentemente construyen un cuadro delimitador
# (bounding box) alrededor de las regiones donde han encontrado las características objetivo.
# Por ejemplo, si un algoritmo detecta un rostro en una imagen,
# colocará un cuadro delimitador a su alrededor. Estos cuadros delimitadores
# constituyen `y_pred` para :class:`~ fairlearn.metrics.MetricFrame`.
# Los valores de `y_true` proceden de los cuadros delimitadores marcados con
# etiquetadores humanos.
#
# Los cuadros delimitadores a menudo se comparan utilizando la métrica 'iou'.
# Ésta calcula la intersección y la unión de los dos
# cuadros delimitadores y devuelve la proporción de sus áreas.
# Si los cuadros delimitadores son idénticos, entonces la métrica
# be 1; si está disjunto, será 0. Una función para hacer esto es:

def bounding_box_iou(box_A_input, box_B_input):
    # The inputs are array-likes in the form
    # [x_0, y_0, delta_x,delta_y]
    # where the deltas are positive

    box_A = np.array(box_A_input)
    box_B = np.array(box_B_input)

    if box_A[2] < 0:
        raise ValueError("Bad delta_x for box_A")
    if box_A[3] < 0:
        raise ValueError("Bad delta y for box_A")
    if box_B[2] < 0:
        raise ValueError("Bad delta x for box_B")
    if box_B[3] < 0:
        raise ValueError("Bad delta y for box_B")

    # Convert deltas to co-ordinates
    box_A[2:4] = box_A[0:2] + box_A[2:4]
    box_B[2:4] = box_B[0:2] + box_B[2:4]

    # Determine the (x, y)-coordinates of the intersection rectangle
    x_A = max(box_A[0], box_B[0])
    y_A = max(box_A[1], box_B[1])
    x_B = min(box_A[2], box_B[2])
    y_B = min(box_A[3], box_B[3])

    if (x_B < x_A) or (y_B < y_A):
        return 0

    # Compute the area of intersection rectangle
    interArea = (x_B - x_A) * (y_B - y_A)

    # Compute the area of both the prediction and ground-truth
    # rectangles
    box_A_area = (box_A[2] - box_A[0]) * (box_A[3] - box_A[1])
    box_B_area = (box_B[2] - box_B[0]) * (box_B[3] - box_B[1])

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(box_A_area + box_B_area - interArea)

    return iou

# %%
# Esta es una métrica para dos cuadros delimitadores, pero para :class:`~ fairlearn.metrics.MetricFrame`
# necesitamos comparar dos listas de cuadros delimitadores. Por
# simplicidad, devolveremos el valor medio de 'iou' para las
# dos listas, pero esta no es la única opción:


def mean_iou(true_boxes, predicted_boxes):
    if len(true_boxes) != len(predicted_boxes):
        raise ValueError("Array size mismatch")

    all_iou = [
        bounding_box_iou(y_true, y_pred)
        for y_true, y_pred in zip(true_boxes, predicted_boxes)
    ]

    return np.mean(all_iou)

# %%
# Necesitamos generar algunos datos de entrada, así que primero crearemos una función para
# generar un solo cuadro delimitador aleatorio:


def generate_bounding_box(max_coord, max_delta, rng):
    corner = max_coord * rng.random(size=2)
    delta = max_delta * rng.random(size=2)

    return np.concatenate((corner, delta))

# %%
# Usaremos esto para crear matrices de muestra `y_true` e `y_pred` de
# cuadros delimitadores:


def many_bounding_boxes(n_rows, max_coord, max_delta, rng):
    return [
        generate_bounding_box(max_coord, max_delta, rng)
        for _ in range(n_rows)
    ]


true_bounding_boxes = many_bounding_boxes(n_rows, 5, 10, rng)
pred_bounding_boxes = many_bounding_boxes(n_rows, 5, 10, rng)

# %%
# Finalmente, podemos usarlos en :class:`~ fairlearn.metrics.MetricFrame`:

mf_bb = MetricFrame(metrics={'mean_iou': mean_iou},
                    y_true=true_bounding_boxes,
                    y_pred=pred_bounding_boxes,
                    sensitive_features=s_f)

print("Overall metric")
print(mf_bb.overall)
print("Metrics by group")
print(mf_bb.by_group)

# %%
# Las entradas individuales en las matrices `y_true` e `y_pred`
# puede ser arbitrariamente complejas. Son las funciones métricas
# que les dan sentido. De manera similar,
# :class:`~ fairlearn.metrics.MetricFrame` no impone
# restricciones sobre el tipo de resultado obtenido. Uno puede imaginarse una tarea
# de imagen de reconocimiento donde hay múltiples objetos detectables en cada
# imagen, y el algoritmo de reconocimiento de imágenes produce
# varios cuadros delimitadores (no necesariamente en un mapeo 1-a-1). El resultado de tal escenario podría
# ser una matriz de alguna descripción.
# Otro caso en el que tanto los datos de entrada como las métricas
# serán complejos es el procesamiento del lenguaje natural,
# donde cada fila de la entrada podría ser una oración completa,
# posiblemente con incrustaciones de palabras complejas incluidas.

# %%
# Conclusión
# ==========
#
# Este tutorial ha probado la flexibilidad
# de :class:`~ fairlearn.metrics.MetricFrame` cuando se trata
# de argumentos de entradas, salida y de funciones métricas.
# Las argumentos de entradas de tipo lista (array) pueden tener elementos de tipos arbitrarios,
# y los valores de retorno de las funciones métricas también pueden
# ser de cualquier tipo (aunque métodos como
# :meth:`~ fairlearn.metrics.MetricFrame.group_min` puede no
# trabajo).
