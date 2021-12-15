.. _quickstart:

Introducción
============

Instalación
------------

Fairlearn puede ser instalado usando :code:`pip` de
`PyPI <https://pypi.org/project/fairlearn>`_ de la siguiente manera:

.. code-block:: bash

   pip install fairlearn

Fairlearn también está disponible a través de
`conda-forge <https://anaconda.org/conda-forge/fairlearn>`_:

.. code-block:: bash

    conda install -c conda-forge fairlearn

Para más información sobre cómo instalar Fairlearn y sus
dependencias opcionales, consulte :ref:`guía_instalación`.

Si está actualizando desde una versión anterior de Fairlearn, consulte :ref:`guía_versión`.

.. note::

    La API de Fairlearn aún está evolucionando, por lo que es posible que el código de
    ejemplo de esta documentación no funcione con todas las versiones de Fairlearn.
    Utilice el selector de versión para obtener las instrucciones de la versión adecuada.
    Las instrucciones para la rama principal requieren que Fairlearn se instale desde un
    clon del repositorio.

Descripción general de Fairlearn
--------------------------------

El paquete Fairlearn tiene dos componentes:

- *Métricas* para evaluar qué grupos se ven afectados negativamente por un modelo
  y para comparar varios modelos en términos de diversas métricas de equidad y precisión.

- *Algoritmos* para mitigar la injusticia en una variedad de tareas de IA y en una variedad de definiciones de equidad.

Fairlearn en 10 minutos
-----------------------

El conjunto de herramientas de Fairlearn puede ayudar a evaluar y mitigar la injusticia en
Modelos de Aprendizaje Automático (MAA). Es imposible proporcionar una descripción general suficiente de
equidad en MAA en este tutorial de inicio rápido, por lo que recomendamos comenzar
con nuestra: ref: `guía_usuario`. La equidad es fundamentalmente un
desafío socio técnico y no se puede resolver con herramientas técnicas solamente.
Pueden ser de ayuda para ciertas tareas, como evaluar la injusticia a través de
varias métricas, o para mitigar la injusticia observada al entrenar un modelo.
Además, la equidad tiene diferentes definiciones en diferentes contextos y puede que no sea posible
representarlo cuantitativamente en absoluto.

Dadas estas consideraciones, este tutorial de inicio rápido simplemente proporciona breves
ejemplos de fragmentos de código de cómo utilizar la funcionalidad básica de Fairlearn para aquellos
que ya están íntimamente familiarizados con la equidad en MAA. El siguiente ejemplo
se trata de clasificación binaria, pero también apoyamos la regresión.

Cargando el conjunto de datos
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Para este ejemplo usamos el
`Conjunto de datos de adultos UCI <https://archive.ics.uci.edu/ml/datasets/Adult>`_ donde el
el objetivo es predecir si una persona gana más (etiqueta 1) o menos (0)
de $ 50.000 al año.

.. doctest:: quickstart

    >>> import numpy as np
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt
    >>> from sklearn.datasets import fetch_openml
    >>> data = fetch_openml(data_id=1590, as_frame=True)
    >>> X = pd.get_dummies(data.data)
    >>> y_true = (data.target == '>50K') * 1
    >>> sex = data.data['sex']
    >>> sex.value_counts()
    Male      32650
    Female    16192
    Name: sex, dtype: int64

.. figure:: auto_examples/images/sphx_glr_plot_quickstart_selection_rate_001.png
    :target: auto_examples/plot_quickstart_selection_rate.html
    :align: center

Evaluación de métricas relacionadas con la equidad
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

En primer lugar, Fairlearn proporciona métricas relacionadas con la equidad que se pueden comparar
entre grupos y para la población en general. Usando métrica existente
definiciones de
`scikit-learn <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics>`_
podemos evaluar métricas para subgrupos dentro de los datos de la siguiente manera:

.. doctest:: quickstart
    :options:  +NORMALIZE_WHITESPACE

    >>> from fairlearn.metrics import MetricFrame
    >>> from sklearn.metrics import accuracy_score
    >>> from sklearn.tree import DecisionTreeClassifier
    >>>
    >>> classifier = DecisionTreeClassifier(min_samples_leaf=10, max_depth=4)
    >>> classifier.fit(X, y_true)
    DecisionTreeClassifier(...)
    >>> y_pred = classifier.predict(X)
    >>> gm = MetricFrame(metrics=accuracy_score, y_true=y_true, y_pred=y_pred, sensitive_features=sex)
    >>> print(gm.overall)
    0.8443...
    >>> print(gm.by_group)
    sex
    Female    0.9251...
    Male      0.8042...
    Name: accuracy_score, dtype: object

Además, Fairlearn tiene muchas otras métricas estándar integradas, como
tasa de selección, es decir, el porcentaje de la población que tiene '1' como
su etiqueta:

.. doctest:: quickstart
    :options:  +NORMALIZE_WHITESPACE

    >>> from fairlearn.metrics import selection_rate
    >>> sr = MetricFrame(metrics=selection_rate, y_true=y_true, y_pred=y_pred, sensitive_features=sex)
    >>> sr.overall
    0.1638...
    >>> sr.by_group
    sex
    Female    0.0635...
    Male      0.2135...
    Name: selection_rate, dtype: object

Fairlearn también nos permite trazar rápidamente estas métricas desde
:class:`fairlearn.metrics.MetricFrame`

.. literalinclude:: auto_examples/plot_quickstart.py
    :language: python
    :start-after: # Analyze metrics using MetricFrame
    :end-before: # Customize plots with ylim

.. figure:: auto_examples/images/sphx_glr_plot_quickstart_001.png
    :target: auto_examples/plot_quickstart.html
    :align: center


Disminuyendo los imbalances
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Si observamos imbalances entre grupos, es posible que deseemos crear un nuevo modelo.
mientras se especifica una restricción de equidad adecuada. Tenga en cuenta que la elección de
Las restricciones de equidad son cruciales para el modelo resultante y varían en función de
contexto de la aplicación. Si la tasa de selección es muy relevante para la equidad en este
ejemplo artificial, podemos intentar mitigar la disparidad observada utilizando el
restricción de equidad correspondiente denominada paridad demográfica. En el mundo real
aplicaciones tenemos que ser conscientes del contexto socio técnico al hacer
tales decisiones. La técnica de mitigación de gradiente exponencial utilizada se ajusta al
proporcionó un clasificador utilizando la paridad demográfica como objetivo, lo que
una diferencia enormemente reducida en la tasa de selección:

.. doctest:: quickstart
    :options:  +NORMALIZE_WHITESPACE

    >>> from fairlearn.reductions import ExponentiatedGradient, DemographicParity
    >>> np.random.seed(0)  # set seed for consistent results with ExponentiatedGradient
    >>>
    >>> constraint = DemographicParity()
    >>> classifier = DecisionTreeClassifier(min_samples_leaf=10, max_depth=4)
    >>> mitigator = ExponentiatedGradient(classifier, constraint)
    >>> mitigator.fit(X, y_true, sensitive_features=sex)
    ExponentiatedGradient(...)
    >>> y_pred_mitigated = mitigator.predict(X)
    >>>
    >>> sr_mitigated = MetricFrame(metrics=selection_rate, y_true=y_true, y_pred=y_pred_mitigated, sensitive_features=sex)
    >>> print(sr_mitigated.overall)
    0.1661...
    >>> print(sr_mitigated.by_group)
    sex
    Female    0.1552...
    Male      0.1715...
    Name: selection_rate, dtype: object


¿Qué sigue?
------------

Consulte nuestra :ref:`guía_usuario` para obtener una visión completa de la equidad en
aprendizaje automático y cómo encaja Fairlearn, así como una guía exhaustiva sobre
todas las partes del juego de herramientas. Para obtener ejemplos concretos, consulte
la sección :ref:`sphx_glr_auto_examples`. Finalmente, también tenemos una colección
de :ref:`preguntas_frequentes`.
