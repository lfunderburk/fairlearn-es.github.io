.. _acerca_de:

Acerca de nosotros
==================

.. _mision:

Misión
------

Fairlearn es un proyecto de código abierto impulsado por la comunidad para ayudar a los científicos de datos
mejorar la equidad de los sistemas de inteligencia artificial.

El proyecto aspira a incluir:

- Una biblioteca de Python para la evaluación y mejora de la equidad (métricas de equidad,
  algoritmos de mitigación, trazado, etc.)
- Recursos educativos que abarcan procesos organizativos y técnicos para
  mitigación de la injusticia (guía de usuario completa, estudios de casos detallados,
  Jupyter notebooks, Libros Blancos, etc.)

El desarrollo de Fairlearn se basa firmemente en el entendimiento de que la equidad
en los sistemas de inteligencia artificial (IA) es un desafío sociotécnico.
Debido a que hay muchas fuentes complejas de injusticia, algunas sociales y
algunas técnicos --- no es posible eliminar el prejuicio completamente en un sistema o
garantizar la equidad.
Nuestro objetivo es permitir que los humanos evalúen los daños relacionados con la equidad, revisar el
impactos de diferentes estrategias de mitigación y luego hacer concesiones
apropiado a su escenario.

¡Fairlearn es un proyecto de código abierto impulsado por la comunidad!
El desarrollo y crecimiento de Fairlearn están guiados por la creencia de que
el progreso hacia sistemas de IA más justos requiere la participación de una amplia variedad de
de perspectivas, que van desde científicos de datos, desarrolladores y empresas
que toman decisiones a las personas cuyas vidas pueden verse afectadas por las predicciones
de los sistemas de IA.

.. _code_of_conduct:

Código de conducta
------------------

Fairlearn se guía en el siguiente código de conducta:
`Código de Conducta de la organización (en Inglés) <https://github.com/fairlearn/governance/blob/main/code-of-conduct.md>`_.

.. _roadmap:

Áreas de enfoque en el proyecto
-------------------------------

* Última actualización: 16 de mayo de 2021 *

Como proyecto de código abierto, Fairlearn se esfuerza por incorporar lo mejor de
investigación y práctica.

La IA es un campo en rápida evolución, y la justicia en la IA lo es aún más.
Por lo tanto, alentamos a los investigadores y profesionales interesados a
contribuir con métricas de equidad y herramientas de evaluación, mitigación de la injusticia
algoritmos, estudios de casos y otros materiales educativos a Fairlearn para que podamos
experimentar, aprender y revolucionar el proyecto juntos.

A continuación, enumeramos las áreas clave que priorizamos en el corto
y mediano plazo, pero nos complace considerar otras direcciones
si están alineados con la misión de Fairlearn y hay suficiente compromiso
de los contribuyentes. Si quieres involucrarte, por favor comuníquese con
:ref:`reach out <communication>`. Para oportunidades concretas y
trabajo en progreso por favor revise nuestras
`issues <https://github.com/fairlearn/fairlearn/issues>`_.

#. *Disminuir las barreras de adopción para las herramientas actuales de evaluación y mitigación en Fairlearn*

   - **Mejore los casos de uso existentes y cree nuevos** que
     `hacen que los problemas de equidad sean concretos <https://fairlearn.github.io/contributor_guide/contributing_example_notebooks.html>`_.
     Estos casos de uso pueden utilizar o no el paquete Fairlearn.
     Para acelerar este proceso, estamos experimentando con
     :ref:`sesiones semanales <community_calls>` donde la gente puede discutir ideas,
     proyectos en curso y cuadernos de ejemplo individuales en detalle.

   - **Mejorar la documentación del proyecto**: criticar el contenido actual,
     ampliar las guías de usuario, mejorar la escritura y los ejemplos, migrar la
     documentación de Python actual al formato numpydoc.
     Convertir los Jupyter notebooks existentes de `.ipynb` en archivos` .py` que
     `rendericen correctamente en el sitio web <https://fairlearn.github.io/auto_examples/notebooks/index.html>`_,
     sin dejar de ser descargable como archivos `.py` o` .ipynb`.

   - **Mejorar la usabilidad, la relevancia y el aspecto del sitio web de Fairlearn**
     con la audiencia de practicantes en mente.
     Participe participando en los debates sobre
     `llamadas de la comunidad <comunidades_llamadas>`_ o en las correspondientes
     `discusiones <https://github.com/fairlearn/fairlearn/discussions>`_.
     También puede enviarnos sus comentarios presentando un nuevo número.

   - **Mejorar la usabilidad y relevancia de las métricas de equidad** al
     criticar y mejorar la API de métricas actual, sugiriendo nuevas métricas
     motivadas por casos de uso concretos e implementando las nuevas métricas.
     La página de problemas
     `contiene varias tareas de métricas <https://github.com/fairlearn/fairlearn/issues?q=is%3Aissue+is%3Aopen+metric>`_.

   - **Avanzar hacia la compatibilidad con scikit-learn**:
     identificar aspectos incompatibles, mejorar el código hacia la compatibilidad.
     Si bien nuestro objetivo es la compatibilidad, puede haber aspectos que sean demasiado
     restringidos para Fairlearn, por lo que esto puede necesitar ser evaluado en un
     caso por caso.

#. *Crecer y nutrir una comunidad diversa de colaboradores*

   - **Comuníquese** con comentarios sobre lo que está funcionando y lo que
     no necesita mejorar; sugiera cómo mejorar las cosas; señale donde la
     documentación, procesos o cualquier otro aspecto del proyecto crean
     barreras de entrada.

   - **Participe** en nuestras :ref:`llamadas comunitarias semanales <community_calls>`.
     También trabajamos con universidades para involucrar a estudiantes
     a través de proyectos y otras formas de
     colaboración --- háganos saber si está interesado.

   - **Mejorar el sitio web y la documentación de Fairlearn**.
     Consulte la guía para colaboradores en
     :ref:`cómo contribuir a nuestros documentos <contributing_documentation>`.

   - **Agregue pruebas de código y mejore la infraestructura de las pruebas.**

#. *Crear métricas, herramientas de evaluación y algoritmos para cubrir tareas de aprendizaje automático más complejas*

   - **Crear material y casos de uso** que se ocupen de
     :ref:`problemas concretos de equidad <contributing_example_notebooks>`
     en tareas complejas de aprendizaje automático que incluyen clasificación, estimación contrafactual,
     procesamiento de texto, visión por computadora, habla, etc.

   - **Liderar y participar en los esfuerzos de contribución**
     en torno a áreas de aprendizaje automático poco investigadas, pero relevantes de manera
     práctica en la clasificación, estimación contrafactual, texto, visión por computadora,
     habla, etc. Es probable que estos sean esfuerzos mixtos de investigación / práctica y esperamos
     compromiso sustancial de los contribuyentes antes de embarcarse en estos.

.. _governance:

Gobernanza
----------

Fairlearn es un proyecto de la
`organización Fairlearn <https://github.com/fairlearn/governance/blob/main/ORG-GOVERNANCE.md>`_
y sigue el
`gobierno de proyectos de la organización Fairlearn <https://github.com/fairlearn/governance/blob/main/PROJECT-GOVERNANCE.md>`_.

.. _maintainers:

Mantenedores
^^^^^^^^^^^^

Los mantenedores actuales del proyecto Fairlearn son

.. include:: maintainers.rst

Traductores a Español
^^^^^^^^^^^^^^^^^^^^^

.. include:: traductores_espanol.rst

.. _history:

Historia del proyecto
---------------------

Fairlearn se inició en 2018 por Miro Dudik de Microsoft Research como un
Paquete de Python para acompañar el trabajo de investigación,
`Un enfoque de reducciones para una clasificación justa <http://proceedings.mlr.press/v80/agarwal18a/agarwal18a.pdf>`_.
El paquete proporcionó un algoritmo de reducción para mitigar la injusticia en
modelos de clasificación binarios --- un escenario que fue comúnmente estudiado en la
comunidad de aprendizaje automático.
El documento y el paquete de Python fueron bien recibidos, por lo que Miro Dudik y Hanna
Wallach con sus colaboradores buscaron traducir la investigación en un
contexto de la industria.
Sin embargo, descubrieron que los profesionales normalmente necesitan abordar más
cuestiones fundamentales de equidad antes de aplicar algoritmos específicos, y que
mitigar la injusticia en los modelos de clasificación binaria es un uso relativamente raro
caso.
También descubrieron que la evaluación de la equidad es una necesidad común, junto con
acceso a guías específicas del dominio para métricas de equidad y mitigación de injusticias
algoritmos.
Además, muchos casos de uso toman la forma de regresión o clasificación, en lugar de
que la clasificación.
Como resultado de estos conocimientos, la evaluación de la equidad y los cuadernos de casos de uso
se convirtieron en componentes clave de Fairlearn.
Fairlearn también se enfoca en tareas de aprendizaje automático más allá de la clasificación binaria.

El proyecto se amplió enormemente en el segundo semestre de 2019 gracias a la
participación de muchos colaboradores de Azure ML y Microsoft Research.
En ese momento, el proyecto comenzó a tener lanzamientos regulares.

En 2021 Fairlearn adoptó
`gobernanza neutral <https://github.com/fairlearn/governance>`_
y desde entonces el proyecto está completamente impulsado por la comunidad.

Citando a Fairlearn
-------------------

Si desea citar Fairlearn en su trabajo, utilice lo siguiente:

.. code ::

    @techreport{bird2020fairlearn,
        author = {Bird, Sarah and Dud{\'i}k, Miro and Edgar, Richard and Horn, Brandon and Lutz, Roman and Milan, Vanessa and Sameki, Mehrnoosh and Wallach, Hanna and Walker, Kathleen},
        title = {Fairlearn: A toolkit for assessing and improving fairness in {AI}},
        institution = {Microsoft},
        year = {2020},
        month = {May},
        url = "https://www.microsoft.com/en-us/research/publication/fairlearn-a-toolkit-for-assessing-and-improving-fairness-in-ai/",
        number = {MSR-TR-2020-32},
    }

Preguntas frecuentes
--------------------------

Vea nuestra página :ref:`preguntas_frequentes`.

Fondos
------

Fairlearn es un proyecto impulsado por la comunidad. Sin embargo, varias empresas y
Las instituciones académicas ayudan a asegurar su sostenibilidad.
Nos gustaría agradecer a los siguientes patrocinadores.

.. raw:: html

   <div class="sponsor-div">
   <div class="sponsor-div-box">

`Microsoft <https://www.microsoft.com/>`_ ha financiado a varios contribuyentes
incluidos varios mantenedores (Miro Dudik, Richard Edgar, Roman Lutz, Michael
Madaio) desde el inicio del proyecto en 2018.

.. raw:: html

   </div>
   <div class="sponsor-div-box">

.. image:: ../_static/images/microsoft.png
   :width: 100pt
   :align: center
   :target: https://www.microsoft.com/

.. raw:: html

   </div>
   </div>
   <div class="sponsor-div">
   <div class="sponsor-div-box">

`Eindhoven University of Technology <https://www.tue.nl/en/>`_ ha financiado
Hilde Weerts desde marzo de 2020.

.. raw:: html

  </div>
  <div class="sponsor-div-box">

.. image:: ../_static/images/tu_eindhoven.png
  :width: 100pt
  :align: center
  :target: https://www.tue.nl/en/

.. raw:: html

  </div>
  </div>
  <div class="sponsor-div">
  <div class="sponsor-div-box">

`Zalando <https://corporate.zalando.com/en>`_ ha financiado a Adrin Jalali desde
Agosto de 2020.

.. raw:: html

  </div>
  <div class="sponsor-div-box">

.. image:: ../_static/images/zalando.png
  :width: 100pt
  :align: center
  :target: https://corporate.zalando.com/en

.. raw:: html

  </div>
  </div>

Patrocinadores anteriores
^^^^^^^^^^^^^^^^^^^^^^^^^

.. raw:: html

  <div class="sponsor-div">
  <div class="sponsor-div-box">

`Anaconda <https://www.anaconda.com/>`_ financió a Adrin Jalali en 2019.

.. raw:: html

  </div>
  <div class="sponsor-div-box">

.. image:: ../_static/images/anaconda.png
  :width: 100pt
  :align: center
  :target: https://www.anaconda.com/

.. raw:: html

  </div>
  </div>

Soporte de infraestructura
--------------------------

También nos gustaría agradecer a las siguientes personas por el tiempo de CPU libre en sus
servidores de integración continua:

- `Microsoft Azure <https://azure.microsoft.com/en-us/>`_
- `GitHub <https://github.com>`_
- `CircleCI <https://circleci.com/>`_
