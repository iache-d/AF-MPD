# AF-MPD — Modelo de erosión catódica en un propulsor magnetoplasmadinámico de campo aplicado

Proyecto de FIS205. Este repositorio contiene el modelo computacional de erosión catódica de un propulsor **AF-MPD** (*Applied-Field Magnetoplasmadynamic thruster*). El modelo que produce los resultados del informe es un **trazado de partículas de prueba (*test-particle*)** de iones de argón, implementado en **Python (NumPy)** dentro de un notebook de Jupyter. Adicionalmente, el repositorio incluye un **prototipo en C++** que se usó para una etapa de verificación con malla.

> **Cómo está organizado el cálculo:**
> - El **modelo definitivo** (test-particle, sin malla, campos analíticos) vive íntegro en el notebook de Python y se ejecuta solo con `numpy`, `matplotlib`, `scipy` y `tqdm`. **No requiere compilar nada.**
> - El **prototipo en C++** (`malla.cpp`) se compila como un módulo de Python llamado `motor_mpd_cpp`. Solo las celdas de verificación de malla y la animación del modelo de campos prescritos lo importan; el resto del notebook corre sin él.

---

## Estructura del repositorio

~~~
AF-MPD/
├── Informe/                      # Informe del proyecto (paper)
└── ARCHIVOS/
    ├── CMakeLists.txt            # Configuración de compilación (CMake)
    ├── requirements.txt          # Dependencias de Python
    ├── src/
    │   └── cpp/                  # Prototipo en C++ (etapa de verificación con malla)
    │       ├── malla.hpp         # Declaración de la malla / dominio de cálculo
    │       ├── malla.cpp         # Implementación de la malla y el integrador de prueba
    │       └── bindings.cpp      # Enlaces C++ <-> Python -> genera el módulo motor_mpd_cpp
    ├── build/                    # Proyecto generado por CMake (ver nota al final)
    ├── notebooks/                # Notebook principal + figuras y animaciones generadas
    └── multimedia/               # Recursos generados (figuras, animaciones)
~~~

---

## Descripción de los componentes

### `notebooks/` — modelo definitivo y análisis
Es el punto de entrada y el corazón del proyecto. El notebook principal contiene el **modelo test-particle de campos analíticos** (el que genera todos los resultados del informe), además de las etapas de diseño geométrico, la verificación del prototipo y el estudio de topologías HTS. Está organizado por secciones numeradas, desde los parámetros maestros hasta la comparación final de configuraciones de campo.

### `src/cpp/` — prototipo de verificación en C++
Contiene el prototipo discretizado sobre malla que se usó para **verificar** la implementación del integrador contra la solución analítica, antes de adoptar el modelo definitivo sin malla.

- **`malla.hpp` / `malla.cpp`** definen y construyen la malla y el cálculo sobre ella.
- **`bindings.cpp`** expone las clases de C++ a Python; al compilar, produce el módulo `motor_mpd_cpp`.

Este prototipo **no es el motor de producción**: el modelo final, sin malla y con campos evaluados analíticamente, está implementado en Python en el notebook. El C++ se conserva como referencia de la etapa de verificación.

### `CMakeLists.txt`
Describe cómo compilar el módulo `motor_mpd_cpp` (estándar de C++, dependencias y *target*).

### `build/`
Carpeta generada por CMake con el proyecto de compilación ya configurado (Windows x64). Se incluye como referencia; más abajo se explica cómo regenerarla desde cero.

### `multimedia/` e `Informe/`
Recursos generados (figuras, animaciones) y el informe escrito del proyecto.

---

## Requisitos

### Python (para el modelo definitivo)
Basta un entorno con las dependencias de `requirements.txt`:

~~~
numpy
matplotlib
scipy
tqdm
pybind11
~~~

Más Jupyter (JupyterLab o Notebook) para abrir el notebook.

### C++ (solo para el prototipo de verificación)
> Verifica las versiones y dependencias exactas en `ARCHIVOS/CMakeLists.txt` (cualquier `find_package(...)` que aparezca ahí es una dependencia a instalar).

- **CMake** (3.15 o superior recomendado).
- **Compilador de C++** con soporte para el estándar del proyecto (C++17 es lo habitual; confírmalo en el `CMakeLists.txt`).
  - *Windows:* Visual Studio 2019/2022 con el componente **"Desarrollo para escritorio con C++"**.
  - *Linux / macOS:* `g++` o `clang`.
- **Python 3.x** con sus cabeceras de desarrollo (`python3-dev` en Linux).
- **pybind11** (confírmalo en el `CMakeLists.txt`).

---

## Compilación del módulo `motor_mpd_cpp` (opcional)

Solo es necesaria si quieres ejecutar las celdas de verificación de malla o la animación del modelo de campos prescritos. El resto del notebook corre sin compilar nada.

Todos los comandos se ejecutan desde la carpeta `ARCHIVOS/`.

### Windows (Visual Studio)

~~~bash
cmake -S . -B build
cmake --build build --config Release
~~~

O abrir `build/SimuladorAF_MPD.sln` en Visual Studio, seleccionar **Release / x64** y compilar el objetivo `motor_mpd_cpp`. El módulo quedará como `motor_mpd_cpp*.pyd` dentro de `build/` (por ejemplo en `build/Release/`).

### Linux / macOS

~~~bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
~~~

El módulo se genera como `motor_mpd_cpp*.so`.

---

## Cómo usar el módulo en el notebook

Para que el notebook pueda hacer `import motor_mpd_cpp`, Python tiene que encontrar el archivo compilado (`.pyd` en Windows, `.so` en Linux/macOS). La forma más simple es agregar la carpeta de salida al *path* al inicio del notebook:

~~~python
import sys
sys.path.append(r"../build/Release")   # ajusta esta ruta a donde quedó el módulo
import motor_mpd_cpp
~~~

(Como alternativa, se puede copiar el archivo `.pyd`/`.so` directamente junto al notebook.)

---

**Nota técnica:** el módulo compilado (`.pyd` / `.so`) depende del sistema operativo y de la versión de Python con que se generó, así que el `build/` incluido sirve sobre todo como referencia (Windows x64). La vía fiable y portable es **compilar desde el código fuente** siguiendo los pasos de arriba.
