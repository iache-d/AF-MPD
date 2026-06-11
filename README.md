# AF-MPD — Simulador de un propulsor magnetoplasmadinámico de campo aplicado

Proyecto de FIS205. Este repositorio contiene el simulador de un propulsor **AF-MPD** (*Applied-Field Magnetoplasmadynamic thruster*): el núcleo de cálculo está escrito en C++ y se utiliza desde Python a través de notebooks de Jupyter para configurar geometrías, lanzar simulaciones y analizar resultados.

> **Idea clave para compilar y ejecutar:** el código C++ no produce un ejecutable `.exe` independiente. Se compila como un **módulo de Python llamado `motor_mpd_cpp`**. El "programa principal" del proyecto son los notebooks, que hacen `import motor_mpd_cpp` para correr las simulaciones. Por lo tanto, **antes de ejecutar cualquier notebook hay que compilar el módulo `motor_mpd_cpp`** (ver la sección [Compilación](#compilación-del-módulo-motor_mpd_cpp)).

---

## Estructura del repositorio

```
AF-MPD/
├── Informe/                      # Informe del proyecto
└── ARCHIVOS/
    ├── CMakeLists.txt            # Configuración de compilación (CMake)
    ├── src/
    │   └── cpp/                  # Código fuente C++ del simulador
    │       ├── malla.hpp         # Declaración de la malla / dominio de cálculo
    │       ├── malla.cpp         # Implementación de la malla
    │       ├── solver_mhd.cpp    # Solver magnetohidrodinámico (núcleo físico)
    │       └── bindings.cpp      # Enlaces C++ ↔ Python → genera el módulo motor_mpd_cpp
    ├── build/                    # Solución/proyecto generados por CMake (ver nota al final)
    ├── notebooks/                # Notebooks de Jupyter (análisis, validación, animaciones)
    └── multimedia/               # Figuras, animaciones y demás recursos generados
```

---

## Descripción de los componentes

### `src/cpp/` — núcleo en C++
Es donde vive toda la física y el cálculo pesado:

- **`malla.hpp` / `malla.cpp`** definen y construyen la malla, es decir, el dominio discretizado sobre el que se resuelve el problema.
- **`solver_mhd.cpp`** implementa el solver magnetohidrodinámico (MHD), el corazón de la simulación del propulsor.
- **`bindings.cpp`** expone las clases y funciones de C++ a Python. Al compilar, este archivo es el que produce el módulo `motor_mpd_cpp` que luego importan los notebooks.

### `CMakeLists.txt`
Describe cómo compilar el módulo. Ahí se definen el estándar de C++, las dependencias y el objetivo (*target*) `motor_mpd_cpp`.

### `build/`
Carpeta generada por CMake con la solución de Visual Studio (`SimuladorAF_MPD.sln`) y los archivos de proyecto (`motor_mpd_cpp.vcxproj`, etc.). Se incluye en el repositorio para tener disponible la configuración de compilación ya generada; más abajo se explica también cómo regenerarla desde cero.

### `notebooks/` — uso, análisis y visualización
Son el punto de entrada del proyecto. Importan `motor_mpd_cpp` y lo usan para simular y graficar. Estado actual:

- **`Estructua_basica_inicial.ipynb`** — estructura base / punto de partida del flujo de trabajo.
- **`Testeo_de_topologías.ipynb`** — pruebas de topologías (notebook activo más reciente).
- Los notebooks marcados con **`(DESACTUALIZADO)`** —`02_tobera_magnetica`, `05_validacion_empuje`, `06_optimizador_4D`, `10_animacion_int2`— corresponden a exploraciones anteriores que se conservan como referencia, pero que ya no reflejan el estado actual del simulador.

### `multimedia/`
Recursos generados (figuras, animaciones, etc.) producidos por los notebooks.

### `Informe/`
El informe escrito del proyecto.

---

## Requisitos

> Verifica las versiones y dependencias exactas en `ARCHIVOS/CMakeLists.txt` (cualquier `find_package(...)` que aparezca ahí es una dependencia que hay que tener instalada).

- **CMake** (3.15 o superior recomendado).
- **Compilador de C++** con soporte para el estándar usado en el proyecto (C++17 es lo habitual; confírmalo en el `CMakeLists.txt`).
  - *Windows:* Visual Studio 2019/2022 con el componente **"Desarrollo para escritorio con C++"**.
  - *Linux / macOS:* `g++` o `clang`.
- **Python 3.x**, con sus cabeceras de desarrollo (`python3-dev` en Linux).
- **pybind11** (asumido a partir de `bindings.cpp`; confirmá en el `CMakeLists.txt`).
- **Jupyter** (JupyterLab o Notebook) para correr los notebooks, junto con las librerías que estos usen (típicamente `numpy` y `matplotlib`).

---

## Compilación del módulo `motor_mpd_cpp`

Todos los comandos se ejecutan desde la carpeta `ARCHIVOS/`.

### Windows (Visual Studio)

Opción por línea de comandos:

```bash
cmake -S . -B build
cmake --build build --config Release
```

O bien abrir `build/SimuladorAF_MPD.sln` en Visual Studio, seleccionar la configuración **Release / x64** y compilar el objetivo `motor_mpd_cpp`.

El módulo compilado quedará como un archivo `motor_mpd_cpp*.pyd` dentro de `build/` (por ejemplo en `build/Release/`; la ruta exacta depende del `CMakeLists.txt`).

### Linux / macOS

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

En este caso el módulo se genera como `motor_mpd_cpp*.so`.

---

## Cómo usar el módulo en los notebooks

Para que un notebook pueda hacer `import motor_mpd_cpp`, Python tiene que encontrar el archivo compilado (`.pyd` en Windows, `.so` en Linux/macOS). La forma más simple y robusta es agregar la carpeta de salida al *path* al inicio del notebook:

```python
import sys
sys.path.append(r"../build/Release")   # ajustá esta ruta a donde quedó el módulo
import motor_mpd_cpp
```

(Como alternativa, se puede copiar el archivo `.pyd`/`.so` directamente junto a los notebooks.)

Una vez que el `import` funciona, los notebooks pueden ejecutarse de principio a fin.

---


**Nota técnica:** el módulo compilado (`.pyd` / `.so`) depende del sistema operativo y de la versión de Python con que se generó, así que el `build/` incluido sirve sobre todo como referencia (Windows x64). La vía fiable y portable para evaluar el proyecto es **compilar desde el código fuente** siguiendo los pasos de arriba.
