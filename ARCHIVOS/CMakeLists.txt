
cmake_minimum_required(VERSION 3.15)

# Nombre de nuestro gran proyecto
project(SimuladorAF_MPD)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



find_package(Python3 COMPONENTS Interpreter Development REQUIRED)


execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)


find_package(pybind11 REQUIRED CONFIG PATHS "${pybind11_DIR}")


pybind11_add_module(motor_mpd_cpp src/cpp/malla.cpp)
