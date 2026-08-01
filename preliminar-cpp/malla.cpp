#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <cstdlib> // Para generar las posiciones aleatorias iniciales
#include <algorithm>
#include <vector>
#include <random>

namespace py = pybind11;


struct IonArgon {
    double x, y, z;
    double vx, vy, vz;
    int estado;
    double energia_impacto;
};


class Malla2D {
public:
    int Nr, Nz;
    double r_catodo, r_anodo, L_catodo, L_anodo;
    bool es_campana;


    Malla2D(int Nr_, int Nz_, double r_c, double r_a, double L_c_, double L_a_, bool campana)
        : Nr(Nr_), Nz(Nz_), r_catodo(r_c), r_anodo(r_a), L_catodo(L_c_), L_anodo(L_a_), es_campana(campana) {}

    py::array_t<double> obtener_Z() {
        auto resultado = py::array_t<double>({Nr, Nz});
        double* ptr = (double*) resultado.request().ptr;
        

        double dz = L_anodo / (Nz - 1); 
        for (int i = 0; i < Nr; i++) {
            for (int j = 0; j < Nz; j++) {
                ptr[i * Nz + j] = j * dz;
            }
        }
        return resultado;
    }

    py::array_t<double> obtener_R() {
        auto resultado = py::array_t<double>({Nr, Nz});
        double* ptr = (double*) resultado.request().ptr;
        double dz = L_anodo / (Nz - 1);
        
        for (int j = 0; j < Nz; j++) {
            double z = j * dz;
            double r_out = r_anodo; 
            
            if (es_campana) {
                double a = 4.0;
                r_out = r_anodo * std::pow(1.0 + std::pow(z / a, 2), 0.75);
            }


            double r_in_actual = (z <= L_catodo) ? r_catodo : 0.0001;
            
            double dr = (r_out - r_in_actual) / (Nr - 1);
            for (int i = 0; i < Nr; i++) {
                ptr[i * Nz + j] = r_in_actual + i * dr;
            }
        }
        return resultado;
    }
};


class PlasmaArgon {
public:
    int Nr, Nz;
    py::array_t<double> densidad, vel_r, vel_z, temperatura;

    PlasmaArgon(int Nr_, int Nz_) : Nr(Nr_), Nz(Nz_) {
        densidad = py::array_t<double>({Nr, Nz});
        vel_r = py::array_t<double>({Nr, Nz});
        vel_z = py::array_t<double>({Nr, Nz});
        temperatura = py::array_t<double>({Nr, Nz});
    }

    void inicializar_condiciones(double rho_0, double T_0, double u_z_0) {
        double* ptr_rho = (double*) densidad.request().ptr;
        double* ptr_vr  = (double*) vel_r.request().ptr;
        double* ptr_vz  = (double*) vel_z.request().ptr;
        double* ptr_T   = (double*) temperatura.request().ptr;
        for(int i = 0; i < Nr * Nz; i++) {
            ptr_rho[i] = rho_0; ptr_T[i] = T_0;
            ptr_vr[i] = 0.0;    ptr_vz[i] = u_z_0;
        }
    }
};


class CampoMagnetico {
public:
    int Nr, Nz;
    py::array_t<double> Br, Bz;

    CampoMagnetico(int Nr_, int Nz_) : Nr(Nr_), Nz(Nz_) {
        Br = py::array_t<double>({Nr, Nz});
        Bz = py::array_t<double>({Nr, Nz});
    }


    void calcular_campo_aplicado(double B0, double L_catodo, double L_anodo, double r_catodo, double r_anodo, bool es_campana) {
        double* ptr_Br = (double*) Br.request().ptr;
        double* ptr_Bz = (double*) Bz.request().ptr;
        double dz = L_anodo / (Nz - 1); 

        for(int j = 0; j < Nz; j++) {
            double z = j * dz;
            double a = 4.0; 
            
            // Campo de Biot-Savart real en el eje
            double Bz_val = B0 / std::pow(1.0 + std::pow(z / a, 2), 1.5);
            
            // Derivada real de Bz respecto a z (usada para la divergencia de Br)
            double dBz_dz = -3.0 * B0 * z / (a * a * std::pow(1.0 + std::pow(z / a, 2), 2.5));

            double r_out = r_anodo;
            if (es_campana) {
                r_out = r_anodo * std::pow(1.0 + std::pow(z / a, 2), 0.75);
            }
            

            // El radio interno debe coincidir EXACTAMENTE con el de Malla2D
            double r_in_actual = (z <= L_catodo) ? r_catodo : 0.0001; 
            
            double dr = (r_out - r_in_actual) / (Nr - 1);

            for(int i = 0; i < Nr; i++) {
                double r = r_in_actual + i * dr;
                
                // Asignación de los tensores de campo magnético
                ptr_Bz[i * Nz + j] = Bz_val;
                
                // Usando la aproximación paraxial: Br = -0.5 * r * (dBz/dz)
                ptr_Br[i * Nz + j] = -0.5 * r * dBz_dz;
            }
        }
    }
};

class FuerzaLorentz {
public:
    int Nr, Nz;
    py::array_t<double> Jr, Jtheta, Fz, Fr;

    FuerzaLorentz(int Nr_, int Nz_) : Nr(Nr_), Nz(Nz_) {
        Jr = py::array_t<double>({Nr, Nz});
        Jtheta = py::array_t<double>({Nr, Nz});
        Fz = py::array_t<double>({Nr, Nz});
        Fr = py::array_t<double>({Nr, Nz});
    }

    // Actualizado: Ahora recibe L_catodo
    void calcular_tensores(double I_arc, double param_hall, double L_catodo, 
                           py::array_t<double> R_matriz, 
                           py::array_t<double> Z_matriz, // Necesitamos Z para saber dónde estamos
                           py::array_t<double> Br_matriz, 
                           py::array_t<double> Bz_matriz) {
        
        double* ptr_R  = (double*) R_matriz.request().ptr;
        double* ptr_Z  = (double*) Z_matriz.request().ptr; // Puntero a Z
        double* ptr_Br = (double*) Br_matriz.request().ptr;
        double* ptr_Bz = (double*) Bz_matriz.request().ptr;

        double* ptr_Jr = (double*) Jr.request().ptr;
        double* ptr_Jt = (double*) Jtheta.request().ptr;
        double* ptr_Fz = (double*) Fz.request().ptr;
        double* ptr_Fr = (double*) Fr.request().ptr;

        for(int i = 0; i < Nr * Nz; i++) {
            double r = ptr_R[i];
            double z = ptr_Z[i];
            
            double j_r = 0.0;
            double j_t = 0.0;


            // Asumimos que la corriente radial macroscópica solo fluye sobre el cuerpo del cátodo
            if (z <= L_catodo && r >= 0.01) { 

                j_r = -I_arc / (2.0 * M_PI * r * L_catodo); 
                j_t = param_hall * j_r;
            } else {

                // En un modelo real, las líneas se curvan hacia la punta aquí.
                j_r = 0.0; 
                j_t = 0.0;
            }

            ptr_Jr[i] = j_r;
            ptr_Jt[i] = j_t;

            ptr_Fz[i] = -j_t * ptr_Br[i]; 
            ptr_Fr[i] = j_t * ptr_Bz[i];  
        }
    }
};





class SimuladorPIC {
public:
    std::vector<IonArgon> iones;
    double Te_eV;
    double masa_ion;
    double carga_e;
    
    double r_catodo, r_anodo, L_catodo, L_anodo, dr_celda;
    int Nr; 

    SimuladorPIC(int num_particulas, double Te_eV_, double r_c, double r_a, double L_c, double L_a, int Nr_) 
        : Te_eV(Te_eV_), r_catodo(r_c), r_anodo(r_a), L_catodo(L_c), L_anodo(L_a), Nr(Nr_) { 
        
        masa_ion = 6.63e-26; 
        carga_e = 1.602e-19; 
        dr_celda = (r_anodo - r_catodo) / (Nr - 1); 
        iones.resize(num_particulas);
    }

    // Añadimos 'r_anodo_base_cm' como segundo parámetro
    void inicializar_particulas(double u_z_inicial_m_s, double r_anodo_base_cm) {

        std::random_device rd;
        std::mt19937 gen(rd());
        


        std::uniform_real_distribution<> dist_r(r_catodo + 0.02, r_anodo_base_cm - 0.05); 
        
        std::uniform_real_distribution<> dist_theta(0.0, 2.0 * M_PI);
        std::uniform_real_distribution<> dist_z(-0.05, 0.0); 


        for (auto& ion : iones) {
            double r_inicial_cm = dist_r(gen);
            double theta_inicial = dist_theta(gen);

            ion.x = (r_inicial_cm / 100.0) * std::cos(theta_inicial);
            ion.y = (r_inicial_cm / 100.0) * std::sin(theta_inicial);
            ion.z = dist_z(gen); 

            ion.vx = 0.0;
            ion.vy = 0.0;
            ion.vz = u_z_inicial_m_s; 
            
            ion.estado = 0;             
            ion.energia_impacto = 0.0;  
        }
    }

    py::array_t<double> obtener_posiciones_rz() {
        auto resultado = py::array_t<double>({ (int)iones.size(), 2 });
        double* ptr = (double*) resultado.request().ptr;
        for (size_t i = 0; i < iones.size(); i++) {
            double r_m = std::sqrt(iones[i].x * iones[i].x + iones[i].y * iones[i].y);
            ptr[i * 2 + 0] = r_m;          
            ptr[i * 2 + 1] = iones[i].z; 
        }
        return resultado;
    }

    py::array_t<int> obtener_estados() {
        auto resultado = py::array_t<int>((int)iones.size());
        int* ptr = (int*) resultado.request().ptr;
        for (size_t i = 0; i < iones.size(); i++) {
            ptr[i] = iones[i].estado;
        }
        return resultado;
    }
    

    void avanzar_paso_temporal_boris(double dt, py::array_t<double> Ez_in, py::array_t<double> Er_in,
                                 py::array_t<double> Bz_in, py::array_t<double> Br_in,
                                 py::array_t<double> lim_pared, double L_sim, int Nz) {
    
    double* ptr_Ez = (double*) Ez_in.request().ptr;
    double* ptr_Er = (double*) Er_in.request().ptr;
    double* ptr_Bz = (double*) Bz_in.request().ptr;
    double* ptr_Br = (double*) Br_in.request().ptr;
    double* ptr_lim = (double*) lim_pared.request().ptr;

    double q_m = carga_e / masa_ion;
    double dr_m = (r_anodo / 100.0) / (Nr - 1);
    double dz_m = (L_sim / 100.0) / (Nz - 1);
    double rc_m = r_catodo / 100.0;
    double L_cat_m = L_catodo / 100.0;

    for (auto& ion : iones) {
        if (ion.estado != 0) continue;

        double r_m = std::sqrt(ion.x * ion.x + ion.y * ion.y) + 1e-9;
        double z_m = ion.z;


        // Guard: en el reservorio (z < 0) no hay campo definido.
        // Sin este check, j se clampea a 0 y los iones leen el campo
        // de Z=0 incorrectamente, matándolos antes de entrar al dominio.
        double Ez = 0.0, Er = 0.0, Bz = 0.0, Br = 0.0;
        if (z_m >= 0.0) {
            int i = std::max(0, std::min(Nr - 1, (int)(r_m / dr_m)));
            int j = std::max(0, std::min(Nz - 1, (int)(z_m / dz_m)));
            int idx = i * Nz + j;
            Ez = ptr_Ez[idx];
            Er = ptr_Er[idx];
            Bz = ptr_Bz[idx];
            Br = ptr_Br[idx];
        }

        double Ex = Er * (ion.x / r_m);
        double Ey = Er * (ion.y / r_m);


        double q_prime = q_m * dt / 2.0;
        
        // Medio empuje eléctrico (v-)
        double v_minus_x = ion.vx + q_prime * Ex;
        double v_minus_y = ion.vy + q_prime * Ey;
        double v_minus_z = ion.vz + q_prime * Ez;

        // Rotación magnética (t)
        double tx = q_prime * (Br * (ion.x / r_m)), ty = q_prime * (Br * (ion.y / r_m)), tz = q_prime * Bz;
        double t_mag2 = tx*tx + ty*ty + tz*tz;
        double sx = 2.0 * tx / (1.0 + t_mag2), sy = 2.0 * ty / (1.0 + t_mag2), sz = 2.0 * tz / (1.0 + t_mag2);

        double v_prime_x = v_minus_x + (v_minus_y * tz - v_minus_z * ty);
        double v_prime_y = v_minus_y + (v_minus_z * tx - v_minus_x * tz);
        double v_prime_z = v_minus_z + (v_minus_x * ty - v_minus_y * tx);

        double v_plus_x = v_minus_x + (v_prime_y * sz - v_prime_z * sy);
        double v_plus_y = v_minus_y + (v_prime_z * sx - v_prime_x * sz);
        double v_plus_z = v_minus_z + (v_prime_x * sy - v_prime_y * sx);

        // Segundo medio empuje eléctrico
        ion.vx = v_plus_x + q_prime * Ex;
        ion.vy = v_plus_y + q_prime * Ey;
        ion.vz = v_plus_z + q_prime * Ez;

        ion.x += ion.vx * dt;
        ion.y += ion.vy * dt;
        ion.z += ion.vz * dt;

        double r_new = std::sqrt(ion.x*ion.x + ion.y*ion.y);
        double z_new = ion.z;
        int j_col = std::max(0, std::min(Nz - 1, (int)(z_new / dz_m))); 
        double lim_anodo = ptr_lim[j_col] / 100.0;                      
        double v2 = ion.vx*ion.vx + ion.vy*ion.vy + ion.vz*ion.vz;
        double energia_eV = (0.5 * masa_ion * v2) / carga_e;


        if (z_new >= (L_sim / 100.0)) {
            ion.estado = 3; // Verde: Escape
        } else if (r_new >= lim_anodo && z_new <= (L_anodo / 100.0)) {
            ion.estado = 4; // Gris: Choca con el Ánodo
        


        } else if (z_new >= 0.0 && z_new <= L_cat_m && r_new <= rc_m) {
            

            if (z_new >= L_cat_m - 0.0001) {
                ion.estado = 2; // Morado: PUNTA
            } else {
                ion.estado = 1; // Rojo: BARRIL
            }
            ion.energia_impacto = energia_eV;
        }
    }
}
};



PYBIND11_MODULE(motor_mpd_cpp, m) {

    m.doc() = "Módulo C++ para el motor AF-MPD";

    // 1. Malla
    py::class_<Malla2D>(m, "Malla2D")

        .def(py::init<int, int, double, double, double, double, bool>(),
             py::arg("Nr"), py::arg("Nz"), py::arg("r_c"), py::arg("r_a"), 
             py::arg("L_c"), py::arg("L_a"), py::arg("campana"))
        .def("obtener_Z", &Malla2D::obtener_Z)
        .def("obtener_R", &Malla2D::obtener_R);

    // 2. Plasma
    py::class_<PlasmaArgon>(m, "PlasmaArgon")
        .def(py::init<int, int>())
        .def("inicializar_condiciones", &PlasmaArgon::inicializar_condiciones)
        .def_readonly("densidad", &PlasmaArgon::densidad)
        .def_readonly("vel_z", &PlasmaArgon::vel_z)
        .def_readonly("temperatura", &PlasmaArgon::temperatura);

    // 3. Campo magnético
        py::class_<CampoMagnetico>(m, "CampoMagnetico")
            .def(py::init<int, int>())

            .def("calcular_campo_aplicado", &CampoMagnetico::calcular_campo_aplicado,
                py::arg("B0"), py::arg("L_catodo"), py::arg("L_anodo"), 
                py::arg("r_catodo"), py::arg("r_anodo"), py::arg("es_campana"))
            .def_readonly("Br", &CampoMagnetico::Br)
            .def_readonly("Bz", &CampoMagnetico::Bz);

    // 4. Fuerza de Lorentz (fluido, opcional mantener)
    py::class_<FuerzaLorentz>(m, "FuerzaLorentz")
        .def(py::init<int, int>())

        .def("calcular_tensores", &FuerzaLorentz::calcular_tensores)
        .def_readwrite("Jr", &FuerzaLorentz::Jr)
        .def_readwrite("Jtheta", &FuerzaLorentz::Jtheta)
        .def_readwrite("Fz", &FuerzaLorentz::Fz)
        .def_readwrite("Fr", &FuerzaLorentz::Fr);


// estoycansao. Simulador PIC de Iones

    py::class_<SimuladorPIC>(m, "SimuladorPIC")
        .def(py::init<int, double, double, double, double, double, int>())
        .def("inicializar_particulas", &SimuladorPIC::inicializar_particulas)
        .def("obtener_posiciones_rz", &SimuladorPIC::obtener_posiciones_rz)
        .def("obtener_estados", &SimuladorPIC::obtener_estados)

        .def("avanzar_paso_temporal_boris", &SimuladorPIC::avanzar_paso_temporal_boris);
}