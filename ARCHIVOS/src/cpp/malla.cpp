#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <cstdlib> // Para generar las posiciones aleatorias iniciales
#include <algorithm>

namespace py = pybind11;


class Malla2D {
public:
    int Nr, Nz;
    double r_catodo, r_anodo, L;
    bool es_campana;

    Malla2D(int Nr_, int Nz_, double r_c, double r_a, double L_, bool campana)
        : Nr(Nr_), Nz(Nz_), r_catodo(r_c), r_anodo(r_a), L(L_), es_campana(campana) {}

    py::array_t<double> obtener_Z() {
        auto resultado = py::array_t<double>({Nr, Nz});
        double* ptr = (double*) resultado.request().ptr;
        double dz = L / (Nz - 1);
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
        double dz = L / (Nz - 1);
        
        for (int j = 0; j < Nz; j++) {
            double z = j * dz;
            double r_out = r_anodo; 
            
            if (es_campana) {

                double a = 4.0;
                r_out = r_anodo * std::pow(1.0 + std::pow(z / a, 2), 0.75);
            }

            double dr = (r_out - r_catodo) / (Nr - 1);
            for (int i = 0; i < Nr; i++) {
                ptr[i * Nz + j] = r_catodo + i * dr;
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

    void calcular_campo_aplicado(double B0, double L, double r_catodo, double r_anodo, bool es_campana) {
        double* ptr_Br = (double*) Br.request().ptr;
        double* ptr_Bz = (double*) Bz.request().ptr;
        double dz = L / (Nz - 1);

        for(int j = 0; j < Nz; j++) {
            double z = j * dz;
            double a = 4.0; 
            
            // Campo de Biot-Savart real
            double Bz_val = B0 / std::pow(1.0 + std::pow(z / a, 2), 1.5);
            
            // Derivada real de Bz respecto a z
            double dBz_dz = -3.0 * B0 * z / (a * a * std::pow(1.0 + std::pow(z / a, 2), 2.5));

            double r_out = r_anodo;
            if (es_campana) {
                r_out = r_anodo * std::pow(1.0 + std::pow(z / a, 2), 0.75);
            }
            double dr = (r_out - r_catodo) / (Nr - 1);

            for(int i = 0; i < Nr; i++) {
                double r = r_catodo + i * dr;
                ptr_Bz[i * Nz + j] = Bz_val;
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

    // Método que calcula F = J x B
    void calcular_tensores(double I_arc, double param_hall, double L, 
                           py::array_t<double> R_matriz, 
                           py::array_t<double> Br_matriz, 
                           py::array_t<double> Bz_matriz) {
        

        double* ptr_R  = (double*) R_matriz.request().ptr;
        double* ptr_Br = (double*) Br_matriz.request().ptr;
        double* ptr_Bz = (double*) Bz_matriz.request().ptr;

        double* ptr_Jr = (double*) Jr.request().ptr;
        double* ptr_Jt = (double*) Jtheta.request().ptr;
        double* ptr_Fz = (double*) Fz.request().ptr;
        double* ptr_Fr = (double*) Fr.request().ptr;

        for(int i = 0; i < Nr * Nz; i++) {
            double r = ptr_R[i];
            

            double j_r = -I_arc / (2.0 * M_PI * r * L);
            

            double j_t = param_hall * j_r;

            ptr_Jr[i] = j_r;
            ptr_Jt[i] = j_t;


            ptr_Fz[i] = -j_t * ptr_Br[i]; 
            

            ptr_Fr[i] = j_t * ptr_Bz[i];  
        }
    }
};


//IGNORAR TODO ESTA CLASE, NO FUNCIONA POR MÁS QUE TRATÉ
class RastreadorIones {
public:
    int N, Nr, Nz;
    py::array_t<double> Rp, Zp, Vr, Vz, Vtheta;
    py::array_t<double> Zp_anterior;
    py::array_t<int> Estado;

    RastreadorIones(int N_, int Nr_, int Nz_, double r_c, double r_a_max)
        : N(N_), Nr(Nr_), Nz(Nz_) {

        Rp = py::array_t<double>(N);
        Zp = py::array_t<double>(N);
        Zp_anterior = py::array_t<double>(N);
        Vr = py::array_t<double>(N);
        Vz = py::array_t<double>(N);
        Vtheta = py::array_t<double>(N);
        Estado = py::array_t<int>(N);

        auto buf_R = Rp.mutable_unchecked();
        auto buf_Z = Zp.mutable_unchecked();
        auto buf_Z_ant = Zp_anterior.mutable_unchecked();
        auto buf_Vr = Vr.mutable_unchecked();
        auto buf_Vz = Vz.mutable_unchecked();
        auto buf_Vth = Vtheta.mutable_unchecked();
        auto buf_E = Estado.mutable_unchecked();

        srand(42);



        double L = 4.455;
        
        double r_min = r_c + 0.1;
        double r_max = r_c + 1.0; 

        for (int k = 0; k < N; k++) {
            double u = (double)rand() / RAND_MAX;
            

            if (k % 2 == 0) {

                buf_Z(k) = 0.05;
                buf_R(k) = std::sqrt(r_min * r_min + u * (r_max * r_max - r_min * r_min));
            } else {

                buf_Z(k) = L + 0.1; 
                buf_R(k) = u * r_c; 
            }
            
            buf_Vr(k) = ((double)rand() / RAND_MAX) * 100.0;
            buf_Vz(k) = 6500.0 + ((double)rand() / RAND_MAX) * 4000.0;
            buf_Vth(k) = ((double)rand() / RAND_MAX) * 500.0;
            buf_E(k) = 0;
        }
    }

    void calcular_paso(double dt,
                       py::array_t<double> Ez_mat,
                       py::array_t<double> Er_mat,
                       py::array_t<double> Bz_mat,
                       py::array_t<double> Br_mat,
                       py::array_t<double> R_pared_arr, 
                       double r_c, double r_a_max, double L) {

        auto R = Rp.mutable_unchecked();
        auto Z = Zp.mutable_unchecked();
        auto Z_ant = Zp_anterior.mutable_unchecked();
        auto Vr_ = Vr.mutable_unchecked();
        auto Vz_ = Vz.mutable_unchecked();
        auto Vth_ = Vtheta.mutable_unchecked();
        auto E = Estado.mutable_unchecked();

        auto Ez = Ez_mat.unchecked<2>();
        auto Er = Er_mat.unchecked<2>();
        auto Bz = Bz_mat.unchecked<2>();
        auto Br = Br_mat.unchecked<2>();
        auto R_pared = R_pared_arr.unchecked<1>(); 

        const double q = 1.602e-19;
        const double m = 6.63e-26; 
        const double q_m = q / m;
        const double mu_0 = 1.25663706e-6; 

        double I_arc = -222.0; 
        double L_sim = 6.5;
        double r_min_malla = 0.001; 

        for (int k = 0; k < N; k++) {
            if (E(k) != 0) continue;

            double z_previo = Z(k);
            
            double r_cm = R(k);
            double z_cm = Z(k);

            double r_m = r_cm / 100.0;
            if (r_m < 1e-6) r_m = 1e-6; 

            int j = (int)round((z_cm / L_sim) * (Nz - 1));
            if (j < 0) j = 0;
            if (j >= Nz) j = Nz - 1;

            double ratio_r = (r_cm - r_min_malla) / (r_a_max - r_min_malla);
            int i = (int)round(ratio_r * (Nr - 1));
            if (i < 0) i = 0;
            if (i >= Nr) i = Nr - 1;

            double limite_anodo_actual = R_pared(j);

            double Er_val = Er(i, j);
            double Ez_val = Ez(i, j);
            double Br_val = Br(i, j);
            double Bz_val = Bz(i, j);

            double Btheta = (mu_0 * I_arc) / (2.0 * M_PI * r_m);

            double vr = Vr_(k);
            double vz = Vz_(k);
            double vth = Vth_(k);

            double term_r = Er_val + vth * Bz_val - vz * Btheta;
            double term_z = Ez_val + vr * Btheta - vth * Br_val;
            double term_th = vz * Br_val - vr * Bz_val;


            double ar = q_m * term_r + (vth * vth) / r_m;
            double az = q_m * term_z;
            double ath = q_m * term_th - (vr * vth) / r_m;

            Vr_(k) += ar * dt;
            Vz_(k) += az * dt;
            Vth_(k) += ath * dt;

            Z(k) += Vz_(k) * dt * 100.0;
            R(k) += Vr_(k) * dt * 100.0;

            if (R(k) < 0.0) {
                R(k) = -R(k);
                Vr_(k) = -Vr_(k);
            }


            
            if (Z(k) >= L_sim) {
                E(k) = 1;  // Escape
                continue;
            }
            

            if (Vz_(k) < 0 && Z(k) > L - 0.3 && R(k) <= r_c * 1.15) {
                E(k) = -2;
                Z(k) = L;
                R(k) = std::min(R(k), r_c);
                continue;
            }
            
            if (z_previo > L && Z(k) <= L && R(k) <= r_c * 1.15) {
                E(k) = -2;
                Z(k) = L;
                R(k) = std::min(R(k), r_c);
                continue;
            }
            
            // EROSIÓN LATERAL
            if (Z(k) < L && R(k) <= r_c) {
                E(k) = -1;
                R(k) = r_c;
                continue;
            }
            
            // ÁNODO
            if (R(k) >= limite_anodo_actual) { 
                R(k) = limite_anodo_actual - 0.01;
                if (Vr_(k) > 0) Vr_(k) *= -0.2; 
            }
            
            Z_ant(k) = z_previo;
        }
    }
};


//ESTA CLASE NECESITA MEJORES URGENTES PUES NO LOGRÉ MODELAR DE FORMA EFECTIVA LA FÍSICA EN LA PUNTA DEL CÁTODO :(
class SimuladorIonesMPD {
public:
    int N, Nr, Nz;
    py::array_t<double> Rp, Zp, Vr, Vz, Vtheta;
    py::array_t<int> Estado;

    SimuladorIonesMPD(int N_, int Nr_, int Nz_, double r_c, double r_a_base, double L)
        : N(N_), Nr(Nr_), Nz(Nz_) {

        Rp = py::array_t<double>(N);
        Zp = py::array_t<double>(N);
        Vr = py::array_t<double>(N);
        Vz = py::array_t<double>(N);
        Vtheta = py::array_t<double>(N);
        Estado = py::array_t<int>(N);

        auto buf_R = Rp.mutable_unchecked();
        auto buf_Z = Zp.mutable_unchecked();
        auto buf_Vr = Vr.mutable_unchecked();
        auto buf_Vz = Vz.mutable_unchecked();
        auto buf_Vth = Vtheta.mutable_unchecked();
        auto buf_E = Estado.mutable_unchecked();

        srand(42); 

        double margen = 0.1; 
        double r_min = r_c + margen;
        double r_max = r_a_base - margen;

        for (int k = 0; k < N; k++) {
            double u_r = (double)rand() / RAND_MAX;
            double u_z = (double)rand() / RAND_MAX;
            
            buf_R(k) = r_min + u_r * (r_max - r_min);
            buf_Z(k) = 0.5 + u_z * (L - 1.0); 
            
            buf_Vr(k) = (((double)rand() / RAND_MAX) - 0.5) * 500.0; 
            buf_Vz(k) = 2000.0 + ((double)rand() / RAND_MAX) * 2000.0; 
            buf_Vth(k) = 500.0 + ((double)rand() / RAND_MAX) * 1000.0; 
            
            buf_E(k) = 0; 
        }
    }

    void calcular_paso(double dt,
                       py::array_t<double> Ez_mat,
                       py::array_t<double> Er_mat,
                       py::array_t<double> Bz_mat,
                       py::array_t<double> Br_mat,
                       py::array_t<double> R_pared_arr, 
                       double r_c, double r_a_max, double L) {

        auto R = Rp.mutable_unchecked();
        auto Z = Zp.mutable_unchecked();
        auto Vr_ = Vr.mutable_unchecked();
        auto Vz_ = Vz.mutable_unchecked();
        auto Vth_ = Vtheta.mutable_unchecked();
        auto E = Estado.mutable_unchecked();

        auto Ez = Ez_mat.unchecked<2>();
        auto Er = Er_mat.unchecked<2>();
        auto Bz = Bz_mat.unchecked<2>();
        auto Br = Br_mat.unchecked<2>();
        auto R_pared = R_pared_arr.unchecked<1>(); 

        const double q = 1.602e-19;
        const double m = 6.63e-26; 
        const double q_m = q / m;
        const double mu_0 = 1.25663706e-6; 

        double I_arc = -222.0; 
        double L_sim = 6.5;

        for (int k = 0; k < N; k++) {
            if (E(k) != 0) continue; 

            double z_previo = Z(k);
            double r_cm = R(k);
            double z_cm = Z(k);
            double r_m = r_cm / 100.0; 
            if (r_m < 1e-6) r_m = 1e-6; 

            int j = (int)round((z_cm / L_sim) * (Nz - 1));
            j = std::clamp(j, 0, Nz - 1);
            
            double ratio_r = r_cm / r_a_max;
            int i = (int)round(ratio_r * (Nr - 1));
            i = std::clamp(i, 0, Nr - 1);

            double Er_val = Er(i, j);
            double Ez_val = Ez(i, j);
            double Br_val = Br(i, j);
            double Bz_val = Bz(i, j);

            double Btheta = (mu_0 * I_arc) / (2.0 * M_PI * r_m);


            if (z_cm >= L - 0.05 && z_cm < L + 0.15 && r_cm <= r_c * 1.5) {
                double lambda_num = 0.05; 
                double delta_phi = 15.0;  
                

                if (z_cm >= L) {
                    double dist_z = z_cm - L;
                    double Ez_sheath = -(delta_phi / lambda_num) * exp(-dist_z / lambda_num) * 100.0; 
                    Ez_val += Ez_sheath; 
                }


                if (r_cm > r_c) {
                    double dist_r = r_cm - r_c;

                    double Er_sheath = -(delta_phi / lambda_num) * exp(-dist_r / lambda_num) * 100.0;
                    Er_val += Er_sheath;
                }
            }


            double vr = Vr_(k);
            double vz = Vz_(k);
            double vth = Vth_(k);


            double F_r_Lorentz  = Er_val + vth * Bz_val - vz * Btheta;
            double F_z_Lorentz  = Ez_val + vr * Btheta - vth * Br_val;
            double F_th_Lorentz = vz * Br_val - vr * Bz_val;

            double ar  = q_m * F_r_Lorentz  + (vth * vth) / r_m;
            double az  = q_m * F_z_Lorentz;
            double ath = q_m * F_th_Lorentz - (vr * vth) / r_m;


            double altura_succion = 1.0; 
            double rango_z = 0.2; 
            

            if (z_cm > L - rango_z && z_cm < L + rango_z && r_cm <= altura_succion) {

                

                az = (L - z_cm) * 2.0e12; 
                

                ar = -5.0e11; 
                

                ath = -Vth_(k) / dt; 
            }

            Vr_(k)  += ar * dt;
            Vz_(k)  += az * dt;
            Vth_(k) += ath * dt;

            Z(k) += Vz_(k) * dt * 100.0; 
            R(k) += Vr_(k) * dt * 100.0;

            if (R(k) < 0.0) {
                R(k) = -R(k);
                Vr_(k) = -Vr_(k);
            }
            

            if (Z(k) >= L_sim) {
                E(k) = 1; 
                continue;
            }
            

            if (Z(k) >= L && Z(k) <= 6.5 && R(k) <= 0.6) {
                E(k) = -2; 
                Z(k) = L;  
                

                R(k) = ((double)rand() / RAND_MAX) * r_c;
                continue;
            }
            

            if (Z(k) < L && R(k) <= r_c) {
                E(k) = -1; 
                R(k) = r_c;
                continue;
            }
            

            double limite_anodo_actual = R_pared(j);
            if (R(k) >= limite_anodo_actual) { 
                E(k) = 2; 
                R(k) = limite_anodo_actual;
            }
        }
    }
};


PYBIND11_MODULE(motor_mpd_cpp, m) {

    // 1. Malla
    py::class_<Malla2D>(m, "Malla2D")
        .def(py::init<int, int, double, double, double, bool>())
        .def("obtener_Z", &Malla2D::obtener_Z)
        .def("obtener_R", &Malla2D::obtener_R)
        .def_readonly("Nr", &Malla2D::Nr)
        .def_readonly("Nz", &Malla2D::Nz);

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
        .def("calcular_campo_aplicado", &CampoMagnetico::calcular_campo_aplicado)
        .def_readonly("Br", &CampoMagnetico::Br)
        .def_readonly("Bz", &CampoMagnetico::Bz);

    // 4. Fuerza de Lorentz (fluido, opcional mantener)
    py::class_<FuerzaLorentz>(m, "FuerzaLorentz")
        .def(py::init<int, int>(), py::arg("Nr"), py::arg("Nz"))
        .def("calcular_tensores", &FuerzaLorentz::calcular_tensores,
             py::arg("I_arc"), py::arg("param_hall"), py::arg("L"),
             py::arg("R_matriz"), py::arg("Br_matriz"), py::arg("Bz_matriz"))
        .def_readonly("Jr", &FuerzaLorentz::Jr)
        .def_readonly("Jtheta", &FuerzaLorentz::Jtheta)
        .def_readonly("Fz", &FuerzaLorentz::Fz)
        .def_readonly("Fr", &FuerzaLorentz::Fr);

// 5. Rastreador de iones (NO FUNCIONA, NO APORTA EN NADA)
    py::class_<RastreadorIones>(m, "RastreadorIones")
        .def(py::init<int, int, int, double, double>())
        .def("calcular_paso", &RastreadorIones::calcular_paso,
             py::arg("dt"),
             py::arg("Ez_mat"),
             py::arg("Er_mat"),
             py::arg("Bz_mat"),
             py::arg("Br_mat"),
             py::arg("R_pared_arr"), 
             py::arg("r_c"),
             py::arg("r_a_max"),   
             py::arg("L"))
        .def_readonly("Rp", &RastreadorIones::Rp)
        .def_readonly("Zp", &RastreadorIones::Zp)
        .def_readonly("Vtheta", &RastreadorIones::Vtheta)
        .def_readonly("Estado", &RastreadorIones::Estado)
        .def_readwrite("Vz", &RastreadorIones::Vz);


// 6. Es el de la simulación xd, no logré terminar de ajustarlo :c
    py::class_<SimuladorIonesMPD>(m, "SimuladorIonesMPD")
        .def(py::init<int, int, int, double, double, double>())
        .def("calcular_paso", &SimuladorIonesMPD::calcular_paso)
        .def_readwrite("Rp", &SimuladorIonesMPD::Rp)
        .def_readwrite("Zp", &SimuladorIonesMPD::Zp)
        .def_readwrite("Vr", &SimuladorIonesMPD::Vr)
        .def_readwrite("Vz", &SimuladorIonesMPD::Vz)
        .def_readwrite("Vtheta", &SimuladorIonesMPD::Vtheta)
        .def_readwrite("Estado", &SimuladorIonesMPD::Estado);
}