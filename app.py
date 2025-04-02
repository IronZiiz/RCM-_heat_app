import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# Função para montar as matrizes do método de Crank-Nicolson

def crank_nicolson_matrices(T, alpha, r, dr, dt, N):
    A = np.zeros((N+1, N+1))
    B = np.zeros((N+1, N+1))
    
    # Preenchimento para pontos internos (i = 1 até N-1)
    for i in range(1, N):
        a = alpha[i] * dt / (2 * dr**2)
        # Para evitar divisão por zero no centro
        b = alpha[i] * dt / (4 * r[i] * dr) if r[i] != 0 else 0  
        A[i, i-1] = -a + b
        A[i, i]   = 1 + 2*a
        A[i, i+1] = -a - b

        B[i, i-1] = a - b
        B[i, i]   = 1 - 2*a
        B[i, i+1] = a + b

    # Condição no centro (i=0)
    A[0, 0] = 1 + 4 * alpha[0] * dt / dr**2
    A[0, 1] = -4 * alpha[0] * dt / dr**2
    B[0, 0] = 1 - 4 * alpha[0] * dt / dr**2
    B[0, 1] = 4 * alpha[0] * dt / dr**2

    # Condição na borda (i=N) - temperatura fixa
    A[N, N] = 1  
    B[N, N] = 1
    return A, B

# Função que realiza a simulação
def run_simulation(Dc, Rs, Ra, kc, rho_c, cp_c, 
                   ks, rho_s, cp_s, 
                   ka, rho_a, cp_a, 
                   Tf, Ta, T0, dt, max_time, N):
    # Definindo o raio do fio de cobre e o domínio total
    Rc = (Dc - 2*Rs)/2
    r_total = Rc + Rs + Ra
    dr = r_total / N
    r = np.linspace(0, r_total, N+1)

    # Índices das interfaces
    i_c = int(Rc / dr)
    i_s = int((Rc + Rs) / dr)

    # Difusividades térmicas
    alpha_cobre = kc / (rho_c * cp_c)
    alpha_silicone = ks / (rho_s * cp_s)
    alpha_ar = ka / (rho_a * cp_a)
    alpha = np.zeros(N+1)
    alpha[:i_c] = alpha_cobre          # Região do cobre
    alpha[i_c:i_s] = alpha_silicone    # Região do silicone
    alpha[i_s:] = alpha_ar             # Região do ar

    # Condição inicial
    T = np.full(N+1, T0)
    T[-1] = Tf  # Condição de contorno na borda (forno)

    # Montar as matrizes para o método Crank-Nicolson
    A, B = crank_nicolson_matrices(T, alpha, r, dr, dt, N)

    # Listas para armazenar a evolução da temperatura na interface
    times = []
    T_interface = []
    t = 0

    while t < max_time:
        T_new = spsolve(A, B.dot(T))
        T_new[-1] = Tf  # Garante a condição de contorno
        
        times.append(t)
        T_interface.append(T_new[i_c])
        
        if T_new[i_c] >= Ta:
            break
        T = T_new
        t += dt

    return times, T_interface, t

# Interface Streamlit
st.title("Simulação de Evolução da Temperatura na Interface Cobre-Silicone")

st.sidebar.header("Dados dos Materiais")

# Dados geométricos
Dc = st.sidebar.number_input("Diâmetro do fio de cobre (m)", value=2.48/1000, format="%.6f")
Rs = st.sidebar.number_input("Espessura da camada de silicone (m)", value=0.8/1000, format="%.6f")
Ra = st.sidebar.number_input("Espessura da camada de ar (m)", value=15/100, format="%.6f")

# Propriedades do cobre
kc = st.sidebar.number_input("Condutividade térmica do cobre (W/m.K)", value=401.0)
rho_c = st.sidebar.number_input("Densidade do cobre (kg/m³)", value=8900.0)
cp_c = st.sidebar.number_input("Calor específico do cobre (J/kg.K)", value=385.0)

# Propriedades do silicone
ks = st.sidebar.number_input("Condutividade térmica do silicone (W/m.K)", value=0.5)
rho_s = st.sidebar.number_input("Densidade do silicone (kg/m³)", value=1200.0)
cp_s = st.sidebar.number_input("Calor específico do silicone (J/kg.K)", value=1100.0)

# Propriedades do ar
ka = st.sidebar.number_input("Condutividade térmica do ar (W/m.K)", value=0.026)
rho_a = st.sidebar.number_input("Densidade do ar (kg/m³)", value=1.225)
cp_a = st.sidebar.number_input("Calor específico do ar (J/kg.K)", value=287.058)

st.sidebar.header("Parâmetros do Problema")
Tf = st.sidebar.number_input("Temperatura do forno (°C)", value=200.0)
Ta = st.sidebar.number_input("Temperatura desejada na interface (°C)", value=100.0)
T0 = st.sidebar.number_input("Temperatura inicial (°C)", value=30.0)

dt = st.sidebar.number_input("Passo de tempo (s)", value=0.01, format="%.4f")
max_time = st.sidebar.number_input("Tempo máximo de simulação (s)", value=10000.0)
N = st.sidebar.number_input("Número de pontos radiais", value=200, step=1)

if st.sidebar.button("Executar Simulação"):
    times, T_interface, final_time = run_simulation(Dc, Rs, Ra, kc, rho_c, cp_c, 
                                                     ks, rho_s, cp_s, 
                                                     ka, rho_a, cp_a, 
                                                     Tf, Ta, T0, dt, max_time, int(N))
    # Exibição do resultado
    if final_time < max_time:
        st.write(f"Tempo necessário para atingir {Ta}°C na interface: {final_time:.2f} segundos")
    else:
        st.write("Tempo máximo atingido sem alcançar a temperatura desejada.")
    
    # Criação do gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, T_interface, label=f"Temperatura na interface (Diâmetro do cobre = {Dc} m)")
    ax.axhline(y=Ta, color='r', linestyle='--', label=f"Temperatura alvo ({Ta}°C)")
    ax.axhline(y=T0, color='r', linestyle='--', label=f"Temperatura inicial ({T0}°C)")
    ax.axvline(x=final_time, color='g', linestyle='--', label=f"Tempo atingido ({final_time:.2f} s)")
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Temperatura (°C)")
    ax.set_title("Evolução da Temperatura na Interface Cobre-Silicone")
    ax.legend()
    ax.grid(True)
    
    st.pyplot(fig)