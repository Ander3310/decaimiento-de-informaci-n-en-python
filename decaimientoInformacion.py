import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Parámetros del modelo ---
lmbda = 0.3       # Constante de decaimiento (λ): qué tan rápido pierde relevancia la noticia
R0 = [100]        # Relevancia inicial de la noticia (R₀)
t_inicio = 0      # Tiempo inicial (en unidades arbitrarias, ej. días)
t_fin = 20        # Tiempo final
t_eval = np.linspace(t_inicio, t_fin, 200)  # 200 puntos de evaluación

# --- Definición de la ecuación diferencial dR/dt = -λR ---
def modelo(t, R):
    dRdt = -lmbda * R[0]  # Tasa de cambio de la relevancia
    return [dRdt]

# --- Solución numérica con solve_ivp ---
sol = solve_ivp(modelo, [t_inicio, t_fin], R0, t_eval=t_eval)

# --- Solución analítica: R(t) = R₀ · e^(-λt) ---
R_analitica = R0[0] * np.exp(-lmbda * t_eval)

# --- Gráfica de ambas soluciones ---
plt.figure(figsize=(8, 5))
plt.plot(sol.t, sol.y[0], label='Solución numérica (solve_ivp)')
plt.plot(t_eval, R_analitica, '--', label='Solución analítica')
plt.title('Decaimiento de la relevancia de una noticia')
plt.xlabel('Tiempo')
plt.ylabel('Relevancia R(t)')
plt.legend()
plt.grid(True)
plt.show()