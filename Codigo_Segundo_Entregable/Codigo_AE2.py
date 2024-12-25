#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 13:12:19 2024

@author: mariomanzano
"""

import numpy as np
import matplotlib.pyplot as plt

# Cargar los datos (asegúrate de que las rutas sean correctas)
datos_5000 = np.loadtxt('/Users/mariomanzano/Desktop/Máster en Astrofísica ULL/01 - Primer cuatrimestre/Atmósferas Estelares/Entregables/Segundo Entregable/t5000.dat', comments='#', skiprows=47)
datos_8000 = np.loadtxt('/Users/mariomanzano/Desktop/Máster en Astrofísica ULL/01 - Primer cuatrimestre/Atmósferas Estelares/Entregables/Segundo Entregable/t8000.dat', comments='#', skiprows=47)

# Extraer columnas relevantes (ajustadas según tu captura)
lgtaur_5000 = datos_5000[:, 1]  # Segunda columna: log(τ_R)
r_5000 = datos_5000[:, 2]       # Tercera columna: r

lgtaur_8000 = datos_8000[:, 1]  # Segunda columna: log(τ_R)
r_8000 = datos_8000[:, 2]       # Tercera columna: r

# Graficar
plt.figure(figsize=(10, 6))
plt.plot(r_5000, lgtaur_5000, label='Modelo 5000 K', marker='o')
plt.plot(r_8000, lgtaur_8000, label='Modelo 8000 K', marker='x')

# Configuración del gráfico
plt.xlabel('Profundidad geométrica r (unidades del modelo)')
plt.ylabel('Logaritmo de la profundidad óptica (log(τ_R))')
plt.title('Log(τ_R) frente a r')
plt.legend()
plt.grid(True)

# Guardar y mostrar el gráfico
plt.savefig('logtaur_vs_r.png', dpi=300)
plt.show()