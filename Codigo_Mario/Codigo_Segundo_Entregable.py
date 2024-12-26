#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creado el Miércoles 25 de diciembre de 2024

@autor: mariomanzano
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import os

# Rutas de los archivos
archivo_5000 = "/Users/mariomanzano/Desktop/Máster en Astrofísica ULL/01 - Primer cuatrimestre/Atmósferas Estelares/Entregables/Segundo Entregable/t5000.dat"
archivo_8000 = "/Users/mariomanzano/Desktop/Máster en Astrofísica ULL/01 - Primer cuatrimestre/Atmósferas Estelares/Entregables/Segundo Entregable/t8000.dat"

def procesar_archivo(archivo):
    with open(archivo, 'r') as file:
        lineas = file.readlines()

    for indice in range(len(lineas)):
        if lineas[indice] == "Model structure\n":
            indice_tabla = indice + 1
            break

    datos = [linea.split() for linea in lineas[indice_tabla+1:]]
    nombres_columnas = lineas[indice_tabla].split()

    df_estructura_modelo = pd.DataFrame(datos, columns=nombres_columnas, index=None)
    return {
        'logtauR': np.array(df_estructura_modelo['lgTauR'], dtype=float),
        'profundidad': np.array(df_estructura_modelo['Depth'], dtype=float),
        'temperatura': np.array(df_estructura_modelo['T'], dtype=float),
        'presion_elec': np.array(df_estructura_modelo['Pe'], dtype=float),
        'presion_gas': np.array(df_estructura_modelo['Pg'], dtype=float),
        'presion_rad': np.array(df_estructura_modelo['Prad'], dtype=float)
    }

# Procesar los archivos
modelo_5000 = procesar_archivo(archivo_5000)
modelo_8000 = procesar_archivo(archivo_8000)

# Colores ajustados
color_5000 = "#1f9ac9"  # Azul claro o cian para 5000 K
color_8000 = "#5c068c"  # Morado para 8000 K

# Crear directorio para guardar las figuras
if not os.path.exists('Figuras'):
    os.makedirs('Figuras')

# Tamaño de la fuente de la leyenda
tamaño_fuente_leyenda = 14

# %%

# =============================================================================
# Punto 1: dibujar lgtaur frente a r
# =============================================================================

# 1. Profundidad geométrica vs log(τR)
plt.figure(figsize=(10, 8))
plt.plot(modelo_5000['profundidad'], modelo_5000['logtauR'], 'o-', color=color_5000, label='$T_{\mathrm{eff}} = 5000 \, \mathrm{K}$')  # Círculos con línea
plt.plot(modelo_8000['profundidad'], modelo_8000['logtauR'], 's-', color=color_8000, label='$T_{\mathrm{eff}} = 8000 \, \mathrm{K}$')  # Cuadrados con línea
plt.xlabel('$r$ [cm]')
plt.ylabel(r'$\log(\tau_R)$')
plt.legend(fontsize=tamaño_fuente_leyenda)
plt.grid(True)
plt.savefig('Figuras/profundidad_logtauR.pdf')



# %%

# =============================================================================
# Punto 2: dibujar T, Pe, Pe/Pg y Prad/Pg frente a lgtaur
# =============================================================================

# 2. log(τR) vs Temperatura
plt.figure(figsize=(10, 8))
plt.plot(modelo_5000['logtauR'], modelo_5000['temperatura'], 'o-', color=color_5000, label='$T_{\mathrm{eff}} = 5000 \, \mathrm{K}$')  # Círculos con línea
plt.plot(modelo_8000['logtauR'], modelo_8000['temperatura'], 's-', color=color_8000, label='$T_{\mathrm{eff}} = 8000 \, \mathrm{K}$')  # Cuadrados con línea
plt.xlabel(r'$\log(\tau_R)$')
plt.ylabel('$T$ [K]')
plt.legend(fontsize=tamaño_fuente_leyenda)
plt.grid(True)
plt.savefig('Figuras/logtauR_temperatura.pdf')

# 3. log(τR) vs Presión electrónica
plt.figure(figsize=(10, 8))
plt.plot(modelo_5000['logtauR'], modelo_5000['presion_elec'], 'o-', color=color_5000, label='$T_{\mathrm{eff}} = 5000 \, \mathrm{K}$')  # Círculos con línea
plt.plot(modelo_8000['logtauR'], modelo_8000['presion_elec'], 's-', color=color_8000, label='$T_{\mathrm{eff}} = 8000 \, \mathrm{K}$')  # Cuadrados con línea
plt.xlabel(r'$\log(\tau_R)$')
plt.ylabel('$P_e$ [dyn cm$^{-2}$]')
plt.legend(fontsize=tamaño_fuente_leyenda)
plt.grid(True)
plt.savefig('Figuras/logtauR_presion_electronica.pdf')

# 4. log(τR) vs $P_e / P_g$
plt.figure(figsize=(10, 8))
plt.plot(modelo_5000['logtauR'], modelo_5000['presion_elec'] / modelo_5000['presion_gas'], 'o-', color=color_5000, label='$T_{\mathrm{eff}} = 5000 \, \mathrm{K}$')  # Círculos con línea
plt.plot(modelo_8000['logtauR'], modelo_8000['presion_elec'] / modelo_8000['presion_gas'], 's-', color=color_8000, label='$T_{\mathrm{eff}} = 8000 \, \mathrm{K}$')  # Cuadrados con línea
plt.xlabel(r'$\log(\tau_R)$')
plt.ylabel(r'$P_e / P_g$')
plt.legend(fontsize=tamaño_fuente_leyenda)
plt.grid(True)
plt.savefig('Figuras/logtauR_presion_relativa.pdf')

# 5. log(τR) vs $P_{rad} / P_g$
plt.figure(figsize=(10, 8))
plt.plot(modelo_5000['logtauR'], modelo_5000['presion_rad'] / modelo_5000['presion_gas'], 'o-', color=color_5000, label='$T_{\mathrm{eff}} = 5000 \, \mathrm{K}$')  # Círculos con línea
plt.plot(modelo_8000['logtauR'], modelo_8000['presion_rad'] / modelo_8000['presion_gas'], 's-', color=color_8000, label='$T_{\mathrm{eff}} = 8000 \, \mathrm{K}$')  # Cuadrados con línea
plt.xlabel(r'$\log(\tau_R)$')
plt.ylabel(r'$P_{rad} / P_g$')
plt.legend(fontsize=tamaño_fuente_leyenda)
plt.grid(True)
plt.savefig('Figuras/logtauR_presion_radiacion.pdf')

# %%

# =============================================================================
# Punto 3: Dibujar T frente a T_{gris} en gráficas separadas para cada modelo
# =============================================================================

# Función para calcular T_{gris} con la ecuación ajustada
def calcular_T_gris_ajustada(logtauR, T_ef):
    tau = 10**logtauR  # Convertir log(τ_R) a τ_R
    return T_ef * ((3 / 4) * (tau + 2 / 3))**0.25

# Calcular T_{gris} ajustada para ambos modelos
T_gris_5000 = calcular_T_gris_ajustada(modelo_5000['logtauR'], 5000)
T_gris_8000 = calcular_T_gris_ajustada(modelo_8000['logtauR'], 8000)

# Color para T_gris
color_T_gris = "#8B0000"  # Rojo oscuro

# Gráfica: Temperatura frente a T_{gris} para 5000 K
plt.figure(figsize=(10, 8))
plt.plot(modelo_5000['logtauR'], modelo_5000['temperatura'], 'o-', color=color_5000, label='$T_{\mathrm{modelo}} \, (5000 \, \mathrm{K})$')  # T modelo 5000 K
plt.plot(modelo_5000['logtauR'], T_gris_5000, '--', color=color_T_gris, label='$T_{\mathrm{gris}} \, (5000 \, \mathrm{K})$')  # T gris 5000 K
plt.xlabel(r'$\log(\tau_R)$')
plt.ylabel('$T$ [K]')
plt.title('$T_{\mathrm{eff}} = 5000 \, \mathrm{K}$')
plt.legend(fontsize=tamaño_fuente_leyenda)
plt.grid(True)
plt.savefig('Figuras/logtauR_T_vs_Tgris_5000.pdf')

# Gráfica: Temperatura frente a T_{gris} para 8000 K
plt.figure(figsize=(10, 8))
plt.plot(modelo_8000['logtauR'], modelo_8000['temperatura'], 's-', color=color_8000, label='$T_{\mathrm{modelo}} \, (8000 \, \mathrm{K})$')  # T modelo 8000 K
plt.plot(modelo_8000['logtauR'], T_gris_8000, '--', color=color_T_gris, label='$T_{\mathrm{gris}} \, (8000 \, \mathrm{K})$')  # T gris 8000 K
plt.xlabel(r'$\log(\tau_R)$')
plt.ylabel('$T$ [K]')
plt.title('$T_{\mathrm{eff}} = 8000 \, \mathrm{K}$')
plt.legend(fontsize=tamaño_fuente_leyenda)
plt.grid(True)
plt.savefig('Figuras/logtauR_T_vs_Tgris_8000.pdf')