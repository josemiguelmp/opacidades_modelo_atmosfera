#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creado el Miércoles 25 de diciembre de 2024

@autor: mariomanzano
"""


from scipy.optimize import fsolve
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


# Leemos el primer archivo (T = 5000 K)

file_name_1 = "/Users/mariomanzano/Desktop/Máster en Astrofísica ULL/01 - Primer cuatrimestre/Atmósferas Estelares/Entregables/Segundo Entregable/t5000.dat"
Teff_1 = 5000
log_g_1 = log_g_2 = 4.5


with open(file_name_1, 'r') as file:
    lines_1 = file.readlines()
    

for index in range(len(lines_1)):
    if lines_1[index] == "Model structure\n":
        index_table = index + 1
        break

data_1 = [line.split() for line in lines_1[index_table+1:]]
column_names_1 = lines_1[index_table].split()

df_model_structure_1 = pd.DataFrame(data_1, columns=column_names_1, index=None)

numeral_1      = np.array(df_model_structure_1['k'],      dtype=float)            # numeral del punto
logtauR_1      = np.array(df_model_structure_1['lgTauR'], dtype=float)            # logaritmo de la profundidad óptica de Rosseland
logtau_5000_1  = np.array(df_model_structure_1['lgTau5'], dtype=float)            # logaritmo de la profundidad óptica a 5000 A
r_1            = np.array(df_model_structure_1['Depth'],  dtype=float)            # profundidad geométrica
T_1            = np.array(df_model_structure_1['T'],      dtype=float)            # Temperatura
Pe_1           = np.array(df_model_structure_1['Pe'],     dtype=float)            # Presión electrónica
Pg_1           = np.array(df_model_structure_1['Pg'],     dtype=float)            # Presión del gas
Prad_1         = np.array(df_model_structure_1['Prad'],   dtype=float)            # Presión de radiación



# Leemos el segundo archivo (T = 5000 K)

file_name_2 = "/Users/mariomanzano/Desktop/Máster en Astrofísica ULL/01 - Primer cuatrimestre/Atmósferas Estelares/Entregables/Segundo Entregable/t8000.dat"
Teff_2 = 8000

with open(file_name_2, 'r') as file:
    lines_2 = file.readlines()
    

for index in range(len(lines_2)):
    if lines_2[index] == "Model structure\n":
        index_table = index + 1
        break

data_2 = [line.split() for line in lines_2[index_table+1:]]
column_names_2 = lines_2[index_table].split()

df_model_structure_2 = pd.DataFrame(data_2, columns=column_names_2, index=None)

numeral_2      = np.array(df_model_structure_2['k'],      dtype=float)            # numeral del punto
logtauR_2      = np.array(df_model_structure_2['lgTauR'], dtype=float)            # logaritmo de la profundidad óptica de Rosseland
logtau_5000_2  = np.array(df_model_structure_2['lgTau5'], dtype=float)            # logaritmo de la profundidad óptica a 5000 A
r_2            = np.array(df_model_structure_2['Depth'],  dtype=float)            # profundidad geométrica
T_2            = np.array(df_model_structure_2['T'],      dtype=float)            # Temperatura
Pe_2           = np.array(df_model_structure_2['Pe'],     dtype=float)            # Presión electrónica
Pg_2           = np.array(df_model_structure_2['Pg'],     dtype=float)            # Presión del gas
Prad_2         = np.array(df_model_structure_2['Prad'],   dtype=float)            # Presión de radiación


# %%
# Primeros plots: 

# 1. Profundidad geométrica vs log(τR)
plt.figure(figsize=(10, 8))
plt.plot(modelo_5000['profundidad'], modelo_5000['logtauR'], 'o-', color=color_5000, label='$T_{\mathrm{eff}} = 5000 \, \mathrm{K}$')  # Círculos con línea
plt.plot(modelo_8000['profundidad'], modelo_8000['logtauR'], 's-', color=color_8000, label='$T_{\mathrm{eff}} = 8000 \, \mathrm{K}$')  # Cuadrados con línea
plt.xlabel('$r$ [cm]')
plt.ylabel(r'$\log(\tau_R)$')
plt.legend(fontsize=tamaño_fuente_leyenda)
plt.grid(True)
plt.savefig('Figuras/profundidad_logtauR.pdf')

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

# %%

# Constantes en cgs
k_B = 1.380622e-16
eV_to_erg = 1.602176634e-12

# Constantes en eV
kB_eV = 8.617e-5
chi_HI = 13.6             # extraído del Gray
chi_Hneg = 0.755          # extraído del Gray

# Funciones de partición 
g_HI = 2
g_HII = 1
g_Hneg = 1


# =============================================================================
# Funciones para obtener poblaciones en de H-, HI, HII
# =============================================================================

def Ne_ideal_gases(Pe, T):
    return Pe / ( k_B * T )

def Saha_1(Ne, T):        # n(HI)/n(HII)
    return 2.07e-16 * Ne * (g_HI/g_HII) * T**(-3/2) * np.exp( chi_HI / (kB_eV * T) )
    
def Saha_2(Ne, T):        # n(Hneg)/n(HI)
    return 2.07e-16 * Ne * (g_Hneg/g_HI) * T**(-3/2) * np.exp( chi_Hneg / (kB_eV * T) )

def charge_conservation(saha_factor_1, saha_factor_2, Ne, n_HII):
    
    n_HI   = saha_factor_1 * n_HII
    n_Hneg = saha_factor_2 * n_HI
    
    return Ne + n_Hneg - n_HII



def populations_finder(Pe, T):
    
    Ne = Ne_ideal_gases(Pe, T)
    saha_factor_1 = Saha_1(Ne, T)
    saha_factor_2 = Saha_2(Ne, T)
    #print(Ne, saha_factor_1, saha_factor_2)
    
    func = lambda n: charge_conservation(saha_factor_1, saha_factor_2, Ne, n)
    initial_guess = Ne * 0.01
    solution = np.clip(fsolve(func, initial_guess), 0, None)

    n_HII = solution[0]
    
    n_HI   = saha_factor_1 * n_HII
    n_Hneg = saha_factor_2 * n_HI
    n_vector = np.array([n_Hneg, n_HI, n_HII])
    
    return n_vector


# =============================================================================
# Funciones para obtener poblaciones en los niveles n=1,2,3 de HI
# =============================================================================

def g(n):
    return 2 * n**2

def E_n(n):
    return -13.6 / (n**2)     # eV

def Boltzmann(T):
    g1, g2, g3 = g(1), g(2), g(3)
    E1, E2, E3 = E_n(1), E_n(2), E_n(3)
    
    n2_n1 = (g2/g1) * np.exp(-(E2-E1)/(kB_eV*T))
    n3_n1 = (g3/g1) * np.exp(-(E3-E1)/(kB_eV*T))
    
    return n2_n1, n3_n1


def n_Boltzmann(n2_n1, n3_n1, n1):
    n2 = n2_n1 * n1
    n3 = n3_n1 * n1
    
    return n_HI - (n1 + n2 + n3)


def n_levels_finder(n_HI, T):
    
    n2_n1, n3_n1 = Boltzmann(T)
    
    func = lambda n: n_Boltzmann(n2_n1, n3_n1, n)
    initial_guess = n_HI * 0.6
    solution = np.clip(fsolve(func, initial_guess), 0, None)

    n1 = solution[0]
    n2 = n2_n1 * n1
    n3 = n3_n1 * n1
    n_vector = np.array([n1, n2, n3])
    
    return n_vector


# =============================================================================
# Cálculos para obtener las tablas
# =============================================================================

# Estrella con Teff = 5000 K

tauR_1 = 10**logtauR_1

# Buscamos el índice para el cual tau = 0.5, 5
for ii in range(len(tauR_1)):
    if abs(tauR_1[ii]-0.5)<0.1:
        tau_0_5_index = ii
    elif abs(tauR_1[ii]-5)<0.1:
        tau_5_index = ii

T_1_tau_0_5 = T_1[tau_0_5_index]
Pe_1_tau_0_5 = Pe_1[tau_0_5_index]
Ne_1_tau_0_5 = Ne_ideal_gases(Pe_1_tau_0_5, T_1_tau_0_5)
n_vector = populations_finder(Pe_1_tau_0_5, T_1_tau_0_5)
n_HI = n_vector[1]
n_levels = n_levels_finder(n_HI, T_1_tau_0_5)

df_model_1 = pd.DataFrame(columns=['tauR', 'n(H-)', 'n(HI)', 'n(HII)', 'Ne', 'n(HI, n=1)', 'n(HI, n=2)', 'n(HI, n=3)'])
new_row = [0.5, n_vector[0], n_vector[1], n_vector[2], Ne_1_tau_0_5, n_levels[0], n_levels[1], n_levels[2]]
df_model_1.loc[len(df_model_1)] = new_row

T_1_tau_5 = T_1[tau_5_index]
Pe_1_tau_5 = Pe_1[tau_5_index]
Ne_1_tau_5 = Ne_ideal_gases(Pe_1_tau_5, T_1_tau_5)
n_vector = populations_finder(Pe_1_tau_5, T_1_tau_5)
n_HI = n_vector[1]
n_levels = n_levels_finder(n_HI, T_1_tau_5)

new_row = [5, n_vector[0], n_vector[1], n_vector[2], Ne_1_tau_5, n_levels[0], n_levels[1], n_levels[2]]
df_model_1.loc[len(df_model_1)] = new_row

print('1) Estrella con Teff = 5000 K')
print(df_model_1)



# Estrella con Teff = 8000 K

tauR_2 = 10**logtauR_2

# Buscamos el índice para el cual tau = 0.5, 5
for ii in range(len(tauR_2)):
    if abs(tauR_2[ii]-0.5)<0.1:
        tau_0_5_index = ii
    elif abs(tauR_2[ii]-5)<0.1:
        tau_5_index = ii
        
T_2_tau_0_5 = T_2[tau_0_5_index]
Pe_2_tau_0_5 = Pe_2[tau_0_5_index]
Ne_2_tau_0_5 = Ne_ideal_gases(Pe_2_tau_0_5, T_2_tau_0_5)
n_vector = populations_finder(Pe_2_tau_0_5, T_2_tau_0_5)
n_HI = n_vector[1]
n_levels = n_levels_finder(n_HI, T_2_tau_0_5)

df_model_2 = pd.DataFrame(columns=['tauR', 'n(H-)', 'n(HI)', 'n(HII)', 'Ne', 'n(HI, n=1)', 'n(HI, n=2)', 'n(HI, n=3)'])
new_row = [0.5, n_vector[0], n_vector[1], n_vector[2], Ne_2_tau_0_5, n_levels[0], n_levels[1], n_levels[2]]
df_model_2.loc[len(df_model_2)] = new_row

T_2_tau_5 = T_2[tau_5_index]
Pe_2_tau_5 = Pe_2[tau_5_index]
Ne_2_tau_5 = Ne_ideal_gases(Pe_2_tau_5, T_2_tau_5)
n_vector = populations_finder(Pe_2_tau_5, T_2_tau_5)
n_HI = n_vector[1]
n_levels = n_levels_finder(n_HI, T_2_tau_5)

new_row = [5, n_vector[0], n_vector[1], n_vector[2], Ne_2_tau_5, n_levels[0], n_levels[1], n_levels[2]]
df_model_2.loc[len(df_model_2)] = new_row

print('\n2) Estrella con Teff = 8000 K')
print(df_model_2)




# %%

# =============================================================================
# Funciones para las opacidades
# =============================================================================

# Constants in cgs
R = 1.0968e5         # Rydberg constant
e = 4.803e-10        # esu
h = 6.626196e-27     # Planck constant
c = 2.997924562e10   # Speed of light
m_e = 9.109558e-28   # Electron mass
k_B = 1.380622e-16   # Boltzmann constant



# =============================================================================
# Opacidades del HI
# =============================================================================

# Opacidad free-free del HI
def g_ff(ldo, T):
    """
    Args:
        ldo(np.array): Array with the considered wavelengths, in cm
        T (float): Temperature, in K
        
    Returns:
        np.array: Gaunt factor, adimensional
    """ 
    return 1 + ( 0.3456 / ( (ldo*R)**(1/3) ) ) * ( ldo*k_B*T / (h*c)  + 1/2 )

def sigma_ff_HI(Z, ldo, T):
    """
    Args:
        Z (float): charge of the nucleus of the atom or ion
        ldo(np.array): Array with the considered wavelengths in cm
        T (float): Temperature, in K   

    Returns:
        np.array: Arrays with f-f cross sections of HI, in cm^2
    """  
    prefactor = 2/(3**(3/2)) * h**2 * e**2 * R * np.sqrt(2 * m_e / (np.pi * k_B)) / ( np.pi * m_e**3 )      # = 3.69e8
    freq = c / ldo
    return prefactor * Z**2 / ( T**(1/2) * freq**3 ) * g_ff(ldo, T)

def kappa_ff_HI(Z, ldo, T, Ne, n_HII):
    """
    Args:
        Z (float): charge of the nucleus of the atom or ion
        ldo(np.array): Array with the considered wavelengths in cm
        T (float): Temperature, in K
        Ne (float): Number density of electrons, in cm^-3
        n_HII (float): Number density of ionized Hydrogen, in cm^-3

    Returns:
        np.array: Array with f-f opacities of HI, in cm^-1
    """
    freq = c / ldo
    sigma = sigma_ff_HI(Z, ldo, T)
    return sigma * Ne * n_HII * ( 1 - np.exp( -h * freq / (k_B * T) ) )


# Opacidad bound-free del HI
def g_bf(ldo, n):
    """
    Args:
        ldo(np.array): Array with the considered wavelengths, in cm
        n (int): quantum number n
        
    Returns:
        np.array: Gaunt factor, adimensional
    """   
    return 1 - ( 0.3456 / ( (ldo*R)**(1/3) ) ) * ( ldo * R / (n**2) - 1/2 )

def sigma_bf_HI(Z, n, ldo):
    """
    Args:
        Z (float): charge of the nucleus of the atom or ion
        n (int): quantum number n
        ldo(np.array): Array with the considered wavelengths in cm

    Returns:
        np.array: cross sections in cm^2
    """       
    prefactor = 32/(3**(3/2)) * np.pi**2 * e**6 * R / (h)**3     # = 2.813e29
    freq = c / ldo
    ldo_max = n**2 / R
    freq_max = c / ldo_max
    
    sigma_list = []
    for jj in range(len(freq)):
        if freq[jj] >= freq_max:
            sigma = prefactor * Z**4 / (n**5 * freq[jj]**3) * g_bf(ldo[jj], n)
            sigma_list.append( sigma )
        elif freq[jj] < freq_max:
            sigma_list.append(0)
    
    return np.array(sigma_list)

def kappa_bf_HI(Z, n, ldo, T, ni):
    """
    Args:
        Z (float): charge of the nucleus of the atom or ion
        n (int): quantum number n
        ldo(np.array): Array with the considered wavelengths in cm
        T (float): Temperature, in K
        ni (float): Number density of neutral Hydrogen in energy level n, in cm^-3

    Returns:
        np.array: Array with b-f opacities of HI, in cm^-1
    """
    f = c / ldo
    sigma = sigma_bf_HI(Z, n, ldo)
    return sigma * ni * ( 1 - np.exp( -h * f / (k_B * T) ) )


# Opacidad bound-bound del HI
def g_bb_HI(u, serie):
    """
    Args:
        u (float): n of upper level in the transition
        serie (str): transition. It can takes the value Balmer, Lymann alpha or Lymann beta

    Returns:
        float: Gaunt factor for bound-bound transitions in HI
    """
    if serie == 'Lymann alpha':
        g_bb = 0.717
    elif serie == 'Lymann beta':
        g_bb = 0.765
    elif serie == 'Balmer':
        g_bb = 0.869 - 3 / (u**3)
    
    return g_bb     
    
def sigma_bb_HI(l, u):
    """
    Args:
        l (int): quantum number n of the lower energy level of the atomic transition
        u (int): quantum number n of the upper energy level of the atomic transition

    Returns:
        float: b-b cross section of HI, in cm^2
    """  
    if l==1 and u==2:
        serie = 'Lymann alpha'
    elif l==1 and u==3:
        serie = 'Lymann beta'
    elif l==2 and u==3:
        serie = 'Balmer'
    
    prefactor = np.pi * e**2 / (m_e * c)
    g_bb = g_bb_HI(u, serie)
    l, u = float(l), float(u)
    f = 2**5 / (3**(3/2) * np.pi) * l**(-5) * u**(-3) * ( 1/l**2 - 1/u**2 )**(-3) * g_bb
    
    return prefactor * f

def kappa_bb_HI(l, u, n_l, n_u):
    """
    Args:
        l (int): quantum number n of the lower energy level of the atomic transition
        u (int): quantum number n of the upper energy level of the atomic transition
        n_l (int): population of the lower energy level of the atomic transition, in cm^-3
        n_u (int): population of the upper energy level of the atomic transition, in cm^-3

    Returns:
        float: b-b opacity for HI, in cm^2
    """ 
    sigma = sigma_bb_HI(l, u)
    return sigma * (n_l - n_u)
    
def lambda_Rydberg(l, u):
    """
    Args:
        l (int): quantum number n of the lower energy level of the atomic transition
        u (int): quantum number n of the upper energy level of the atomic transition

    Returns:
        float: light wavelength for transition u -> l, in cm
    """  
    ldo =  1 / ( R * ( 1/(l**2) - 1/(u**2) ) )
    return ldo



# =============================================================================
# Opacidades del H-
# =============================================================================

# Opacidad free-free del H-
def sigma_ff_Hneg(ldo, T):
    """
    Args:
        ldo (float or np.array): wavelengths, in cm
        T (float): Temperature, in K

    Returns:
        float or np.array: f-f cross section of H anion, in cm^2
    """
    ldo_A = ldo * 1e8
    f0 = -2.2763 - 1.6850 * np.log10(ldo_A) + 0.76661 * (np.log10(ldo_A))**2 - 0.053346 * (np.log10(ldo_A))**3
    f1 = 15.2827 - 9.2846 * np.log10(ldo_A) + 1.99381 * (np.log10(ldo_A))**2 - 0.142631 * (np.log10(ldo_A))**3
    f2 = -197.789 + 190.266 * np.log10(ldo_A) - 67.9775 * (np.log10(ldo_A))**2 + 10.6913 * (np.log10(ldo_A))**3 - 0.625151 * (np.log10(ldo_A))**4
    
    theta = 5040/T
    sigma = 1e-26 * 10**( f0 + f1 * np.log10(theta) + f2 * (np.log10(theta))**2 )
    return sigma

def kappa_ff_Hneg(ldo, T, Pe, n_HI):
    """
    Args:
        ldo(np.array): Array with the considered wavelengths, in cm
        T (float): Temperature, in K
        Pe (float): Electronic pressure, in dyn/cm^2
        n_HI (float): Number density of neutral Hydrogen, in cm^-3

    Returns:
        np.array: Array with f-f opacities of H-, in cm^-1
    """
    sigma = sigma_ff_Hneg(ldo, T)
    return sigma * Pe * n_HI


# Opacidad bound-free del H-
def sigma_bf_Hneg(ldo):
    """
    Args:
        ldo (np.array): wavelengths, in cm

    Returns:
        np.array: cross section, in cm^2
    """
    ldo_A = ldo * 1e8      # Pasamos a Angstrom para usar las constantes
    
    a0 = 1.99654
    a1 = -1.18267e-5
    a2 = 2.64243e-6
    a3 = -4.40524e-10
    a4 = 3.23992e-14
    a5 = -1.39568e-18
    a6 = 2.78701e-23
    
    sigma = (a0 + a1*ldo_A + a2*ldo_A**2 + a3*ldo_A**3 + a4*ldo_A**4 + a5*ldo_A**5 + a6*ldo_A**6) * 1e-18
    
    cut_index = len(sigma)
    for jj in np.arange(len(sigma)):
        if sigma[jj]<1e-30:
            cut_index = jj
            break
    
    physical_sigma = np.concatenate( ( sigma[0:cut_index], np.zeros((len(sigma) - cut_index)) ), 0 )

    return physical_sigma

def kappa_bf_Hneg(ldo, T, Pe, n_vector):
    """
    Args:
        ldo(np.array): Array with the considered wavelengths, in cm
        T (float): Temperature, in K
        Pe (float): Electronic pressure, in dyn/cm^2
        n_vector (np.array): Array with number densities of H-, HI and HII (in this order), in cm^-3

    Returns:
        np.array: Array with b-f opacities of H-, in cm^-1
    """
    n_HI = n_vector[1]
    sigma = sigma_bf_Hneg(ldo)
    theta = 5040 / T
    kappa = 4.158e-10 * sigma * Pe * theta**(5/2) * 10**(0.754*theta) * n_HI
    return kappa


# =============================================================================
# Opacidad electrones
# =============================================================================

sigma_e = 6.648e-25         # Gray

def kappa_e(Ne):
    """
    Args:
        Ne (float): Number density of electrons, in cm^-3

    Returns:
        float: opacity of electronic scattering, in cm^2
    """
    return Ne * sigma_e





# %%

# =============================================================================
# Gráficas de cross sections
# =============================================================================

# Buscamos el índice para el cual tau = 1, en ambos modelos
for ii in range(len(tauR_1)):
    if abs(tauR_1[ii]-1)<0.1:
        tau_1_index_1 = ii

for ii in range(len(tauR_2)):
    if abs(tauR_2[ii]-1)<0.1:
        tau_2_index_2 = ii
        
T_1_tau_1 = T_1[tau_1_index_1]
Pe_1_tau_1 = Pe_1[tau_1_index_1]
Ne_1_tau_1 = Ne_ideal_gases(Pe_1_tau_1, T_1_tau_1)
n_vector_1 = populations_finder(Pe_1_tau_1, T_1_tau_1)
n_HI = n_vector_1[1]
n_levels_1 = n_levels_finder(n_HI, T_1_tau_1)
n1_1, n2_1, n3_1 = n_levels_1

T_2_tau_1 = T_2[tau_2_index_2]
Pe_2_tau_1 = Pe_2[tau_2_index_2]
Ne_2_tau_1 = Ne_ideal_gases(Pe_2_tau_1, T_2_tau_1)
n_vector_2 = populations_finder(Pe_2_tau_1, T_2_tau_1)
n_HI = n_vector_2[1]
n_levels_2 = n_levels_finder(n_HI, T_2_tau_1)
n1_2, n2_2, n3_2 = n_levels_2

lambda_array_A = np.arange(500, 20000, 0.5)            # En Angstrom
lambda_array_cm = lambda_array_A * 1e-8

sigma_ff_HI_1 = sigma_ff_HI(Z=1, ldo=lambda_array_cm, T=T_1_tau_1)
sigma_ff_HI_2 = sigma_ff_HI(Z=1, ldo=lambda_array_cm, T=T_2_tau_1)

plt.figure(figsize=(10, 8))
plt.title('$\sigma_{ff}$ (HI)', fontsize=18)

plt.plot(lambda_array_A, sigma_ff_HI_1, label='$T_{eff}$ = 5000 K', color='#1f9ac9')
plt.plot(lambda_array_A, sigma_ff_HI_2, label='$T_{eff}$ = 8000 K', color='#5c068c')

plt.yscale('log')
plt.xscale('log')
plt.ylabel('$\sigma_{ff}$ [cm$^2$]', fontsize=20)
plt.xlabel('$\lambda$ [$\mathrm{\AA}$]', fontsize=20)
plt.legend(fontsize=20)
plt.grid(which='both', alpha=0.4)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.savefig('Figures/sigma_ff_HI.pdf')

sigma_ff_Hneg_1 = sigma_ff_Hneg(ldo=lambda_array_cm, T=T_1_tau_1)
sigma_ff_Hneg_2 = sigma_ff_Hneg(ldo=lambda_array_cm, T=T_2_tau_1)

plt.figure(figsize=(10, 8))
plt.title('$\sigma_{ff}$ (H$^-$)', fontsize=20)

plt.plot(lambda_array_A, sigma_ff_Hneg_1, label='$T_{eff}$ = 5000 K', color='#1f9ac9')
plt.plot(lambda_array_A, sigma_ff_Hneg_2, label='$T_{eff}$ = 8000 K', color='#5c068c')

plt.yscale('log')
plt.xscale('log')
plt.ylabel('$\sigma_{ff}$ [cm$^2$]', fontsize=20)
plt.xlabel('$\lambda$ [$\mathrm{\AA}$]', fontsize=20)
plt.legend(fontsize=20)
plt.grid(which='both', alpha=0.4)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.savefig('Figures/sigma_ff_Hneg.pdf')





# %%

# Sigma para las transiciones b-f desde los 3 niveles de HI
sigma_bf_HI_n1 = sigma_bf_HI(Z=1, n=1, ldo = lambda_array_cm)
sigma_bf_HI_n2 = sigma_bf_HI(Z=1, n=2, ldo = lambda_array_cm)
sigma_bf_HI_n3 = sigma_bf_HI(Z=1, n=3, ldo = lambda_array_cm)

sigma_bf_Hneg_ar = sigma_bf_Hneg(ldo=lambda_array_cm)

plt.figure(figsize=(10, 8))
plt.title('$\sigma_{bf}$', fontsize=20)

plt.plot(lambda_array_A, sigma_bf_HI_n1, label='HI, n=1', color='#1f9ac9')  # Válido para los dos modelos
plt.plot(lambda_array_A, sigma_bf_HI_n2, label='HI, n=2', color='#5c068c')
plt.plot(lambda_array_A, sigma_bf_HI_n3, label='HI, n=3', color='#1f9ac9', linestyle='--')
plt.plot(lambda_array_A, sigma_bf_Hneg_ar, label='H$^-$', color='#5c068c', linestyle='--')

plt.yscale('log')
plt.xscale('log')
plt.ylabel('$\sigma_{bf}$ [cm$^2$]', fontsize=20)
plt.xlabel('$\lambda$ [$\mathrm{\AA}$]', fontsize=20)
plt.legend(fontsize=16)
plt.grid(which='both', alpha=0.4)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.savefig('Figures/sigma_bf.pdf')


# %%

# =============================================================================
# Gráficas de opacidades
# =============================================================================

# Estrella con Teff = 5000 K

kappa_ff_HI_1    = kappa_ff_HI(Z=1, ldo=lambda_array_cm, T=T_1_tau_1, Ne=Ne_1_tau_1, n_HII=n_vector_1[2])
kappa_bf_HI_n1_1 = kappa_bf_HI(Z=1, n=1, ldo=lambda_array_cm, T=T_1_tau_1, ni=n1_1)
kappa_bf_HI_n2_1 = kappa_bf_HI(Z=1, n=2, ldo=lambda_array_cm, T=T_1_tau_1, ni=n2_1)
kappa_bf_HI_n3_1 = kappa_bf_HI(Z=1, n=3, ldo=lambda_array_cm, T=T_1_tau_1, ni=n3_1)
kappa_ff_Hneg_1  = kappa_ff_Hneg(ldo=lambda_array_cm, T=T_1_tau_1, Pe=Pe_1_tau_1, n_HI=n_vector_1[1])
kappa_bf_Hneg_1  = kappa_bf_Hneg(ldo=lambda_array_cm, T=T_1_tau_1, Pe=Pe_1_tau_1, n_vector=n_vector_1)
kappa_e_1        = kappa_e(Ne_1_tau_1) * np.ones(len(lambda_array_cm))

kappa_total_1 = kappa_ff_HI_1 + kappa_bf_HI_n1_1 + kappa_bf_HI_n2_1 + kappa_bf_HI_n3_1 + kappa_ff_Hneg_1 + kappa_bf_Hneg_1 + kappa_e_1 

plt.figure(figsize=(10, 8))
plt.title('Star with $T_{eff}$ = 5000 K', fontsize=20)

# Usando colores diferentes para cada línea
plt.plot(lambda_array_A, kappa_ff_HI_1, label='f-f HI', color='#1f77b4')
plt.plot(lambda_array_A, kappa_bf_HI_n1_1, label='b-f HI, n = 1', color='#ff7f0e')
plt.plot(lambda_array_A, kappa_bf_HI_n2_1, label='b-f HI, n = 2', color='#2ca02c')
plt.plot(lambda_array_A, kappa_bf_HI_n3_1, label='b-f HI, n = 3', color='#d62728')
plt.plot(lambda_array_A, kappa_ff_Hneg_1, label='f-f H$^-$', color='#9467bd')
plt.plot(lambda_array_A, kappa_bf_Hneg_1, label='b-f H$^-$', color='#8c564b')
plt.plot(lambda_array_A, kappa_e_1, label='e$^-$ scattering', color='#e377c2', linestyle='-.')
plt.plot(lambda_array_A, kappa_total_1, label='Total opacity', color='black')

plt.yscale('log')
plt.xscale('log')
plt.ylabel('$\kappa$ [cm$^{-1}$]', fontsize=20)
plt.xlabel('$\lambda$ [$\mathrm{\AA}$]', fontsize=20)
plt.legend(fontsize=15.5)
plt.grid(which='both', alpha=0.4)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.savefig('Figures/kappa_1.pdf')

# %%


# Estrella con Teff = 8000 K 

kappa_ff_HI_2    = kappa_ff_HI(Z=1, ldo=lambda_array_cm, T=T_2_tau_1, Ne=Ne_2_tau_1, n_HII=n_vector_2[2])
kappa_bf_HI_n1_2 = kappa_bf_HI(Z=1, n=1, ldo=lambda_array_cm, T=T_2_tau_1, ni=n1_2)
kappa_bf_HI_n2_2 = kappa_bf_HI(Z=1, n=2, ldo=lambda_array_cm, T=T_2_tau_1, ni=n2_2)
kappa_bf_HI_n3_2 = kappa_bf_HI(Z=1, n=3, ldo=lambda_array_cm, T=T_2_tau_1, ni=n3_2)
kappa_ff_Hneg_2  = kappa_ff_Hneg(ldo=lambda_array_cm, T=T_2_tau_1, Pe=Pe_2_tau_1, n_HI=n_vector_2[1])
kappa_bf_Hneg_2  = kappa_bf_Hneg(ldo=lambda_array_cm, T=T_2_tau_1, Pe=Pe_2_tau_1, n_vector=n_vector_2)
kappa_e_2        = kappa_e(Ne_2_tau_1) * np.ones(len(lambda_array_cm))

kappa_total_2 = kappa_ff_HI_2 + kappa_bf_HI_n1_2 + kappa_bf_HI_n2_2 + kappa_bf_HI_n3_2 + kappa_ff_Hneg_2 + kappa_bf_Hneg_2 + kappa_e_2

plt.figure(figsize=(10, 8))
plt.title('Star with $T_{eff}$ = 8000 K', fontsize=20)

plt.plot(lambda_array_A, kappa_ff_HI_2, label='f-f HI', color='#1f77b4')
plt.plot(lambda_array_A, kappa_bf_HI_n1_2, label='b-f HI, n = 1', color='#ff7f0e')
plt.plot(lambda_array_A, kappa_bf_HI_n2_2, label='b-f HI, n = 2', color='#2ca02c')
plt.plot(lambda_array_A, kappa_bf_HI_n3_2, label='b-f HI, n = 3', color='#d62728')
plt.plot(lambda_array_A, kappa_ff_Hneg_2, label='f-f H$^-$', color='#9467bd')
plt.plot(lambda_array_A, kappa_bf_Hneg_2, label='b-f H$^-$', color='#8c564b')
plt.plot(lambda_array_A, kappa_e_2, label='e$^-$ scattering', color='#e377c2', linestyle='-.')
plt.plot(lambda_array_A, kappa_total_2, label='Total opacity', color='black')

plt.yscale('log')
plt.xscale('log')
plt.ylabel('$\kappa$ [cm$^{-1}$]', fontsize=20)
plt.xlabel('$\lambda$ [$\mathrm{\AA}$]', fontsize=20)
plt.legend(fontsize=17)
plt.grid(which='both', alpha=0.4)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.savefig('Figures/kappa_2.pdf')

f# %%


plt.figure(figsize=(10, 8))
plt.title('Comparison of Total Opacity', fontsize=20)

plt.plot(lambda_array_A, kappa_total_1, label='$\kappa_{total}, T_{eff}$ = 5000 K', color='#1f77b4')
plt.plot(lambda_array_A, kappa_total_2, label='$\kappa_{total}, T_{eff}$ = 8000 K', color='#d62728')

plt.yscale('log')
plt.xscale('log')
plt.ylabel('$\kappa$ [cm$^{-1}$]', fontsize=20)
plt.xlabel('$\lambda$ [$\mathrm{\AA}$]', fontsize=20)
plt.legend(fontsize=20)
plt.grid(which='both', alpha=0.4)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.savefig('Figures/kappa_total_comparison.pdf')

# %%

# =============================================================================
# Tablas de opacidades
# =============================================================================

def longitud_corte(n):
    return (n**2 / R) * 1e8

n = np.array([1, 2, 3])
cortes_cantos = longitud_corte(n)
print(cortes_cantos)



lambdas_tabla = []
for ldo in cortes_cantos:
    delta_ldo = 10                      # Angstrom
    lambdas_tabla.append(ldo - delta_ldo)
    lambdas_tabla.append(ldo + delta_ldo)

lambdas_tabla_A = np.array(lambdas_tabla)
lambdas_tabla_cm = lambdas_tabla_A * 1e-8


# Estrella con Teff = 5000 K

kappa_ff_HI_1    = kappa_ff_HI(Z=1, ldo=lambdas_tabla_cm, T=T_1_tau_1, Ne=Ne_1_tau_1, n_HII=n_vector_1[2])
kappa_bf_HI_n1_1 = kappa_bf_HI(Z=1, n=1, ldo=lambdas_tabla_cm, T=T_1_tau_1, ni=n1_1)
kappa_bf_HI_n2_1 = kappa_bf_HI(Z=1, n=2, ldo=lambdas_tabla_cm, T=T_1_tau_1, ni=n2_1)
kappa_bf_HI_n3_1 = kappa_bf_HI(Z=1, n=3, ldo=lambdas_tabla_cm, T=T_1_tau_1, ni=n3_1)
kappa_ff_Hneg_1  = kappa_ff_Hneg(ldo=lambdas_tabla_cm, T=T_1_tau_1, Pe=Pe_1_tau_1, n_HI=n_vector_1[1])
kappa_bf_Hneg_1  = kappa_bf_Hneg(ldo=lambdas_tabla_cm, T=T_1_tau_1, Pe=Pe_1_tau_1, n_vector=n_vector_1)
kappa_e_1        = kappa_e(Ne_1_tau_1) * np.ones(len(lambdas_tabla_cm))

data_1 = [kappa_ff_HI_1, kappa_bf_HI_n1_1, kappa_bf_HI_n2_1, kappa_bf_HI_n3_1, kappa_ff_Hneg_1, kappa_bf_Hneg_1, kappa_e_1]
row_names = ["kappa_ff_HI", "kappa_bf_HI_n1", "kappa_bf_HI_n2", "kappa_bf_HI_n3", "kappa_ff_Hneg", "kappa_bf_Hneg", "kappa_e"]

df_tabla_final_1 = pd.DataFrame(data_1, columns=lambdas_tabla_A)
df_tabla_final_1.index = row_names
print('\n1) Estrella con Teff = 5000 K')
print(df_tabla_final_1)


#%%
# Estrella con Teff = 8000 K
kappa_ff_HI_2    = kappa_ff_HI(Z=1, ldo=lambdas_tabla_cm, T=T_2_tau_1, Ne=Ne_2_tau_1, n_HII=n_vector_2[2])
kappa_bf_HI_n1_2 = kappa_bf_HI(Z=1, n=1, ldo=lambdas_tabla_cm, T=T_2_tau_1, ni=n1_2)
kappa_bf_HI_n2_2 = kappa_bf_HI(Z=1, n=2, ldo=lambdas_tabla_cm, T=T_2_tau_1, ni=n2_2)
kappa_bf_HI_n3_2 = kappa_bf_HI(Z=1, n=3, ldo=lambdas_tabla_cm, T=T_2_tau_1, ni=n3_2)
kappa_ff_Hneg_2  = kappa_ff_Hneg(ldo=lambdas_tabla_cm, T=T_2_tau_1, Pe=Pe_2_tau_1, n_HI=n_vector_2[1])
kappa_bf_Hneg_2  = kappa_bf_Hneg(ldo=lambdas_tabla_cm, T=T_2_tau_1, Pe=Pe_2_tau_1, n_vector=n_vector_2)
kappa_e_2        = kappa_e(Ne_2_tau_1) * np.ones(len(lambdas_tabla_cm))

data_2 = [kappa_ff_HI_2, kappa_bf_HI_n1_2, kappa_bf_HI_n2_2, kappa_bf_HI_n3_2, kappa_ff_Hneg_2, kappa_bf_Hneg_2, kappa_e_2]
df_tabla_final_2 = pd.DataFrame(data_2, columns=lambdas_tabla_A)
df_tabla_final_2.index = row_names
print('\n2) Estrella con Teff = 8000 K')
print(df_tabla_final_2)


# %%

# =============================================================================
# Tabla de opacidades b-b del HI
# =============================================================================

print('OPACIDADES b-b DEL HI')

niveles_series = [np.array([1, 3]), np.array([1, 2]), np.array([2, 3])]
poblaciones_series_1 = [np.array([n1_1, n3_1]), np.array([n1_1, n2_1]), np.array([n2_1, n3_1])]
poblaciones_series_2 = [np.array([n1_2, n3_2]), np.array([n1_2, n2_2]), np.array([n2_2, n3_2])]


# Estrella con Teff = 5000 K

print('\n1) Estrella con Teff = 5000 K')


poblaciones_series = poblaciones_series_1
for niveles, poblaciones in zip(niveles_series, poblaciones_series):
    l, u      =  niveles[0], niveles[1]
    n_l, n_u  =  poblaciones[0], poblaciones[1]
    ldo_salto_cm = lambda_Rydberg(l, u)
    ldo_salto_A  = ldo_salto_cm * 1e8
    kappa = kappa_bb_HI(l, u, n_l, n_u)
    print(f"n = {l} -> n = {u}:")
    print(f"ldo = {ldo_salto_A:.2f} A, kappa = {kappa:e}")
    

# Estrella con Teff = 8000 K

print('\n2) Estrella con Teff = 8000 K')

poblaciones_series = poblaciones_series_2
for niveles, poblaciones in zip(niveles_series, poblaciones_series):
    l, u      =  niveles[0], niveles[1]
    n_l, n_u  =  poblaciones[0], poblaciones[1]
    ldo_salto_cm = lambda_Rydberg(l, u)
    ldo_salto_A  = ldo_salto_cm * 1e8
    kappa = kappa_bb_HI(l, u, n_l, n_u)
    print(f"n = {l} -> n = {u}:")
    print(f"ldo = {ldo_salto_A:.2f} A, kappa = {kappa:e}")
