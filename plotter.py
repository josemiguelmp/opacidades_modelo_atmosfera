#%%

from scipy.optimize import fsolve
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import os


# Leemos el primer archivo (T = 5000 K)

file_name_1 = "t5000.dat"
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

file_name_2 = "t8000.dat"
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

# log tau vs depth
plt.figure(figsize=(10, 8))
plt.plot(r_1, logtauR_1, 'b-', label='$T_{eff}$ = 5000 K')
plt.plot(r_2, logtauR_2, 'r-', label='$T_{eff}$ = 8000 K')
plt.xlabel('Geometrical depth [cm]')
plt.ylabel('log $\\tau_{Ross}$')
plt.legend()
plt.savefig('Figures/r_logtauR.pdf')

# Temperature vs log(tau)
plt.figure(figsize=(10, 8))
plt.plot(logtauR_1, T_1, 'b-', label='$T_{eff}$ = 5000 K')
plt.plot(logtauR_2, T_2, 'r-', label='$T_{eff}$ = 8000 K')
plt.xlabel('log $\\tau_{Ross}$')
plt.ylabel('T [K]')
plt.legend()
plt.savefig('Figures/logtauR_T.pdf')

# Pe vs log(tau)
plt.figure(figsize=(10, 8))
plt.plot(logtauR_1, Pe_1, 'b-', label='$T_{eff}$ = 5000 K')
plt.plot(logtauR_2, Pe_2, 'r-', label='$T_{eff}$ = 8000 K')
plt.xlabel('log $\\tau_{Ross}$')
plt.ylabel('$P_e$ [dyn/cm$^2$]')
plt.yscale('log')
plt.legend()
plt.savefig('Figures/logtauR_Pe.pdf')

# Pe/Pg vs log(tau)
plt.figure(figsize=(10, 8))
plt.plot(logtauR_1, Pe_1/Pg_1, 'b-', label='$T_{eff}$ = 5000 K')
plt.plot(logtauR_2, Pe_2/Pg_2, 'r-', label='$T_{eff}$ = 8000 K')
plt.xlabel('log $\\tau_{Ross}$')
plt.ylabel('$P_e/P_g$')
plt.yscale('log')
plt.legend()
plt.savefig('Figures/logtauR_Pe_Pg.pdf')

# Prad/Pg vs log(tau)
plt.figure(figsize=(10, 8))
plt.plot(logtauR_1, Prad_1/Pg_1, 'b-', label='$T_{eff}$ = 5000 K')
plt.plot(logtauR_2, Prad_2/Pg_2, 'r-', label='$T_{eff}$ = 8000 K')
plt.xlabel('log $\\tau_{Ross}$')
plt.ylabel('$P_{rad}/P_g$')
plt.yscale('log')
plt.legend()
plt.savefig('Figures/logtauR_Prad_Pg.pdf')


# %%

T_gris_1 = ( (3/4) * Teff_1**4 * (10**(logtauR_1) + 2/3) )**(1/4)
T_gris_2 = ( (3/4) * Teff_2**4 * (10**(logtauR_2) + 2/3) )**(1/4)

plt.figure(figsize=(10, 8))
plt.plot(logtauR_1, T_gris_1, 'g-', label='T gray body')
plt.plot(logtauR_1, T_1, 'b-', label='T modelo MARCS')
plt.xlabel('log $\\tau_{Ross}$')
plt.ylabel('T [K]')
plt.title('Star with $T_{eff}$ = 5000 K')
plt.legend()
plt.savefig('Figures/Temperature_gray_body_1.pdf')

plt.figure(figsize=(10, 8))
plt.plot(logtauR_2, T_gris_2, 'g-', label='T gray body')
plt.plot(logtauR_2, T_2, 'b-', label='T modelo MARCS')
plt.xlabel('log $\\tau_{Ross}$')
plt.ylabel('T [K]')
plt.title('Star with $T_{eff}$ = 8000 K')
plt.legend()
plt.savefig('Figures/Temperature_gray_body_2.pdf')

# %%

# Constantes en cgs
k_B = 1.380622e-16
eV_to_erg = 1.602176634e-12

# Constantes en eV
kB_eV = 8.617e-5
chi_HI = 13.6
chi_Hneg = 0.75

# Funciones de partición 
g_HI = 2
g_HII = 1
g_Hneg = 1

#Ne_1 = Pe_1/(k_B*T_1)     # cm^-3
#Ne_2 = Pe_2/(k_B*T_2)     # cm^-3


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

for ii in range(len(tauR_1)):
    if abs(tauR_1[ii]-0.5)<0.1:
        tau_0_5_index = ii
    elif abs(tauR_1[ii]-5)<0.1:
        tau_5_index = ii
        
T_1_tau_0_5 = T_1[tau_0_5_index]
Pe_1_tau_0_5 = Pe_1[tau_0_5_index]
Ne = Ne_ideal_gases(Pe_1_tau_0_5, T_1_tau_0_5)
n_vector = populations_finder(Pe_1_tau_0_5, T_1_tau_0_5)
n_HI = n_vector[1]
n_levels = n_levels_finder(n_HI, T_1_tau_0_5)

df_model_1 = pd.DataFrame(columns=['tauR', 'n(H-)', 'n(HI)', 'n(HII)', 'Ne', 'n(HI, n=1)', 'n(HI, n=2)', 'n(HI, n=3)'])
new_row = [0.5, n_vector[0], n_vector[1], n_vector[2], Ne, n_levels[0], n_levels[1], n_levels[2]]
df_model_1.loc[len(df_model_1)] = new_row

T_1_tau_5 = T_1[tau_5_index]
Pe_1_tau_5 = Pe_1[tau_5_index]
Ne = Ne_ideal_gases(Pe_1_tau_5, T_1_tau_5)
n_vector = populations_finder(Pe_1_tau_5, T_1_tau_5)
n_HI = n_vector[1]
n_levels = n_levels_finder(n_HI, T_1_tau_5)

new_row = [5, n_vector[0], n_vector[1], n_vector[2], Ne, n_levels[0], n_levels[1], n_levels[2]]
df_model_1.loc[len(df_model_1)] = new_row

print('1) Estrella con Teff = 5000 K')
print(df_model_1)



# Estrella con Teff = 8000 K

tauR_2 = 10**logtauR_2

for ii in range(len(tauR_2)):
    if abs(tauR_2[ii]-0.5)<0.1:
        tau_0_5_index = ii
    elif abs(tauR_2[ii]-5)<0.1:
        tau_5_index = ii
        
T_2_tau_0_5 = T_2[tau_0_5_index]
Pe_2_tau_0_5 = Pe_2[tau_0_5_index]
Ne = Ne_ideal_gases(Pe_2_tau_0_5, T_2_tau_0_5)
n_vector = populations_finder(Pe_2_tau_0_5, T_2_tau_0_5)
n_HI = n_vector[1]
n_levels = n_levels_finder(n_HI, T_2_tau_0_5)

df_model_2 = pd.DataFrame(columns=['tauR', 'n(H-)', 'n(HI)', 'n(HII)', 'Ne', 'n(HI, n=1)', 'n(HI, n=2)', 'n(HI, n=3)'])
new_row = [0.5, n_vector[0], n_vector[1], n_vector[2], Ne, n_levels[0], n_levels[1], n_levels[2]]
df_model_2.loc[len(df_model_2)] = new_row

T_2_tau_5 = T_1[tau_5_index]
Pe_2_tau_5 = Pe_1[tau_5_index]
Ne = Ne_ideal_gases(Pe_2_tau_5, T_2_tau_5)
n_vector = populations_finder(Pe_2_tau_5, T_2_tau_5)
n_HI = n_vector[1]
n_levels = n_levels_finder(n_HI, T_2_tau_5)

new_row = [5, n_vector[0], n_vector[1], n_vector[2], Ne, n_levels[0], n_levels[1], n_levels[2]]
df_model_2.loc[len(df_model_2)] = new_row

print('\n2) Estrella con Teff = 8000 K')
print(df_model_2)
