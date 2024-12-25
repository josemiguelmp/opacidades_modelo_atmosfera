#%%

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
plt.ylabel('Temperature [K]')
plt.legend()
plt.savefig('Figures/logtauR_T.pdf')

# Pe vs log(tau)
plt.figure(figsize=(10, 8))
plt.plot(logtauR_1, Pe_1, 'b-', label='$T_{eff}$ = 5000 K')
plt.plot(logtauR_2, Pe_2, 'r-', label='$T_{eff}$ = 8000 K')
plt.xlabel('log $\\tau_{Ross}$')
plt.ylabel('Electronic pressure [dyn/cm$^2$]')
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
plt.ylabel('Temperature [K]')
plt.title('$T_{eff}$ = 5000 K')
plt.legend()
plt.savefig('Figures/Temperature_gray_body_1.pdf')

plt.figure(figsize=(10, 8))
plt.plot(logtauR_2, T_gris_2, 'g-', label='T gray body')
plt.plot(logtauR_2, T_2, 'b-', label='T modelo MARCS')
plt.xlabel('log $\\tau_{Ross}$')
plt.ylabel('Temperature [K]')
plt.title('$T_{eff}$ = 8000 K')
plt.legend()
plt.savefig('Figures/Temperature_gray_body_2.pdf')

