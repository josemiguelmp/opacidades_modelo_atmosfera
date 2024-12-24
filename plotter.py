import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import os


file_name = "t5000.dat"

with open(file_name, 'r') as file:
    lines = file.readlines()
    

for index in range(len(lines)):
    if lines[index] == "Model structure\n":
        index_table = index + 1
        break

data = [line.split() for line in lines[index_table+1:]]
column_names = lines[index_table].split()

df_model_structure = pd.DataFrame(data, columns=column_names, index=None)

numeral      = np.array(df_model_structure['k'],      dtype=float)            # numeral del punto
logtauR      = np.array(df_model_structure['lgTauR'], dtype=float)            # logaritmo de la profundidad óptica de Rosseland
logtau_5000  = np.array(df_model_structure['lgTau5'], dtype=float)            # logaritmo de la profundidad óptica a 5000 A
r            = np.array(df_model_structure['Depth'],  dtype=float)            # profundidad geométrica
T            = np.array(df_model_structure['T'],      dtype=float)            # Temperatura
Pe           = np.array(df_model_structure['Pe'],     dtype=float)            # Presión electrónica
Pg           = np.array(df_model_structure['Pg'],     dtype=float)            # Presión del gas
Prad         = np.array(df_model_structure['Prad'],   dtype=float)            # Presión de radiación


plt.figure(figsize=(10, 8))
plt.plot(r, logtauR, 'o')
plt.xlabel('Geometrical depth')
plt.ylabel('Logarithm of Rosseland optical depth')
plt.savefig('Figures/r_logtauR.pdf')

plt.figure(figsize=(10, 8))
plt.plot(logtauR, T, 'o')
plt.xlabel('Geometrical depth')
plt.ylabel('Temperature')
plt.gca().invert_xaxis()
plt.savefig('Figures/logtauR_T.pdf')

plt.figure(figsize=(10, 8))
plt.plot(logtauR, Pe, 'o')
plt.xlabel('Geometrical depth')
plt.ylabel('Electronic pressure')
plt.gca().invert_xaxis()
plt.savefig('Figures/logtauR_Pe.pdf')

plt.figure(figsize=(10, 8))
plt.plot(logtauR, Pe/Pg, 'o')
plt.xlabel('Geometrical depth')
plt.ylabel('$P_e/P_g$')
plt.gca().invert_xaxis()
plt.savefig('Figures/logtauR_Pe_Pg.pdf')

plt.figure(figsize=(10, 8))
plt.plot(logtauR, Prad/Pg, 'o')
plt.xlabel('Geometrical depth')
plt.ylabel('$P_{rad}/P_g$')
plt.gca().invert_xaxis()
plt.savefig('Figures/logtauR_Prad_Pg.pdf')
