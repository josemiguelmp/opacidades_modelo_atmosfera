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
plt.grid()
plt.savefig('Figures/r_logtauR.pdf')

# Temperature vs log(tau)
plt.figure(figsize=(10, 8))
plt.plot(logtauR_1, T_1, 'b-', label='$T_{eff}$ = 5000 K')
plt.plot(logtauR_2, T_2, 'r-', label='$T_{eff}$ = 8000 K')
plt.xlabel('log $\\tau_{Ross}$')
plt.ylabel('T [K]')
plt.legend()
plt.grid()
plt.savefig('Figures/logtauR_T.pdf')

# Pe vs log(tau)
plt.figure(figsize=(10, 8))
plt.plot(logtauR_1, Pe_1, 'b-', label='$T_{eff}$ = 5000 K')
plt.plot(logtauR_2, Pe_2, 'r-', label='$T_{eff}$ = 8000 K')
plt.xlabel('log $\\tau_{Ross}$')
plt.ylabel('$P_e$ [dyn/cm$^2$]')
plt.yscale('log')
plt.legend()
plt.grid()
plt.savefig('Figures/logtauR_Pe.pdf')

# Pe/Pg vs log(tau)
plt.figure(figsize=(10, 8))
plt.plot(logtauR_1, Pe_1/Pg_1, 'b-', label='$T_{eff}$ = 5000 K')
plt.plot(logtauR_2, Pe_2/Pg_2, 'r-', label='$T_{eff}$ = 8000 K')
plt.xlabel('log $\\tau_{Ross}$')
plt.ylabel('$P_e/P_g$')
plt.yscale('log')
plt.legend()
plt.grid()
plt.savefig('Figures/logtauR_Pe_Pg.pdf')

# Prad/Pg vs log(tau)
plt.figure(figsize=(10, 8))
plt.plot(logtauR_1, Prad_1/Pg_1, 'b-', label='$T_{eff}$ = 5000 K')
plt.plot(logtauR_2, Prad_2/Pg_2, 'r-', label='$T_{eff}$ = 8000 K')
plt.xlabel('log $\\tau_{Ross}$')
plt.ylabel('$P_{rad}/P_g$')
plt.yscale('log')
plt.legend()
plt.grid()
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
    f = c / ldo
    sigma = sigma_ff_HI(Z, ldo, T)
    return sigma * Ne * n_HII * ( 1 - np.exp( -h * f / (k_B * T) ) )


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
    
#freq = c / ldo 
#return sigma * ( 1 - np.exp( (-h * freq)/(k_B * T) ) )

#kbb[i][j] = sigbb[j]*n_boltz[j][i][0]*(1-np.exp((-h*nu_bb[j])/(kb_cgs*T)))

def lambda_Rydberg(l, u):
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
    sigma = sigma_ff_Hneg(ldo, T)
    return sigma * Pe * n_HI


# Opacidad bound-free del H-
def sigma_bf_Hneg(ldo, n_Hneg):
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
    
    sigma = (a0 + a1*ldo_A + a2*ldo_A**2 + a3*ldo_A**3 + a4*ldo_A**4 + a5*ldo_A**5 + a6*ldo_A**6) * 1e-18 / n_Hneg
    
    cut_index = len(sigma)
    for jj in np.arange(len(sigma)):
        if sigma[jj]<1e-30:
            cut_index = jj
            break
    
    physical_sigma = np.concatenate( ( sigma[0:cut_index], np.zeros((len(sigma) - cut_index)) ), 0 )

    return physical_sigma

def kappa_bf_Hneg(ldo, T, Pe, n_vector):
    n_Hneg, n_HI, n_HII = n_vector
    sigma = sigma_bf_Hneg(ldo, n_Hneg)
    theta = 5040 / T
    kappa = 4.158e-10 * sigma * Pe * theta**(5/2) * 10**(0.754*theta) / n_HI
    return kappa


# =============================================================================
# Opacidad electrones
# =============================================================================

sigma_e = 6.648e-25         # Gray

def kappa_e(Ne):
    return Ne * sigma_e





# %%

# =============================================================================
# Gráficas de cross sections
# =============================================================================

# Buscamos el índice para el cual tau = 1, en ambos modelos
for ii in range(len(tauR_1)):
    if abs(tauR_1[ii]-1)<0.1:
        tau_1_index = ii

for ii in range(len(tauR_2)):
    if abs(tauR_2[ii]-1)<0.1:
        tau_2_index = ii
        
T_1_tau_1 = T_1[tau_1_index]
Pe_1_tau_1 = Pe_1[tau_1_index]
Ne_1_tau_1 = Ne_ideal_gases(Pe_1_tau_1, T_1_tau_1)
n_vector_1 = populations_finder(Pe_1_tau_1, T_1_tau_1)
n_HI = n_vector_1[1]
n_levels_1 = n_levels_finder(n_HI, T_1_tau_1)
n1_1, n2_1, n3_1 = n_levels_1

T_2_tau_1 = T_2[tau_2_index]
Pe_2_tau_1 = Pe_2[tau_2_index]
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
plt.title('$\sigma_{ff}$ (HI)')

plt.plot(lambda_array_A, sigma_ff_HI_1, label='$T_{eff}$ = 5000 K')
plt.plot(lambda_array_A, sigma_ff_HI_2, label='$T_{eff}$ = 8000 K')

plt.yscale('log')
plt.xscale('log')
plt.ylabel('$\sigma_{ff}$ [cm$^2$]')
plt.xlabel('$\lambda$ [$\mathrm{\AA}$]')
plt.legend()
plt.grid()
plt.savefig('Figures/sigma_ff_HI.pdf')


sigma_ff_Hneg_1 = sigma_ff_Hneg(ldo=lambda_array_cm, T=T_1_tau_1)
sigma_ff_Hneg_2 = sigma_ff_Hneg(ldo=lambda_array_cm, T=T_2_tau_1)

plt.figure(figsize=(10, 8))
plt.title('$\sigma_{ff}$ (H$^-$)')

plt.plot(lambda_array_A, sigma_ff_Hneg_1, label='$T_{eff}$ = 5000 K')
plt.plot(lambda_array_A, sigma_ff_Hneg_2, label='$T_{eff}$ = 8000 K')

plt.yscale('log')
plt.xscale('log')
plt.ylabel('$\sigma_{ff}$ [cm$^2$]')
plt.xlabel('$\lambda$ [$\mathrm{\AA}$]')
plt.legend()
plt.grid()
plt.savefig('Figures/sigma_ff_Hneg.pdf')





# %%

# Sigma para las transiciones b-f desde los 3 niveles de HI
sigma_bf_HI_n1 = sigma_bf_HI(Z=1, n=1, ldo = lambda_array_cm)
sigma_bf_HI_n2 = sigma_bf_HI(Z=1, n=2, ldo = lambda_array_cm)
sigma_bf_HI_n3 = sigma_bf_HI(Z=1, n=3, ldo = lambda_array_cm)

sigma_bf_Hneg_ar = sigma_bf_Hneg(ldo=lambda_array_cm, n_Hneg=n_vector_1[0])



plt.figure(figsize=(10, 8))
plt.title('$\sigma_{bf}$')

plt.plot(lambda_array_A, sigma_bf_HI_n1, label='HI, n=1')      # Válido para los dos modelos
plt.plot(lambda_array_A, sigma_bf_HI_n2, label='HI, n=2')
plt.plot(lambda_array_A, sigma_bf_HI_n3, label='HI, n=3')
plt.plot(lambda_array_A, sigma_bf_Hneg_ar, label='H$^-$')


plt.yscale('log')
plt.xscale('log')
plt.ylabel('$\sigma_{bf}$ [cm$^2$]')
plt.xlabel('$\lambda$ [$\mathrm{\AA}$]')
#plt.ylim(1e-18, 1e-15)
plt.legend()
plt.grid()
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
plt.title('Star with $T_{eff}$ = 5000 K')

plt.plot(lambda_array_A, kappa_ff_HI_1, label='f-f HI')
plt.plot(lambda_array_A, kappa_bf_HI_n1_1, label='b-f HI, n = 1')
plt.plot(lambda_array_A, kappa_bf_HI_n2_1, label='b-f HI, n = 2')
plt.plot(lambda_array_A, kappa_bf_HI_n3_1, label='b-f HI, n = 3')
plt.plot(lambda_array_A, kappa_ff_Hneg_1, label='f-f H$^-$')
plt.plot(lambda_array_A, kappa_bf_Hneg_1, label='b-f H$^-$')
plt.plot(lambda_array_A, kappa_e_1, label='e$^-$ scattering')
plt.plot(lambda_array_A, kappa_total_1, label='Total opacity')

plt.yscale('log')
plt.xscale('log')
plt.ylabel('$\kappa$ [cm$^{-1}$]')
plt.xlabel('$\lambda$ [$\mathrm{\AA}$]')
plt.legend()
plt.grid()
plt.savefig('Figures/kappa_1.pdf')


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
plt.title('Star with $T_{eff}$ = 8000 K')

plt.plot(lambda_array_A, kappa_ff_HI_2, label='f-f HI')
plt.plot(lambda_array_A, kappa_bf_HI_n1_2, label='b-f HI, n = 1')
plt.plot(lambda_array_A, kappa_bf_HI_n2_2, label='b-f HI, n = 2')
plt.plot(lambda_array_A, kappa_bf_HI_n3_2, label='b-f HI, n = 3')
plt.plot(lambda_array_A, kappa_ff_Hneg_2, label='f-f H$^-$')
plt.plot(lambda_array_A, kappa_bf_Hneg_2, label='b-f H$^-$')
plt.plot(lambda_array_A, kappa_e_2, label='e$^-$ scattering')
plt.plot(lambda_array_A, kappa_total_2, label='Total opacity')

plt.yscale('log')
plt.xscale('log')
plt.ylabel('$\kappa$ [cm$^{-1}$]')
plt.xlabel('$\lambda$ [$\mathrm{\AA}$]')
plt.legend()
plt.grid()
plt.savefig('Figures/kappa_2.pdf')



plt.figure(figsize=(10, 8))

plt.plot(lambda_array_A, kappa_total_1, label='$\kappa_{total}, T_{eff}$ = 5000 K')
plt.plot(lambda_array_A, kappa_total_2, label='$\kappa_{total}, T_{eff}$ = 8000 K')


plt.yscale('log')
plt.xscale('log')
plt.ylabel('$\kappa$ [cm$^{-1}$]')
plt.xlabel('$\lambda$ [$\mathrm{\AA}$]')
plt.legend()
plt.grid()
plt.savefig('Figures/kappa_total_comparison.pdf')




# %%

# =============================================================================
# Tabla de opacidades
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
# Estrella con Teff = 5000 K
kappa_ff_HI_2    = kappa_ff_HI(Z=1, ldo=lambdas_tabla_cm, T=T_2_tau_1, Ne=Ne_2_tau_1, n_HII=n_vector_2[2])
kappa_bf_HI_n1_2 = kappa_bf_HI(Z=1, n=1, ldo=lambdas_tabla_cm, T=T_2_tau_1, ni=n1_2)
kappa_bf_HI_n2_2 = kappa_bf_HI(Z=1, n=2, ldo=lambdas_tabla_cm, T=T_2_tau_1, ni=n2_2)
kappa_bf_HI_n3_2 = kappa_bf_HI(Z=1, n=3, ldo=lambdas_tabla_cm, T=T_2_tau_1, ni=n3_2)
kappa_ff_Hneg_2  = kappa_ff_Hneg(ldo=lambdas_tabla_cm, T=T_2_tau_1, Pe=Pe_2_tau_1, n_HI=n_vector_2[1])
kappa_bf_Hneg_2  = kappa_bf_Hneg(ldo=lambdas_tabla_cm, T=T_2_tau_1, Pe=Pe_2_tau_1, n_vector=n_vector_2)
kappa_e_2        = kappa_e(Ne_2_tau_1) * np.ones(len(lambdas_tabla_cm))

data_2 = [kappa_ff_HI_1, kappa_bf_HI_n1_2, kappa_bf_HI_n2_2, kappa_bf_HI_n3_2, kappa_ff_Hneg_2, kappa_bf_Hneg_2, kappa_e_2]
df_tabla_final_2 = pd.DataFrame(data_2, columns=lambdas_tabla_A)
df_tabla_final_2.index = row_names
print('\n2) Estrella con Teff = 8000 K')
print(df_tabla_final_2)


# %%

# =============================================================================
# Tabla de opacidades b-b del HI
# =============================================================================

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