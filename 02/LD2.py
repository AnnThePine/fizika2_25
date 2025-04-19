import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress
import os

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

df = pd.read_csv("Sitienissilst.csv", delimiter=';', encoding='utf-8-sig')
df = df.apply(lambda x: x.str.replace(',', '.') if x.dtype == 'object' else x)

da = pd.read_csv("stienisdziest.csv", delimiter=';', encoding='utf-8-sig')
da = da.apply(lambda x: x.str.replace(',', '.') if x.dtype == 'object' else x)

df = df.iloc[::16]
da = da.iloc[::10]



#time = list(df["Time (s) Run #1"])
time = list(df["Time (s) Run #1"].astype(float))

T_8 = df['Temperature 1, Ch P1 (°C) Run #1']
t8 = time
T_10 = df["Temperature 2, Ch P1 (°C) Run #1"]
t10 = time
T_12 = df['Temperature 3, Ch P1 (°C) Run #1']
t12 = time
T_13 = df['Temperature 4, Ch P1 (°C) Run #1']
t13 = time
T_1 = df['Temperature 1, Ch P4 (°C) Run #1']
t1 = time
T_2 = df['Temperature 2, Ch P4 (°C) Run #1']
t2 = time
T_4 = df['Temperature 3, Ch P4 (°C) Run #1']
t4 = time
T_6 = df['Temperature 4, Ch P4 (°C) Run #1']
t6 = time

def filter_spikes(data_list, laiks):
    filtered = []
    filteredlaiks = []
    prev_valid = None
    indeks = 0
    for val in data_list:

        val = float(val)  # Ensure the value is an integer

        if prev_valid is None or val <= prev_valid + 20:
            filtered.append(val)
            prev_valid = val
            filteredlaiks.append(laiks[indeks])
        
        indeks = indeks+1

    return filtered, filteredlaiks

T_8, t8 = filter_spikes(T_8, t8)
T_10,t10 = filter_spikes(T_10,t10)
T_12,t12 = filter_spikes(T_12,t12)
T_13,t13 = filter_spikes(T_13,t13)
T_1,t1 = filter_spikes(T_1,t1)
T_2,t2 =filter_spikes(T_2,t2)
T_4,t4 = filter_spikes(T_4,t4)
T_6,t6 = filter_spikes(T_6,t6)


#Izveido grafiku, kur laikā mainās temperatūra 8 sensoriem
plt.figure()
plt.plot(t1, T_1, label='2')
plt.plot(t2, T_2, label='3')
plt.plot(t4, T_4, label='1')
plt.plot(t6, T_6, label='6')
plt.plot(t8, T_8, label='8')
plt.plot(t10, T_10, label='10')
plt.plot(t12, T_12, label='12')
plt.plot(t13, T_13, label='13')




plt.grid(True)  # Enable grid


# Define tick positions
tick_positions = np.linspace(time[0], time[-1], 10, dtype=int)

plt.xticks(tick_positions)  # Set x-axis ticks

ax = plt.gca()  # Get current axes
ax.set_xticks(tick_positions)  # Ensure integer tick positions
ax.set_xticklabels([f"{int(tick)}" for tick in tick_positions])  # Force integer labels

plt.legend()
plt.ylabel('T, °C')
plt.xlabel('t, s')
plt.tight_layout()
#plt.savefig('t_atkarībā_no_laika.png')
plt.show()






#funkcija, kas aprēķina temperatūras gradientu

def tikai_slīpums(sekundes):
    caurumi = [0.02, 0.06, 0.14, 0.22, 0.3, 0.38, 0.46, 0.5]
    t_pie = df.iloc[sekundes, 2:].astype(float)
    slope, intercept, r_value, p_value, slope_stderr = linregress(caurumi, t_pie)
    return slope, slope_stderr

#TEMPERATŪRAS GRADIENTI LAIKĀ

slīpumi = []
kļūda_s = []
for i in range(len(df)):
    slope, slope_stderr = tikai_slīpums(i)
    kļūda_s.append(slope_stderr)
    slīpumi.append(slope)

#GRAFIKS TEMP GRADIENTA IZMAIŅAI LAIKĀ

#plt.scatter(time, slīpumi, marker='.', s=2)
#plt.savefig('slīpuma_k_atkarībā_no_laika.png')

#plt.show()





#VIDĒJĀ TEMPERATŪRAS GRADIENTA VĒRTĪBA KARSĒŠANAS LAIKĀ, kad temp gardients ir izlīdzinājies

const_temp_grad = slīpumi[-100:] 

const_temp_grad_kļūda = kļūda_s[-1000:]

# Aprēķinām vidējo vērtību
vid_temp_grad = np.mean(const_temp_grad)

vid_temp_grad_kļūda = np.mean(const_temp_grad_kļūda)

print("Vidējā temperatūras gradienta vērtība laikā no 3900 līdz 4200 s:", vid_temp_grad)
print("Vidējā temperatūras gradienta vērtības kļūda laikā no 3900 līdz 4200 s:", vid_temp_grad_kļūda)

#KĻŪDA VID TEMP GRADIENTAM






#divas funkcijas temperatūras gradienta grafika veidošanai

#BEZ TEKSTA

def slīpums_1(dati,sekundes):
    # """Performs linear regression and returns slope and intercept."""
    
    caurumi = [0.02, 0.06, 0.14, 0.22, 0.3, 0.38, 0.46, 0.5, ]
    # caurumi = [0.28, 0.36, 0.44, 0.48, 0, 0.04, 0.12, 0.2]

    x_fit = np.linspace(min(caurumi), max(caurumi), 100)
    #print(dati.iloc[sekundes])
    t_pie = dati.iloc[sekundes, 2:].astype(float)

    if len(dati) == len(df):
        sekundes = sekundes*16

    elif len(dati) == len(da):
        sekundes = sekundes*10

    t_pie = t_pie.sort_values()

    slope, intercept, r_value, p_value, slope_stderr = linregress(caurumi, t_pie)

    y = slope * x_fit + intercept

    plt.plot(caurumi, t_pie, linewidth=1, marker="o",label=f't={sekundes}s', zorder=1, alpha=0.8)
    # plt.plot(x_fit, y, linewidth=1, linestyle="--", label=f'T = {slope:.2f}N + {intercept:.2f}')
    # plt.scatter(caurumi, t_pie)

    plt.xlabel('x, m')
    plt.ylabel('T, °C')
    plt.grid(True)
    plt.legend()

    return slope, intercept, slope_stderr



#AR TEKSTU


def slīpums(dati, text):
    # """Performs linear regression and returns slope and intercept."""

    caurumi = [0.02, 0.06, 0.14, 0.22, 0.3, 0.38, 0.46, 0.5]
    # caurumi = [0.28, 0.36, 0.44, 0.48, 0, 0.04, 0.12, 0.2]

    x_fit = np.linspace(min(caurumi), max(caurumi), 100)
    sekunde = int(float(dati.iloc[-1, 1]))
    t_pie = dati.iloc[len(dati)-1, 2:].astype(float)
    t_pie = t_pie.sort_values()


    if len(t_pie) ==0:
        print(f"nav tādu datu")
        exit()

    slope, intercept, r_value, p_value, slope_stderr = linregress(caurumi, t_pie)

    y = slope * x_fit + intercept

    plt.plot(x_fit, y, linewidth=1, linestyle="--", label=f'T = {slope:.2f}x + {intercept:.2f}')
    plt.scatter(caurumi, t_pie, zorder=2, label=f't={sekunde}s, {text}')    # Higher zorder → on top

    plt.xlabel('x, m')
    plt.ylabel('T, °C')
    plt.grid(True)
    plt.legend()

    return slope, intercept, slope_stderr



#TEMPERATŪRAS GRADIENTA GRAFIKI

#temperatūras gradients karsēšanas gadījumā

# fig = slīpums_1(df,1, 'lightgray')
# fig = slīpums_1(df,10, 'silver')
# fig = slīpums_1(df,100, 'darkgrey')
# fig = slīpums_1(df,200, 'gray')
# fig = slīpums_1(df,300, 'grey')
# fig = slīpums_1(4000, 'dimgrey')
#fig = slīpums(4150, 'red', 'karsēšana')
fig = slīpums_1(df,1)
fig = slīpums_1(df,10)
fig = slīpums_1(df,100)
fig = slīpums_1(df,200)
fig = slīpums_1(df,300)


#plt.savefig('1_grafiks.png')
plt.show()



#temperatūras gradients dzesēšanas gadījumā


fig = slīpums_1(da, 1)
fig = slīpums_1(da,100)
fig = slīpums_1(da,200)
fig = slīpums_1(da,250)
fig = slīpums_1(da,300)
#fig = slīpums(da,6246, 'maroon', 'atdzišana')


#plt.savefig('2_grafiks.png')
plt.show()


#salīdzinājums dzesēšanas un karsēšanas gadījumam

fig = slīpums(df, 'karsēšana')
fig = slīpums(da, 'atdzišana')

#plt.savefig('3_grafiks.png')
plt.show()







s_4150, i_4150, e_4150 = slīpums(df, 'karsēšana')





#aprēķina siltumvadīšanas koeficientu k

def k(s):
    d = 0.025
    U = 5.5
    I = 3.5
    Q = U * I
    S = np.pi * (d/2)**2
    # k = ((-1) * Q)/(s * S)
    k = (-1) * (Q) * (1/s) * (1/S)
    return k
print(s_4150)
print(k(s_4150))



#mērījumi un sistemātiskās kļūdas

d = 0.025
U = 5.5
I = 3.5
Q = U * I
S = np.pi * (d/2)**2

dU = 0.1
dI = 0.01
dd = 0.001
ds = e_4150

print("temperatūras gradienta vērtība kļūda:", e_4150)


#ievietošansa metode absolūtās kļūdas aprēķināšanai 

k_U = k(s_4150) - ((-4) * (U + dU) * I)/(s_4150 * np.pi * d**2)
k_I = k(s_4150) - ((-4) * U * (I + dI))/(s_4150 * np.pi * d**2)
k_d = k(s_4150) - ((-4) * U * I)/(s_4150 * np.pi * (d + dd)**2)
k_s = k(s_4150) - ((-4) * U * I)/((s_4150 + ds) * np.pi * d**2)


#siltumvadīšanas koeficienta absolūtā kļūda un rel kļūda

d_k = np.sqrt(k_U**2 + k_I**2 + k_d**2 + k_s**2)
r_k = d_k/k(s_4150)

print(d_k, r_k)