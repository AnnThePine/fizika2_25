import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def katrs(): 
    saraksts = []
    temperaturas = []
    for i in range(1,9):
        df = pd.read_csv(f"ld3_{i}.txt", sep='\t', decimal=',', encoding='utf-8-sig')
        karsts = np.nanmean(df["Temperature, Ch P3 (°C) Run #1"])
        auksts = np.nanmean(df["Temperature, Ch P2 (°C) Run #1"])
        temperaturas.append([auksts, karsts])
        df = df.drop(columns=["Time (s) Run #1", "Angle (rad) Run #1","Angular Velocity (rad/s) Run #1","Angular Acceleration (rad/s²) Run #1","Velocity (m/s) Run #1","Acceleration (m/s²) Run #1","Temperature, Ch P3 (°C) Run #1", "Temperature, Ch P2 (°C) Run #1"])
        df = df.dropna()
        df.reset_index(drop=True, inplace=True)
        saraksts.append(df)

    return saraksts, temperaturas

print(katrs())
    
def ielasi(filename): 
    df = pd.read_csv(filename, sep='\t', decimal=',', encoding='utf-8-sig')
    return df

def Spied_tilp(x_col,y_col):
    saraksts, temp = katrs()

    num_files = len(saraksts)
    cols = 4  # chocolate bar width (number of plots in a row)
    rows = (num_files + cols - 1) // cols  # auto-calculate needed rows

    fig, axs = plt.subplots(rows, cols, figsize=(16, 8), sharex=False)
    axs = axs.flatten()  # make it easy to index

    for i, df in enumerate(saraksts):
        axs[i].plot(df[x_col], df[y_col], label=f'Dataset {i+1}')
        axs[i].set_title(f'Dataset {i+1}')
        axs[i].set_xlabel('Time (s)')
        axs[i].set_ylabel('Pressure (kPa)')
        axs[i].grid(True)

    # Hide any unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    plt.show()

# Call the plot function
Spied_tilp("Position (m) Run #1",'Absolute Pressure (kPa) Run #1')

