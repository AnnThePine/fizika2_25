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

def shoelace_area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def polygon_centroid(x, y):
    # Make sure the polygon is closed
    if x[0] != x[-1] or y[0] != y[-1]:
        x = np.append(x, x[0])
        y = np.append(y, y[0])

    A = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    Cx = (1 / (6 * A)) * np.sum((x + np.roll(x, -1)) * (x * np.roll(y, -1) - np.roll(x, -1) * y))
    Cy = (1 / (6 * A)) * np.sum((y + np.roll(y, -1)) * (x * np.roll(y, -1) - np.roll(x, -1) * y))

    return Cx, Cy

def Spied_tilp(x_col,y_col):
    saraksts, temp = katrs()

    num_files = len(saraksts)
    cols = 4  # chocolate bar width (number of plots in a row)
    rows = (num_files + cols - 1) // cols  # auto-calculate needed rows

    fig, axs = plt.subplots(rows, cols, figsize=(16, 8), sharex=False)
    axs = axs.flatten()  # make it easy to index

    for i, df in enumerate(saraksts):
        temperaturas = temp[i]
        auksts = int(temperaturas[0])
        karsts = int(temperaturas[1])

        x = df[x_col].to_numpy()
        y = df[y_col].to_numpy()

        # Sort by x to keep the polygon clean
        x_closed = np.append(x, x[0])
        y_closed = np.append(y, y[0])

        area = shoelace_area(x_closed, y_closed)

        axs[i].fill(x_closed, y_closed, color='lightgreen', alpha=0.4, label='Polygon Area')

        centroid_x, centroid_y = polygon_centroid(x_closed, y_closed)
        axs[i].text(centroid_x, centroid_y, f"Darbs:{area:.2f}kJ", fontsize=12, fontweight='bold', color='darkgreen', ha='center', va='center')

        axs[i].plot(df[x_col], df[y_col], color = 'darkgreen')
        axs[i].set_title(f'Siltā ūdens temp: {karsts}°C,\n aukstā ūdens temp: {auksts}°C,\n ΔT: {karsts-auksts}°C')
        axs[i].set_xlabel('Pozīcija (m)')
        axs[i].set_ylabel('Spiediens (kPa)')
        axs[i].grid(True)

    # Hide any unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    plt.show()

# Call the plot function
Spied_tilp("Position (m) Run #1",'Absolute Pressure (kPa) Run #1')

