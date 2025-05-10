import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans

plt.rcParams['font.family'] = 'Times New Roman'

tilp = lambda h0,h: 3.14*0.112*(0.04626/2)**2+3.14*(h0+h)*(0.0325/2)**2

sar1 = [33,35,35,35,39,37,38,38]
sar2 = [18,19,13,19,20,20,27,27]

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
        df['tilpums'] = tilp(sar1[i-1],df['Position (m) Run #1'])
        saraksts.append(df)

    return saraksts, temperaturas

def line_intersection(p1, p2, p3, p4):
    """Finds the intersection point of lines p1-p2 and p3-p4"""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    if np.isclose(denom, 0):
        # Lines are parallel or very close to it
        return ((x1 + x2)/2, (y1 + y2)/2)  # fallback to midpoint of first line

    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom

    return (px, py)

def correct_polygon_tail(x, y, tail_length=10):
    """
    Replace the last long segment (tail) with an intersection point
    between two approximated lines at the end of the polygon.
    """
    # Choose points forming the two edges that make up the final corner
    p1 = (x[-tail_length*2], y[-tail_length*2])
    p2 = (x[-tail_length], y[-tail_length])
    p3 = (x[-1], y[-1])
    p4 = (x[0], y[0])  # Wrap around to the start

    intersect = line_intersection(p1, p2, p3, p4)
    if intersect is not None:
        # Replace the last point with the intersection
        x = np.append(x[:-1], intersect[0])
        y = np.append(y[:-1], intersect[1])
    return x, y


def get_corrected_corners(x, y):
    x = np.array(x)
    y = np.array(y)

    n = len(x)
    q1_x = x[:n//4]
    q1_y = y[:n//4]
    q4_x = x[3*n//4:]
    q4_y = y[3*n//4:]

    # Get threshold x from the first quarter (min x for typical y values)
    y_mean = np.mean(q1_y)
    y_tol = np.std(y) * 0.3  # small tolerance for "similar y"
    mask_q1 = np.abs(q1_y - y_mean) < y_tol
    x_thresh = np.min(q1_x[mask_q1])

    # Remove tail: drop all last-quarter points with x < x_thresh AND similar y
    tail_mask = np.ones(n, dtype=bool)  # start with all True
    for i in range(3 * n // 4, n):
        if x[i] < x_thresh and abs(y[i] - y_mean) < y_tol:
            tail_mask[i] = False

    x_filtered = x[tail_mask]
    y_filtered = y[tail_mask]

    # Use convex hull and k-means for 4 corners
    points = np.column_stack((x_filtered, y_filtered))
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    # Cluster to 4 corners
    kmeans = KMeans(n_clusters=4, n_init=10)
    kmeans.fit(hull_points)
    centers = kmeans.cluster_centers_

    # Sort corners clockwise
    cx, cy = centers[:, 0], centers[:, 1]
    angles = np.arctan2(cy - np.mean(cy), cx - np.mean(cx))
    sort_idx = np.argsort(angles)

    sorted_corners = [(cx[i], cy[i]) for i in sort_idx]
    return sorted_corners
    
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

    aaindex = [
    "T_s,℃",
    "T_a,℃",
    "V_a,ml",
    "V_c,ml",
    "V_d,ml",
    "p_c,kPa",
    "p_d,kPa",
    "Q_(c→d),J",
    "Q_(b→c),J",
    "Q_1,J",
    "A_t,J",
    "A_m,J",
    "e_t,%",
    "e_r,%"]

    aaaaa = pd.DataFrame(index=aaindex, columns = [1,2,3,4,5,6,7,8])

    fig, axs = plt.subplots(rows, cols, figsize=(15,7), sharex=False)
    axs = axs.flatten()  # make it easy to index

    for i, df in enumerate(saraksts):
        temperaturas = temp[i]
        auksts = int(temperaturas[0])
        karsts = int(temperaturas[1])

        x = df[x_col].to_numpy()
        y = df[y_col].to_numpy()

        corners = get_corrected_corners(x, y)
        print(corners)

        # Sort by x to keep the polygon clean
        x_closed = np.append(x, x[0])
        y_closed = np.append(y, y[0])

        area = shoelace_area(x_closed, y_closed)*1000

        aaaaa.loc["T_s,℃", i] = karsts
        aaaaa.loc["T_a,℃", i] = auksts
        aaaaa.loc["V_a,ml",i] = df['tilpums'].iloc[0]
        aaaaa.loc["V_d,ml",i] = auksts+aaaaa.loc["V_a,ml",i]/aaaaa.loc["T_a,℃", i]
        #aaaaa.loc["V_c,ml", i] = 

        axs[i].fill(x_closed, y_closed, color='lightgreen', alpha=0.4, label='Polygon Area')

        centroid_x, centroid_y = polygon_centroid(x_closed, y_closed)
        axs[i].text(centroid_x, centroid_y, f"Darbs:{area:.3f}J", fontsize=12, fontweight='bold', color='darkgreen', ha='center', va='center')

        axs[i].plot(df[x_col], df[y_col], color = 'darkgreen')
        axs[i].set_title(f'Siltā ūdens temp: {karsts}°C,\n aukstā ūdens temp: {auksts}°C,\n ΔT: {karsts-auksts}°C')
        axs[i].set_xlabel('Tilpums (m^3)')
        axs[i].set_ylabel('Spiediens (kPa)')
        axs[i].grid(True)

    # Hide any unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    plt.show()

def Spied_tilp_first(x_col, y_col, data):
    saraksts, temp = katrs()

    if len(saraksts) == 0:
        print("No data available.")
        return
    
    a = data-1

    df = saraksts[a]
    temperaturas = temp[a]
    auksts = int(temperaturas[0])
    karsts = int(temperaturas[1])

    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()

    x_closed = np.append(x, x[0])
    y_closed = np.append(y, y[0])

    area = shoelace_area(x_closed, y_closed) * 1000

    fig, ax = plt.subplots(figsize=(7,5))

    ax.fill(x_closed, y_closed, color='lightgreen', alpha=0.4, label='Polygon Area')
    ax.plot(x, y, color='darkgreen')

    centroid_x, centroid_y = polygon_centroid(x_closed, y_closed)
    ax.text(centroid_x, centroid_y, f"Darbs:{area:.3f}J", fontsize=12, fontweight='bold', color='darkgreen', ha='center', va='center')

    ax.set_title(f'Siltā ūdens temp: {karsts}°C,\n aukstā ūdens temp: {auksts}°C,\n ΔT: {karsts-auksts}°C')
    ax.set_xlabel('Tilpums (m^3)')
    ax.set_ylabel('Spiediens (kPa)')
    ax.grid(True)

    plt.tight_layout()
    plt.show()

# Call the plot function
Spied_tilp('tilpums','Absolute Pressure (kPa) Run #1')
Spied_tilp_first('tilpums','Absolute Pressure (kPa) Run #1',1)
