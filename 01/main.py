import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "Times New Roman"

plt.rcParams["font.size"] = 14
plt.rcParams["axes.grid"] = True
plt.rcParams["axes.axisbelow"] = True


def Dati(filename):    
    data = pd.read_csv(filename, delimiter=';', encoding='utf-8-sig')
    data = data.apply(lambda x: x.str.replace(',', '.') if x.dtype == 'object' else x)
    names = data.columns
    for name in names:
        data[name] = pd.to_numeric(data[name], errors='coerce')
    return data

def hip1(dati):
    spiediens = dati['Absolute Pressure (kPa) Run #1']
    temperatura = dati["Temperature (°C) Run #1"]
    laiks = dati['Time (s) Auto']
    
    plt.figure()
    plt.scatter(laiks,spiediens, s=15)
    plt.xlabel("Laiks (s)")
    plt.ylabel('Spiediens (kPa)')
    plt.title("Spiediena šļircē atkarība no laika")
    
    plt.figure()
    plt.scatter(laiks,temperatura, s=15)
    plt.xlabel("Laiks (s)")
    plt.ylabel('Temperatūra (°C)')
    plt.title("Temperatūras šļircē atkarība no laika")

def Hip2():
    def hip2(dati):
        spiediens = dati['Absolute Pressure (kPa) Run #1']
        temperatura = dati["Temperature (°C) Run #1"]
        tilpums = dati['V (ml) Set']
        
        plt.figure()
        plt.scatter(tilpums,spiediens, s=15)
        plt.xlabel("Tilpums (ml)")
        plt.ylabel('Spiediens (kPa)')
        plt.title("Gaisa spiediena šļircē atkarība no tilpuma")
        
        tilpumss = []
        for i in tilpums:
            tilpumss.append(1/i)
        tilpumss = np.array(tilpumss)
        
        # print(tilpums)
        # print(tilpumss)
        
        kludaV = 1  # Given error in volume (ml)
        kludaP = 2  # Given error in pressure (kPa)

        # Compute least squares fit
        n = len(tilpumss)
        sumTilp = np.sum(tilpumss)
        sumSpied = np.sum(spiediens)
        sumTilp2 = np.sum(tilpumss**2)
        sumTilp_Spied = np.sum(tilpumss * spiediens)

        # Compute slope (a) and intercept (b)
        a = (n * sumTilp_Spied - sumTilp * sumSpied) / (n * sumTilp2 - sumTilp**2)
        b = (sumTilp2*sumSpied - sumTilp * sumTilp_Spied) /(n * sumTilp2 - sumTilp**2)

        def taisne(a,x,b):
            taisne = a*x+b
            return taisne
        
        S = np.sqrt(np.sum((spiediens - taisne(a,tilpumss,b))**2) / (n - 2))


        # **New: Error propagation for slope and intercept**
        a_error = S* np.sqrt((n/ (n * sumTilp2 - sumTilp**2)))
        b_error = S*np.sqrt((sumTilp2 / (n * sumTilp2 - sumTilp**2)))
        
        minn = min(tilpumss)-0.002
        maxx = max(tilpumss)+0.002
        
        x = np.linspace(minn,maxx,20)
        
        
        # print(a)
        # print(b)

        # Create plot
        plt.figure()
        plt.scatter(tilpumss, spiediens, color='blue', label='Eksperimentālie dati')
        plt.plot(x, taisne(a,x,b), color='red', label='Tendences taisne')
        plt.fill_between(x, taisne(a-a_error,x, b-b_error),  taisne(a+a_error,x,b+b_error), color='red', alpha=0.2, label="Taisnes kļūda")
        plt.xlim(left = minn, right = maxx)

        plt.xlabel("1/Tilpums (1/ml)")
        plt.ylabel('Spiediens (kPa)')
        plt.title("Gaisa spiediena šļircē atkarība no tilpuma ar p(x) taisni")
        plt.subplots_adjust(bottom=0.2)
        
        return x, taisne(a,x,b),taisne(a-a_error,x, b-b_error), taisne(a+a_error,x,b+b_error), tilpumss, spiediens
        


    #hip1(Dati("1.csv"))

    x, taisne, mintaisne, maxtaisne,tilpums, spiediens = hip2(Dati("2.csv"))

    x2, taisne2, mintaisne2, maxtaisne2,tilpums2, spiediens2 = hip2(Dati("3.csv"))

    def papildus():
        plt.figure()
        
        plt.scatter(tilpums2, spiediens2, color = "blue",  label='2. mērījums: 60ml',zorder = 3)
        plt.scatter(tilpums, spiediens, color = "red",label='1. mērījums: 50ml', zorder = 3)
        plt.plot(x, taisne, color='red', alpha=0.5)
        plt.fill_between(x, mintaisne, maxtaisne, color='orange', alpha=0.2)

        
        plt.plot(x2, taisne2, color='blue', alpha=0.5)
        plt.fill_between(x2, mintaisne2, maxtaisne2, color='cyan', alpha=0.2)
        
        plt.ylim(bottom=94)
        plt.xlim(0.015,0.042)

        plt.xlabel("1/Tilpums (1/ml)")
        plt.ylabel('Spiediens (kPa)')
        plt.title("Gaisa spiediena šļircē atkarība no tilpuma p(x) taisnes")
        plt.legend()

    papildus()

def taisne(a,x,b):
            taisne = a*x+b
            return taisne


def hip3():
    dati = Dati("4.csv")
        
    def taisnee(tilpumss, spiedienss):
        n = len(tilpumss)
        sumTilp = np.sum(tilpumss)
        sumSpied = np.sum(spiedienss)
        sumTilp2 = np.sum(tilpumss**2)
        sumTilp_Spied = np.sum(tilpumss * spiedienss)

        # Compute slope (a) and intercept (b)
        a = (n * sumTilp_Spied - sumTilp * sumSpied) / (n * sumTilp2 - sumTilp**2)
        b = (sumTilp2*sumSpied - sumTilp * sumTilp_Spied) /(n * sumTilp2 - sumTilp**2)

        
        S = np.sqrt(np.sum((spiedienss - taisne(a,tilpumss,b))**2) / (n - 2))


        # **New: Error propagation for slope and intercept**
        a_error = S* np.sqrt((n/ (n * sumTilp2 - sumTilp**2)))
        b_error = S*np.sqrt((sumTilp2 / (n * sumTilp2 - sumTilp**2)))
        
        minn = min(tilpumss)-0.002
        maxx = max(tilpumss)+0.002
        
        return a,b, a_error, b_error,maxx, minn
    
    di = {}
    for i in range (1,4):
        temperatura = dati[f"Temperature (°C) {i}"]
        spiediens = dati[f"Absolute Pressure (kPa) {i}"]
        di[f"temp{i}"] = temperatura
        di[f"spied{i}"] = spiediens
        a,b, a_error, b_error,maxx, minn = taisnee(spiediens, temperatura)
  
        
        x = np.linspace(0, 130, 100)
        di[f"a{i}"] = a
        di[f"b{i}"] = b
        di[f"taisne{i}"] = taisne (a,x,b)
        di[f"errpirm{i}"] = taisne(a-(a_error*(2/0.5)),x, b+(b_error*2))
        di[f"errpec{i}"] = taisne(a+(a_error*(2/0.5)),x,b-(b_error*2))
        di[f"berr{i}"] = b_error
        di[f"aerr{i}"] = a_error
        
        n=0.54/(8.31*a)
        #print(n)
        #print(b)
        di[f"n{i}"]=n
    
        
    T0 = np.average([di['b1'], di['b2'], di['b3']])

    kluda =T0-di['errpec1'][0]
    
    print(kluda)
    
        
    plt.figure()
    
    plt.fill_between(x, di["errpirm1"], di['errpec1'], color = "orange", alpha = 0.6)
    plt.plot(x,di["taisne1"], color = "red", label = f"T={di['a1']:.2f}p+{di['b1']:.0f}")
    plt.scatter(di['spied1'],di['temp1'], label = f"n={di['n1']:.3f}mol", color = 'red')
    
    plt.fill_between(x, di["errpirm2"], di['errpec2'], color = "cyan", alpha = 0.6)
    plt.plot(x,di["taisne2"], color = "blue", label = f"T={di['a2']:.2f}p+{di['b2']:.0f}")
    plt.scatter(di['spied2'],di['temp2'], label = f"n={di['n2']:.3f}mol", color = 'blue')
    
    
    plt.fill_between(x, di["errpirm3"], di['errpec3'], color = "pink", alpha = 0.6)
    plt.plot(x,di["taisne3"], color = "purple",label = f"T={di['a3']:.2f}p+{di['b3']:.0f}")
    plt.scatter(di['spied3'],di['temp3'], label = f"n={di['n3']:.3f}mol", color = 'purple')
    
    
    plt.scatter(0,T0, color = 'black',label = 'Iegūtā T(0) vērtība' )
    plt.errorbar(0,T0,kluda,color = 'black',alpha=0.6)
    

   
    
    
    plt.xlabel('Spiediens (kPa)')
    plt.ylabel('Temperatūra (°C)')
    plt.title("Temperatūras atkarība no spiediena")
    plt.legend()
    plt.xlim(-5,5)
    plt.ylim(-285, -250)
    
    #plt.xlim(right = 130)
    #plt.xlim(-300, -200)



hip3()


plt.tight_layout()
plt.show()