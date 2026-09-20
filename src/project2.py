# NOTE: Only the input file path below was updated (it originally pointed to
# an absolute path on the author's own computer: F:\...\project.xlsx). It now
# points to ../data/raw/project.xlsx, matching this repository's folder
# layout. No other logic was changed from the original script.

import openpyxl
import pandas as pd
import math
import statistics as st
from scipy import integrate
import numpy as np

def integrand1(y,alfa):
    return y**(alfa - 1) * np.exp(-y)
def integrand2(x,gamma_of_alfa,alfa,beta):
    return (1 / ((beta**round(alfa))*gamma_of_alfa)) * (x**(alfa - 1)) * (np.exp(-x / beta))
def calculate_SPI_month(x):
    x_bar = st.mean(x)
    n=0
    a=0
    m=0
    c0 = 2.535537
    c1 = 0.802853
    c2 = 0.030328
    d1 = 3.432788
    d2 = 0.189269
    d3 = 0.003308
    for i in x:
        if i!= 0:
            a=a+ math.log(i)
        if i==0:
            m=m+1
    n = len(x)
    #print(n)
    if n>0:
        A = math.log(x_bar)-(a/n)
    else:
        A = 0
    #print (A)
    if ((1 + (4 * A) / 3) < 0):
        alfa = 0.001
    else:
        alfa = (1 / (4 * A)) * (1 + math.sqrt(1 + (4 * A) / 3))
    beta =round( x_bar/alfa)
    if beta == 0:
        return 0
    gamma, _ = integrate.quad(integrand1, 0, np.inf, args=(alfa,))
    gamma_of_alfa = round(gamma)
    if gamma_of_alfa == 0 or gamma_of_alfa == 1 :
        return 0
    max_of_x = max(x)
    F, _= integrate.quad(integrand2, 0, max_of_x, args=(gamma_of_alfa, alfa, beta))
    F_of_x = round(F, 5)
    #print(F_of_x)
    q = m/n
    H_of_x = q + (1-q)*(F_of_x)
    H_of_x_round = round(q + (1-q)*(F_of_x),2)
    if (0<H_of_x_round<=0.5):
        t = math.sqrt(math.log(1/(H_of_x ** 2)))
        SPI = -(t - ((c0 + c1*t + (c2 * (t**2)))/1+d1*t+d2*(t**2)+d3*(t*(t**2))))
    elif (0.5 < H_of_x_round <= 1):
        t = math.sqrt(math.log(1/((1 - H_of_x)**2)))
        SPI = t - ((c0 + c1*t + (c2 * (t**2)))/1+d1*t+d2*(t**2)+d3*(t*(t**2)))
    return SPI
path = "../data/raw/project.xlsx"
wb = openpyxl.load_workbook(path)
sheet = wb.active
max_row = sheet.max_row
time=2
for i in range(1,13):#for each year
    for j in range (1,13):#for each month
        rain = []
        for row_number in range(time, max_row + 1):
            cell = sheet.cell(row=row_number, column=1).value
            if cell:
                date = pd.to_datetime(cell, format='%m/%d/%Y')
                if date.month == j:
                    rain_value_raw = sheet.cell(row=row_number, column=6).value  # Assuming rain values are in column 2
                    rain_value = float(rain_value_raw.replace(' mm', ''))  # Remove ' mm' and convert to float
                    rain.append(rain_value)
                    #print(date)
                    time=time+1

                else:
                    print(calculate_SPI_month(rain))
                    #print (rain)
                    break
