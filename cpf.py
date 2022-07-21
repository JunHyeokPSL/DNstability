# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 16:16:16 2022

@author: dddddb
"""

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import time
import matplotlib.ticker as ticker
ax = plt.axes()
start = time.time()

B1 = np.loadtxt('Bus.txt',dtype = float, comments='#')  
L1 = np.loadtxt('Line.txt',dtype = float, comments='#')
#9-bus는 B1 = np.loadtxt('Bus(2).txt',dtype = float, comments='#')
#9-bus는 L1 = np.loadtxt('Line(2).txt',dtype = float, comments='#')


# Line data 정리
From = L1[:,0].astype(int)-1
To = L1[:,1].astype(int)-1
R = L1[:,2]
X = L1[:,3]
yline = 1/(R + 1j*X)
C = (1j/2)*L1[:,4]

# Bus data 정리
Bus_num = B1[:,0]
Bus_type = B1[:,1]
PG = B1[:,2]
QG = B1[:,3]
PD = B1[:,4]
QD = B1[:,5]
Vm = B1[:,6]
theta = B1[:,7]
#%% Ybus matrix
Y_bus = np.zeros((len(Bus_num),len(Bus_num)), dtype=complex)

for i in range(0,len(L1)):
    Y_bus[From[i],From[i]] = Y_bus[From[i],From[i]] + yline[i] + C[i]
    Y_bus[To[i],To[i]] = Y_bus[To[i],To[i]] + yline[i] + C[i]
    Y_bus[From[i],To[i]] = -yline[i]
    Y_bus[To[i],From[i]] = Y_bus[From[i],To[i]]
#%% PQ, PV 버스
Num_PV = 0
Num_PQ = 0
for j in range(0,len(Bus_type)):
    if Bus_type[j] == 2:
        Num_PV += 1
    elif Bus_type[j] == 3:
        Num_PQ += 1
#%% Ybus real/imag part, bus P,Q
G = np.real(Y_bus)
B = np.imag(Y_bus)

Pm = np.subtract(PG,PD)
Qm = np.subtract(QG,QD)
#%% Jacobian matrix, 기준오차설정, iteration
err = 1
p = 0

while err > 0.0001:
    
    p = p + 1
    
    Pcal = np.zeros(len(Pm))
    Qcal = np.zeros(len(Qm))
    
    for i in range(0,len(Bus_num)):
        for j in range(0,len(Bus_num)):
            Pcal[i] = Pcal[i] + Vm[i]*Vm[j]*(G[i,j]*np.cos(theta[i]-theta[j])+B[i,j]*np.sin(theta[i]-theta[j]))
            Qcal[i] = Qcal[i] + Vm[i]*Vm[j]*(G[i,j]*np.sin(theta[i]-theta[j])-B[i,j]*np.cos(theta[i]-theta[j]))
            
    dP = np.subtract(Pm[1:len(Bus_num)], Pcal[1:len(Bus_num)]) #Pm - Pcal
    dQ = np.subtract(Qm[1+Num_PV:len(Bus_num)], Qcal[1+Num_PV:len(Bus_num)]) #Qm - Qcal
    
    J1 = np.zeros((Num_PV+Num_PQ, Num_PV+Num_PQ))
    J2 = np.zeros((Num_PV+Num_PQ, Num_PQ))
    J3 = np.zeros((Num_PQ, Num_PV+Num_PQ))
    J4 = np.zeros((Num_PQ, Num_PQ))
    
    #J1
    for k in range(1,len(Bus_num)):
        for n in range(1,len(Bus_num)):
            if k==n:
                J1[k-1,n-1] = -Qcal[k]-B[k,k]*(np.abs(Vm[k])**2)
            else:
                J1[k-1,n-1] = np.abs(Vm[k])*np.abs(Vm[n])*(G[k,n]*np.sin(theta[k]-theta[n])-B[k,n])
    #J2
    for k in range(1,len(Bus_num)):
        for n in range(1+Num_PV,len(Bus_num)):
            if k==n:
                J2[k-1,n-1-Num_PV] = (Pcal[k])/(np.abs(Vm[k]))+G[k,k]*Vm[k]
            else:
                J2[k-1,n-1-Num_PV] = np.abs(Vm[k])*(G[k,n]*np.cos(theta[k]-theta[n])+B[k,n]*np.sin(theta[k]-theta[n]))
    #J3
    for k in range(1+Num_PV,len(Bus_num)):
        for n in range(1,len(Bus_num)):
            if k==n:
                J3[k-1-Num_PV,n-1] = Pcal[k]-G[k,k]*(np.abs(Vm[k])**2)
            else:
                J3[k-1-Num_PV,n-1] = -np.abs(Vm[k])*np.abs(Vm[n])*(G[k,n]*np.cos(theta[k]-theta[n])+B[k,n]*np.sin(theta[k]-theta[n]))
    #J4
    for k in range(1+Num_PV,len(Bus_num)):
        for n in range(1+Num_PV,len(Bus_num)):
            if k==n:
                J4[k-1-Num_PV,n-1-Num_PV] = (Qcal[k])/(np.abs(Vm[k]))-B[k,k]*np.abs(Vm[k])
            else:
                J4[k-1-Num_PV,n-1-Num_PV] = np.abs(Vm[k])*(G[k,n]*np.sin(theta[k]-theta[n])-B[k,n]*np.cos(theta[k]-theta[n]))
    
    J = np.vstack([np.hstack([J1,J2]),np.hstack([J3,J4])])
    
    #Jacobian matrix 역행렬구하기 / DelT, DelV 구해서 Update
    InvJ = np.linalg.inv(J)
    dPQ = np.transpose(np.hstack((dP,dQ)))
    delta = np.dot(InvJ,dPQ)
    dT = delta[0:len(Bus_num)-1]
    dV = delta[len(Bus_num)-1:len(delta)]
    
    for i in range(1,len(Bus_num)):
        theta[i] = theta[i] + dT[i-1]
        if Bus_type[i]==3:
            Vm[i] = Vm[i] + dV[i-1-Num_PV]
            
        err = np.max(np.abs(delta))
#%% Theta radian to degree변환, Slack bus에 값 채워주기
theta = theta/np.pi * 180

# Slack bus
PG[0] = PD[0] + Pcal[0]

for i in range(0,len(Bus_num)):
    if Bus_type[i]!=3 :
        QG[i] = Qcal[i] + QD[i]
#%% Bus data 정리(소숫점 넷째자리까지)
Bus_Num = B1[:,0].astype(int)
Bus_type = B1[:,1].astype(int)
PG = np.round(PG,4)
QG = np.round(QG,4)
PD = np.round(PD,4)
QD = np.round(QD,4)
VM = np.round(Vm,4)
Theta = np.round(theta,4)

#voltage profile plot
x = Bus_Num
y = VM
plt.rcParams["figure.figsize"]=(10,6)
plt.plot(x,y,linestyle='--', color='orange')
plt.xlabel('Bus Num')
plt.ylabel('Voltage Magnitude[p.u.]')
plt.hlines(1.039,1,len(Bus_Num)+1, color='r', linestyle='--', linewidth=2, alpha=1)
plt.hlines(0.912,1,len(Bus_Num)+1, color='r', linestyle='--', linewidth=2, alpha=1)
plt.title('Voltage Profile of IEEE 33 test feeder', fontsize = 20)
plt.show()


#Continuation Power Flow

lambbda = np.zeros(1)
solution = np.concatenate((Theta[1:len(Bus_Num)],VM[1:len(Bus_Num)],lambbda), axis = 0)

Ek = np.zeros((1,len(solution)-1), dtype = int)
ek = np.append(Ek,1)

K1 = PD[1:]
K2 = QD[1:]

MJ_new = np.vstack([np.column_stack([np.hstack([J1,J2]),K1]), np.column_stack([np.hstack([J3,J4]),K2]), np.transpose(ek)])            

    #### Initilization #### (sigma = step size)
new_lambbda = lambbda.copy()
Y_Vector = np.ones(2*Num_PQ + Num_PV + 1, dtype = float)
Max_iteration = 20
iteration = 0
A_V_L = solution
Kg = 0
sigma = 0.2
a = 2*Num_PQ + Num_PV + 1
xx = 0
ones = np.append(Ek,-1) 
q = []
ek_new = np.zeros((1,len(solution)), dtype = int)
ek_new[0,a] = 1

    #### Derivation of weakest bus. ####
    #아직 수정중#
Weak_bus = []
for i in range(1,len(Bus_num)):
    for k in range(1+Num_PV,len(Bus_num)):
        weak = np.zeros(len(Bus_num)-(1+Num_PV)+1)
        a = np.zeros(len(Bus_num)-(1+Num_PV)+1)
        b = a = np.zeros(len(Bus_num)-(1+Num_PV)+1)
        if i==k:
            a[k] = - ((Pcal[i])/(np.abs(Vm[i])) + G[i,i]*Vm[i])
        else:
            b[k] = b[k] - (Vm[i]*((G[i,k]*np.cos(theta[i]-theta[k])+B[i,k]*np.sin(theta[i]-theta[k]))))
        weak[k] = a[k] + b[k]
        min()
V_Show = []
Lambda = []
Thermal_Limit_V = []
Thermal_Limit_L = []
qq = []
V_Show.append(1)
Lambda.append(0)
    
#### Draw nose curve ####
while new_lambbda >= 0 :
#for p in range(100) :
    print("==============", "no.", xx, "==========================================")
    if iteration >= Max_iteration  :
        sigma = sigma/2
        A = solution[:len(Angle)-1]
        V = solution[len(VM)-1:len(delta)]
        A = np.insert(A,0,0)
        V = np.insert(V,0,1)
    else : 
        sigma = sigma * 1.1
        lambbda = new_lambbda.copy()  
        MJ = MJ_new.copy()
        A = Angle.copy()
        V = VM.copy()
               
