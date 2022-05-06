"""
Created on Thu Oct 10 17:23:45 2019

@author: JunHyeok.Kim
"""
import numpy as np

Bus = np.genfromtxt('Bus_33.txt', comments='%')
data = np.genfromtxt('line_33.txt',comments='%')


#Line Data
#Y_Matrix가 python은 0부터 시작하기 때문에 1을 빼주고 시작함
From = np.subtract(data[0:len(data),0].astype(int),1)
To = np.subtract(data[0:len(data),1].astype(int),1)
#From = data[0:len(data),0].astype(int)
#To = data[0:len(data),0].astype(int)
R = data[0:len(data),2]
X = 1j*data[0:len(data),3]
C = 1j/2*data[0:len(data),4]

#Bus Data
Bus_Num = []
Bus_Type = []
PG = []
QG = []
PL = []
QL = []
VM = []
VAngle = []

Bus_Num = Bus[0:len(Bus),0].astype(int).copy()
Bus_Type = Bus[0:len(Bus),1].astype(int).copy()
PG = Bus[0:len(Bus),2].copy()
QG = Bus[0:len(Bus),3].copy()
PL = Bus[0:len(Bus),4].copy()
QL = Bus[0:len(Bus),5].copy()
VM = Bus[0:len(Bus),6].copy()
VAngle= Bus[0:len(Bus),7].copy()

#Admittance Matrix
Y = np.zeros((len(Bus_Num),len(Bus_Num)),dtype=complex)
Z = np.add(R,X)
invZ = np.divide(1,Z)

#Y_Matrix가 python은 0부터 시작하기 때문에 1을 빼주고 시작함
for i in range(0,len(data)):
    Y[From[i],From[i]] = Y[From[i],From[i]]+invZ[i]+C[i]
    Y[To[i],To[i]] = Y[To[i],To[i]]+invZ[i]+C[i]
    Y[From[i],To[i]] = -invZ[i]
    Y[To[i],From[i]] = -invZ[i]
  
Num_PV = 0
Num_PQ = 0

for j in range(0,len(Bus_Type)):
    if Bus_Type[j] == 2:
        Num_PV = Num_PV+1
    elif Bus_Type[j] ==3:
        Num_PQ = Num_PQ+1

Pm = np.subtract(PG,PL)
Qm = np.subtract(QG,QL)

G = np.real(Y)
B = np.imag(Y)

#iteration for calculating Power Flow

Error = 1
n = 0

while Error > 0.000001 and n<10:
    
    n = n+1

    Pcal = np.zeros(len(Pm))
    Qcal = np.zeros(len(Qm))

    #Calculation of P,Q refer to Voltage and Degree
    for a in range(0,len(Bus_Num)):
        for b in range(0,len(Bus_Num)):
            Pcal[a] = Pcal[a] + VM[a]*VM[b]*(G[a,b]*np.cos(VAngle[a]-VAngle[b])+B[a,b]*np.sin(VAngle[a]-VAngle[b]))
            Qcal[a] = Qcal[a] + VM[a]*VM[b]*(G[a,b]*np.sin(VAngle[a]-VAngle[b])-B[a,b]*np.cos(VAngle[a]-VAngle[b]))

    delP = np.subtract(Pm[1:len(Bus_Num)],Pcal[1:len(Bus_Num)])
    delQ = np.subtract(Qm[1+Num_PV:len(Bus_Num)],Qcal[1+Num_PV:len(Bus_Num)])    

    #Jacobian Matrix

    Jpth = np.zeros((Num_PV+Num_PQ, Num_PV+Num_PQ))
    Jpv = np.zeros((Num_PV+Num_PQ, Num_PQ))
    Jqth = np.zeros((Num_PQ, Num_PV+Num_PQ))
    Jqv = np.zeros((Num_PQ, Num_PQ))

    for a in range(1,len(Bus_Num)):
        for b in range(1,len(Bus_Num)):
            if a==b:
                Jpth[a-1,b-1] = -Qcal[a]-B[a,a]*np.power(np.abs(VM[a]),2)
            else:
                Jpth[a-1,b-1] = np.abs(VM[a])*np.abs(VM[b])*(G[a,b]*np.sin(VAngle[a]-VAngle[b])-B[a,b]*np.cos(VAngle[a]-VAngle[b]))

    for a in range(1,len(Bus_Num)):
        for b in range(1+Num_PV,len(Bus_Num)):
            if a==b: 
                Jpv[a-1,b-1-Num_PV] = Pcal[a]/np.abs(VM[a]) + G[a,a]*np.abs(VM[a]) 
            else:
                Jpv[a-1,b-1-Num_PV] =np.abs(VM[a])*(G[a,b]*np.cos(VAngle[a]-VAngle[b])+B[a,b]*np.sin(VAngle[a]-VAngle[b]))
                
    for a in range(1+Num_PV,len(Bus_Num)):
        for b in range(1,len(Bus_Num)) : 
            if a==b:
                Jqth[a-1-Num_PV,b-1] = Pcal[a] - G[a,a,]*np.power(np.abs(VM[a]),2)
            else:
                Jqth[a-1-Num_PV,b-1] = -np.abs(VM[a])*np.abs(VM[b])*(G[a,b]*np.cos(VAngle[a]-VAngle[b])+B[a,b]*np.sin(VAngle[a]-VAngle[b]))

    for a in range(1+Num_PV,len(Bus_Num)):
        for b in range(1+Num_PV, len(Bus_Num)):

            if a==b:
                Jqv[a-1-Num_PV,b-1-Num_PV] = Qcal[a]/np.abs(VM[a]) - B[a,a]*np.abs(VM[a])
            else:
                Jqv[a-1-Num_PV,b-1-Num_PV] = np.abs(VM[a])*(G[a,b]*np.sin(VAngle[a]-VAngle[b])-B[a,b]*np.cos(VAngle[a]-VAngle[b]))             
    J = np.vstack([np.hstack([Jpth,Jpv]),np.hstack([Jqth,Jqv])])                   
    # Calculate delA, DelV and Update

    InvJ= np.linalg.inv(J)
    
    InvJthp = np.zeros((Num_PV+Num_PQ, Num_PV+Num_PQ))
    InvJthq = np.zeros((Num_PV+Num_PQ, Num_PQ))
    InvJvp = np.zeros((Num_PQ,Num_PV+Num_PQ))
    InvJvq = np.zeros((Num_PQ, Num_PQ))
    
    InvJthp = InvJ[0:Num_PV+Num_PQ-1, 0:Num_PV+Num_PQ-1]
    InvJthq = InvJ[0:Num_PV+Num_PQ-1, Num_PV+Num_PQ:len(data)-1]
    InvJvp = InvJ[Num_PV+Num_PQ:len(data)-1, 0:Num_PV+Num_PQ-1]
    InvJvq = InvJ[Num_PV+Num_PQ:len(data)-1, Num_PV+Num_PQ:len(data)-1]
    
    Jqth_Inv = np.zeros((Num_PQ+Num_PQ,Num_PV+Num_PQ)) 
    JR = np.zeros((Num_PQ,Num_PV+Num_PQ))
    Jqth_Inv = np.linalg.inv(Jqth)
    JR = [Jpv - np.dot(Jpth.copy(),Jqth.copy(),Jqv.copy())]
    VPsensitivity = np.linalg.inv(JR)
    
    delPQ = np.transpose(np.hstack((delP,delQ)))
    delta = np.dot(InvJ,delPQ)
    delA = delta[0:len(Bus_Num)-1]
    delV = delta[len(Bus_Num)-1:len(delta)]
    
    for a in range (1,len(Bus_Num)):
        VAngle[a] = VAngle[a] + delA[a-1]
        if Bus_Type[a] == 3 :
            VM[a] = VM[a] + delV[a-1-Num_PV]

    Error = np.max(np.abs(delta))

# Apply the P,Q,V,Angle data
Pcal = np.zeros(len(Pm))
Qcal = np.zeros(len(Qm))

#Calculation of P,Q refer to Voltage and Degree
for a in range(0,len(Bus_Num)):
    for b in range(0,len(Bus_Num)):
        Pcal[a] = Pcal[a].copy() + VM[a]*VM[b]*(G[a,b]*np.cos(VAngle[a]-VAngle[b])+B[a,b]*np.sin(VAngle[a]-VAngle[b])).copy()
        Qcal[a] = Qcal[a].copy() + VM[a]*VM[b]*(G[a,b]*np.sin(VAngle[a]-VAngle[b])-B[a,b]*np.cos(VAngle[a]-VAngle[b])).copy()    
        
PG[0] = PL[0] + Pcal[0]
VAngle1 = VAngle.copy()/np.pi * 180
for a in range(0, len(Bus_Num)):
    if Bus_Type[a] == 1 :
        QG[a] = Qcal[a] + QL[a]
    elif Bus_Type[a] ==2 :
        QG[a] = Qcal[a] + QL[a]

#Bus Data
Bus[0:len(Bus),2] = np.round(PG,4)
Bus[0:len(Bus),3] = np.round(QG,4)
Bus[0:len(Bus),4] = np.round(PL,4)
Bus[0:len(Bus),5] = np.round(QL,4)
Bus[0:len(Bus),6] = np.round(VM,4)
Bus[0:len(Bus),7] = np.round(VAngle,4)

Result = open("PF_Result_33Bus.txt",'w')
Result.write('% Num\t  Type\t   PG\t   QG\t   PL\t   QL\t   VM\t   VAngle\n')
for c in range(0, len(Bus_Num)):
    for d in range(0, 8):
        Result.write(str(Bus[c,d]))
        Result.write("   ")
    Result.write("\n")
Result.close()
