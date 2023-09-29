import sys
import pandas as pd
import time, numpy as np
import pyomo.environ as pyo
from pyomo.environ import *
import math 
import random
import pandas as pd
import networkx as nx

n = 7 #• nodes to visit
N = n+1 # Problem size (depot + nodes to visit = 1+ n)

v = 1.00 # Speed
    #matrix of distances
dist = np.array([[100000, 5.5, 4.2, 2.6, 2.4, 1.3, 2.5, 4.3], #Node 0 depot
                  [4.7, 100000, 3.7, 2.1, 5.1, 6, 7.2, 9],
                  [4.2, 4.5, 100000, 1.6, 3.2, 5.5, 6.7, 8.5],
                  [2.6, 2.9, 1.6, 100000, 3, 3.9, 5.1, 6.9],
                  [3.8, 4.1, 2.8, 1.2, 100000, 5.1, 6.3, 8.1],
                  [3.9, 7.4, 6.1, 4.5, 3.3, 100000, 1.2, 3],
                  [3.5, 7, 5.7, 4.1, 2.9, 1.2, 100000, 2.3],
                  [5.8, 9.3, 8, 6.4, 5.2, 3, 2.3, 100000]]) # Node 7
                                                            # travel time t_ij where i=j are set with a big amount exclude them from solution


t = np.zeros((N,N)) #matrix of travel time t_ij

t = (dist/v) # Compute travel time t_ij

range_n = range(1,N)
        
LB = np.array([0, 0, 0, 0, 0, 0, 0, 0])
UB = np.array([1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000])

#Create Model
model = pyo.ConcreteModel()

#Define variables
model.T = pyo.Var(range(0,N+1), # index i
                  bounds = (0,None),
                  initialize=0)

model.y = pyo.Var(range(0,N), # index i
                  range(0,N), # index j
                  within = Binary,
                  initialize=0)

T = model.T
y = model.y

# Constraints
model.C1 = pyo.ConstraintList() 
for i in range_n:
    model.C1.add(expr = T[i] - y[0,i]*t[0][i] >= 0)

model.C2 = pyo.ConstraintList() 
for i in range(1,N):
    for j in range(1,N):
        if i!= j:
            M = UB[i]-LB[j]+t[i][j]
            model.C2.add(expr = T[i] - T[j] +t[i][j]-(1-y[i,j])*M <= 0)
            #model.C2.add(expr = T[i] - T[j] +y[i,j]*M <= UB[i]-LB[j])

model.C3 = pyo.ConstraintList() 
for j in range(1,N):
    model.C3.add(expr = sum(y[i,j] for i in range(0,N) if i!=j)  == 1)

model.C4 = pyo.ConstraintList() 
for i in range(1,N):
    model.C4.add(expr = sum(y[i,j] for j in range(0,N) if i!=j)  == 1)

model.C5 = pyo.ConstraintList() 
for i in range_n:
    model.C5.add(expr = T[i] + t[i][0] <= T[n+1])

model.C6 = pyo.ConstraintList() 
for i in range_n:
    model.C6.add(expr = T[i] >= LB[i])

model.C7 = pyo.ConstraintList() 
for i in range_n:
    model.C7.add(expr = T[i] <= UB[i])

model.C8 = pyo.Constraint(expr= sum(y[i,0] for i in range(1,N)) == 1)

model.C9 = pyo.Constraint(expr= sum(y[0,j] for j in range(1,N)) == 1)

model.C10 = pyo.ConstraintList() 
for i in range(1,N):
    model.C10.add(expr = T[i] - T[N]  <= -t[i][0])

model.C11 = pyo.Constraint(expr= T[0] == 0)
    
# Define Objective Function
model.obj = pyo.Objective(expr = T[n+1], 
                          sense = minimize)

begin = time.time()
opt = SolverFactory('cplex')
results = opt.solve(model)

deltaT = time.time() - begin # Compute Exection Duration

model.pprint()

sys.stdout = open("Time_Constrained_Asymmetric_Traveling_Salesman_Problem_Results.txt", "w") #Print Results on a .txt file

print('Time =', np.round(deltaT,2))

if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):

    print('Makespan (Obj value) =', pyo.value(model.obj))
    print('Solver Status is =', results.solver.status)
    print('Termination Condition is =', results.solver.termination_condition)
    print(" " )
    for i in range(0,N+1):
         if i == 0:
             print(" Tour from Node " , i , " begins at time " , pyo.value(T[i]),' (t[',i,'])')
         else:
             print('---> Node',i, 'is visited at time', round(pyo.value(T[i]),2) , ' (t[',i,'])')
    print(" " )
    for i in range(0,N):
        for j in range(0,N):
            if(i!=j and round(pyo.value(y[i,j]),2) != 0):
                print('y[',i, '][', j , ']=',round(pyo.value(y[i,j]),2))
elif (results.solver.termination_condition == TerminationCondition.infeasible):
   print('Model is unfeasible')
  #print('Solver Status is =', results.solver.status)
   print('Termination Condition is =', results.solver.termination_condition)
else:
    # Something else is wrong
    print ('Solver Status: ',  results.solver.status)
    print('Termination Condition is =', results.solver.termination_condition)
    
sys.stdout.close()