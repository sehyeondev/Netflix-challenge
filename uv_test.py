import numpy as np

M = np.array([[5,2,4,4,3],
              [3,1,2,4,1],
              [2,0,3,1,4],
              [2,5,4,3,5],
              [4,4,5,4,0]])
len_cols = 5
len_rows = 5
# UV decomposition
# repeat UV decomposition and take the average of the results

def rmse(M,UdotV):
    nz_M = np.where(M != 0)
    return np.sqrt(np.sum(np.square((M-UdotV)[nz_M])))/np.sqrt(len(M[nz_M]))

r = 2
U = np.zeros((len_rows, r)) + 1
V = np.zeros((len_cols, r)) + 1

lrate = 0.001
regularizer = 0.1
for x in range(1000):
    for i in range(len_rows):
        M_i = np.copy(M[i])
        for j in np.nonzero(M_i)[0]:
            # print j
            # print j
            # print(U[i])
            # print(V[j])
            err = M[i,j] - np.dot(U[i], V[j])
            U[i] = U[i] + lrate*(err*V[j] - regularizer*U[i])
            V[j] = V[j] + lrate*(err*U[i] - regularizer*V[j])
print (rmse(M, np.dot(U,V.T)))
print(np.dot(U, V.T))