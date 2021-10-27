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
start, last, interval = 0.1, 0.1, 0.011
reps = (last-start)/interval
# r = 2*37 # product average number of genres of a movie with total # of genres
r = 2
final_U = np.zeros((len_rows, r))
final_V = np.zeros((r, len_cols))
# perturb = np.arange(start,last, interval) # uniform perturbation
perturb = [0]
permutation = []
for i in range(len_rows):
    for j in range(r):
        permutation.append((0,i,j))
for i in range(r):
    for j in range(len_cols):
        permutation.append((1,i,j))
# print(permutation)
np.random.shuffle(permutation)

U = np.ones((len_rows, r))
V = np.ones((r, len_cols))

def optimize_u(M, U, V, i, x):
    # optimize u_ix
    numer = 0
    v_x = np.copy(V[x])
    print("v_x")
    print(v_x)
    m_i = np.copy(M[i])
    # print(m_i)
    u_i = np.copy(U[i])
    nz_col = np.nonzero(m_i)[0]
    print("nz_col")
    print(nz_col)
    for col in nz_col:
        # print(col)
        for k in range(r):
            v_k = np.copy(V[k])
            if k == x:
                continue
            numer += v_x[col] * (m_i[col] - u_i[k]*v_k[col])
    # test = np.sum(np.square(v_x))
    # print("test = %f" %test)
    denom = np.sum(np.square(v_x[nz_col]))
    print("denom =%f" %denom)
    opt_u = numer/denom
    # print(opt_u)
    U[i][x] = opt_u
    print("opt_u")
    print(i, x, opt_u)
    # optimize_u()
    return opt_u

def optimize_v(M,U,V,x,j):
    # optimize v_xj
    numer = 0
    u_x = np.copy(U[:,x])
    m_j = np.copy(M[:,j])
    v_j = np.copy(V[:,j])
    nz_row = np.nonzero(m_j)[0]
    for row in nz_row:
        for k in range(r):
            u_k = np.copy(U[:,k])
            if k == x:
                continue
            numer += u_x[row] * (m_j[row] - u_k[row]*v_j[k])
    denom = np.sum(np.square(u_x[nz_row]))
    opt_v = numer/denom
    V[x][j] = opt_v
    print("opt_v")
    print(x,j,opt_v)
    return opt_v
    # optimize_v()
U[0][0] = optimize_u(M,U,V,0,0)
print("first")
print(U)
print(V)
V[0][0] = optimize_v(M,U,V,0,0)
print("second")
print(U)
print(V)
U[2][0] = optimize_u(M,U,V,2,0)
print("third")
print(U)
print(V)

# for p in perturb:
#     # initialize U and V with small perturbation    
#     U = np.ones((len_rows, r)) + p
#     V = np.ones((r, len_cols)) + p

#     # order the optimization of the elements of U and V
#     for perm in permutation:
#         if perm[0] == 0:
#             i = perm[1]
#             x = perm[2]
#             # optimize u_ix
#             numer = 0
#             v_x = np.copy(V[x])
#             m_i = np.copy(M[i])
#             u_i = np.copy(U[i])
#             nz_col = np.nonzero(m_i)[0]
#             # print(nz_col)
#             for col in nz_col:
#                 # print(col)
#                 for k in range(r):
#                     v_k = np.copy(V[k])
#                     if k == x:
#                         continue
#                     numer += v_x[col] * (m_i[col] - u_i[k]*v_k[col])
#             test = np.sum(np.square(v_x))
#             # print("test = %f" %test)
#             denom = np.sum(np.square(v_x))
#             # print("denom =%f" %denom)
#             opt_u = numer/denom
#             # print(opt_u)
#             U[i][x] = opt_u
#             # print(i, x, opt_u)
#             # optimize_u()
#         else:
#             x = perm[1]
#             j = perm[2]
#             # optimize v_xj
#             numer = 0
#             u_x = np.copy(U[:,x])
#             m_j = np.copy(M[:,j])
#             v_j = np.copy(V[:,j])
#             nz_row = np.nonzero(m_j)[0]
#             for row in nz_row:
#                 for k in range(r):
#                     u_k = np.copy(U[:,k])
#                     if k == x:
#                         continue
#                     numer += u_x[row] * (m_j[row] - u_k[row]*v_j[k])
#             denom = np.sum(np.square(u_x))
#             opt_v = numer/denom
#             V[x][j] = opt_v
#             # optimize_v()
#     final_U += U
#     final_V += V

# final_U = final_U/reps
# final_V = final_V/reps
# final_UV = np.dot(final_U, final_V)

    # random_row = np.random.permutation(len_rows-1)
    # random_2r = np.random.permutation(2*r-1)
    # random_col = np.random.permutation(len_cols-1)

    # ##TODO: consider nonzero values for Mij
    # # end the attempt at optimization
    # for x in random_2r:
    #     if x < r:
    #         for i in random_row:
    #             # optimize u_ix
    #             numer = 0
    #             v_x = np.copy(V[x])
    #             m_i = np.copy(M[i])
    #             u_i = np.copy(U[i])
    #             nz_col = np.nonzero(m_i)[0]
    #             # print(nz_col)
    #             for col in nz_col:
    #                 # print(col)
    #                 for k in range(r):
    #                     v_k = np.copy(V[k])
    #                     if k == x:
    #                         continue
    #                     numer += v_x[col] * (m_i[col] - u_i[k]*v_k[col])
    #             test = np.sum(np.square(v_x))
    #             # print("test = %f" %test)
    #             denom = np.sum(np.square(v_x))
    #             # print("denom =%f" %denom)
    #             opt_u = numer/denom
    #             # print(opt_u)
    #             U[i][x] = opt_u
    #             # print(i, x, opt_u)
    #             # optimize_u()
    #     else:
    #         x -= r
    #         for j in random_col:
    #             # optimize v_xj
    #             numer = 0
    #             u_x = np.copy(U[:,x])
    #             m_j = np.copy(M[:,j])
    #             v_j = np.copy(V[:,j])
    #             nz_row = np.nonzero(m_j)[0]
    #             for row in nz_row:
    #                 for k in range(r):
    #                     u_k = np.copy(U[:,k])
    #                     if k == x:
    #                         continue
    #                     numer += u_x[row] * (m_j[row] - u_k[row]*v_j[k])
    #             denom = np.sum(np.square(u_x))
    #             opt_v = numer/denom
    #             V[x][j] = opt_v
    #             # optimize_v()
    # final_U += U
    # final_V += V