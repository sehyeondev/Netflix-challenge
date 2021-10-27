import sys
import numpy as np

file = sys.argv[1]
f = open(file, 'r')
lines = f.readlines()
test_file = sys.argv[2]
tf = open(test_file, 'r')
lines_tf = tf.readlines()
target_uid = 600
target_avg = 0

uid_set = set()
mid_set = set()
ums = []
pred_info = [] # store information to be predicted

# data preprocessing
for line in lines:
    info = line.split(',')
    uid, mid, star = (int(info[0]), int(info[1]), float(info[2]))
    uid_set.add(uid)
    mid_set.add(mid)
    ums.append((uid, mid, star))
for line in lines_tf:
    info = line.split(',')
    pred_info.append(info)
    uid, mid = (int(info[0]), int(info[1]))
    uid_set.add(uid)
    mid_set.add(mid)

# make utility matrix M
len_rows = max(uid_set)+1
len_cols = max(mid_set)+1
M = np.zeros((len_rows, len_cols))
for info in ums:
    uid, mid, star = info
    M[uid, mid] = star
original_M = np.copy(M)

avg_users = np.zeros(len_rows) # store average stars of each user
avg_items = np.zeros(len_cols) # store average stars of each movie

# get average stars of movies
for icol in range(len_cols):
    m_stars = np.copy(M[:,icol])
    if np.count_nonzero(m_stars) <= 0:
        continue
    nonzero_index = np.nonzero(m_stars)
    avg_stars = m_stars[nonzero_index].mean()
    avg_items[icol] = avg_stars

# get average stars of users
for irow in range(len_rows):
    u_stars = np.copy(M[irow])
    if np.count_nonzero(u_stars) <= 0:
        continue
    nonzero_index = np.nonzero(u_stars)
    avg_stars = u_stars[nonzero_index].mean()
    avg_users[irow] = avg_stars
    if irow == target_uid:
        target_avg = avg_stars
    # normalize M
    # by subracting average of average stars of item and user
    u_stars[nonzero_index] -= (avg_stars + avg_items[nonzero_index])/2
    M[irow] = u_stars

# UV decomposition
# repeat UV decomposition and take the average of the results
start, last, interval = 1,2,1
reps = (last-start)/interval
r = 37 # total number of genres
final_U = np.zeros((len_rows, r))
final_V = np.zeros((r, len_cols))
perturb = np.arange(start,last, interval) # uniform perturbation

# order the optimization of the elements of U and V
permutation = []
for i in range(len_rows):
    for j in range(r):
        permutation.append((0,i,j))
for i in range(r):
    for j in range(len_cols):
        permutation.append((1,i,j))
np.random.shuffle(permutation)

# perform the optimization
for p in perturb:
    print("im doing")
    # initialize U and V with small perturbation
    U = np.zeros((len_rows, r)) + p
    V = np.zeros((r, len_cols)) + p
    for perm in permutation:
        # print("im working")
        if perm[0] == 0: # optimize u_ix
            i = perm[1]
            x = perm[2]
            print (i)
            numer = 0
            v_x = np.copy(V[x])
            m_i = np.copy(M[i])
            u_i = np.copy(U[i])
            nz_col = np.where(m_i != 0)[0]
            if len(nz_col) <= 0:
                continue
            for col in nz_col:
                for k in range(r):
                    v_k = np.copy(V[k])
                    if k == x:
                        continue
                    numer += v_x[col] * (m_i[col] - u_i[k]*v_k[col])
            denom = np.sum(np.square(v_x[nz_col]))
            opt_u = numer/denom
            U[i][x] = opt_u
        else: # optimize v_xj
            x = perm[1]
            j = perm[2]
            numer = 0
            u_x = np.copy(U[:,x])
            m_j = np.copy(M[:,j])
            v_j = np.copy(V[:,j])
            nz_row = np.where(m_j != 0)[0]
            if len(nz_row) <= 0:
                continue
            for row in nz_row:
                for k in range(r):
                    u_k = np.copy(U[:,k])
                    if k == x:
                        continue
                    numer += u_x[row] * (m_j[row] - u_k[row]*v_j[k])
            denom = np.sum(np.square(u_x[nz_row]))
            opt_v = numer/denom
            V[x][j] = opt_v
    final_U += U
    final_V += V

# to avoid overfitting
final_U = final_U/reps
final_V = final_V/reps
final_UV = np.dot(final_U, final_V)

# write result with predicted star
g = open("output.txt", 'w')
for info in pred_info:
    uid = int(info[0])
    mid = int(info[1])
    add = (avg_users[uid] + avg_items[mid])/2
    pred_star = final_UV[uid][mid] + add
    info[2] = str(pred_star)
    g.write(",".join(info))

## test result
result = np.zeros((len_rows, len_cols))
for row in range(len_rows):
    for col in range(len_cols):
        result[row][col] = final_UV[row][col] + (avg_users[uid] + avg_items[mid])/2
# print(final_U)
# print(final_V)
# print(final_UV)
# print(M)
# print(result)
# print(original_M)