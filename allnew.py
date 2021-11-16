import sys
import numpy as np

file = sys.argv[1]
f = open(file, 'r')
lines = f.readlines()
test_file = sys.argv[2]
tf = open(test_file, 'r')
lines_tf = tf.readlines()
# target_uid = 600
# target_avg = 0

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
    # if irow == target_uid:
    #     target_avg = avg_stars
    # normalize M
    # by subracting average of average stars of item and user
    u_stars[nonzero_index] -= (avg_stars + avg_items[nonzero_index])/2
    M[irow] = u_stars


# def rmse(M,UdotV):
#     nz_M = np.where(M != 0)
#     return np.sqrt(np.sum(np.square((M-UdotV)[nz_M])))/np.sqrt(len(M[nz_M]))

# Compute rmse of a and b
def rmse(a,b):
    if not np.any(a-b):
        return 0
    return np.sqrt(np.mean(np.square(np.array(a)-np.array(b))))

# Compute square sum of a-b (used in optimizing)
def error(a,b):
    return np.sum(np.square(a-b))

def optimum_u(M,U,V,i,x):
    numer = 0
    v_x = np.copy(V[x])
    m_i = np.copy(M[i])
    u_i = np.copy(U[i])
    nz_col = np.where(m_i != 0)[0]
    if len(nz_col) <= 0:
        return 0
    for col in nz_col:
        for k in range(r):
            v_k = np.copy(V[k])
            if k == x:
                continue
            numer += v_x[col] * (m_i[col] - u_i[k]*v_k[col])
    denom = np.sum(np.square(v_x[nz_col]))
    # print(denom)
    opt_u = numer/denom
    U[i][x] = opt_u
    return opt_u
def optimum_v(M,U,V,x,j):
    numer = 0
    u_x = np.copy(U[:,x])
    m_j = np.copy(M[:,j])
    v_j = np.copy(V[:,j])
    nz_row = np.where(m_j != 0)[0]
    if len(nz_row) <= 0:
        return 0
    for row in nz_row:
        for k in range(r):
            u_k = np.copy(U[:,k])
            if k == x:
                continue
            numer += u_x[row] * (m_j[row] - u_k[row]*v_j[k])
    denom = np.sum(np.square(u_x[nz_row]))
    opt_v = numer/denom
    V[x][j] = opt_v

    return opt_v

def optimize_u(M, U, V, i, j):
    base = M[i,:]
    target = U[i,:]
    nonzero_index = np.where(base != 0)[0]
    start = -0.1
    d = 0.01
    target[j] += start
    err = error(base[nonzero_index], np.dot(target,V)[nonzero_index])
    for x in range(3):
        for _ in range(20):
            target[j] += d
            tmp = error(base[nonzero_index], np.dot(target,V)[nonzero_index])
            if tmp > err:
                target[j] -= d
                break
            err = tmp
        d *= 0.1
        start *= 0.1
        target[j] += start

def optimize_v(M, U, V, i, j):
    base = M[:,j]
    target = V[:,j]
    nonzero_index = np.where(base != 0)[0]
    start = -0.1
    d = 0.01
    target[i] += start
    err = error(base[nonzero_index], np.dot(U,target)[nonzero_index])
    for x in range(3):
        for _ in range(20):
            target[i] += d
            tmp = error(base[nonzero_index], np.dot(U,target)[nonzero_index])
            if tmp > err:
                target[i] -= d
                break
            err = tmp
        d *= 0.1
        start *= 0.1
        target[i] += start
# UV decomposition
r = 2
final_U = np.zeros((len_rows, r))
final_V = np.zeros((r, len_cols))
permutation = []
for i in range(len_rows):
    for j in range(r):
        permutation.append((0,i,j))
        # print(permutation)
for i in range(r):
    for j in range(len_cols):
        permutation.append((1,i,j))
np.random.shuffle(permutation)

U = np.zeros((len_rows, r)) + 0.0001
V = np.zeros((r, len_cols)) + 0.0001

standard = len(permutation)
for perm in permutation:
    standard -= 1
    print(standard)
    if perm[0] == 0: # optimize u_ix
        i = perm[1]
        x = perm[2]
        ret = optimize_u(M,U,V,i,x)
    else: # optimize v_xj
        x = perm[1]
        j = perm[2]
        ret = optimize_v(M,U,V,x,j)

final_UV = np.dot(U, V)
r_rmse = rmse(M, final_UV)
print("r = %d" %r)
print("final rmse = %f" %r_rmse)

result_UV = np.zeros((len_rows, len_cols))
for row in range(len_rows):
    for col in range(len_cols):
        result_UV[row][col] = final_UV[row][col] + (avg_users[row] + avg_items[col])/2

# Part 2. collaborative filtering
#         with reslut of UV decomposition
new_M = np.copy(result_UV)

avg_users = np.zeros(len_rows) # store average stars of each user
avg_items = np.zeros(len_cols) # store average stars of each movie

# get average stars of movies
for icol in range(len_cols):
    m_stars = np.copy(new_M[:,icol])
    # if np.count_nonzero(m_stars) <= 0:
    #     continue
    # nonzero_index = np.nonzero(m_stars)
    avg_stars = m_stars.mean()
    avg_items[icol] = avg_stars

# get average stars of users
for irow in range(len_rows):
    u_stars = np.copy(new_M[irow])
    # if np.count_nonzero(u_stars) <= 0:
    #     continue
    # nonzero_index = np.nonzero(u_stars)
    avg_stars = u_stars.mean()
    avg_users[irow] = avg_stars
    # if irow == target_uid:
    #     target_avg = avg_stars
    # normalize M
    # by subracting average of average stars of item and user
    u_stars -= (avg_stars + avg_items)/2
    new_M[irow] = u_stars

mid_M = np.zeros((len_rows, len_cols))
for row in range(len_rows):
    for col in range(len_cols):
        mid_M[row][col] = new_M[row][col] + (avg_users[row] + avg_items[col])/2
# print(final_UV)
# print(new_M)

def cosine_distance(a, b):
    mag_a = np.linalg.norm(a)
    mag_b = np.linalg.norm(b)
    # ret = np.dot(a, b)/(mag_a*mag_b)
    ret = np.matmul(a, np.transpose(b))/(mag_a*mag_b)
    return ret

# find similar 100 users and take average ratings of them
# =====================================
# user-based collaborative filtering
# =====================================
cnt = 0
gg = open("output_blabla.txt", 'w')
standard = len(pred_info)
for info in pred_info:
    print(standard)
    standard -= 1
    target_uid = int(info[0])
    target_mid = int(info[1])
    # print("target_uid = %d" %target_uid)
    # print("target_mid = %d" %target_mid)
    
    # find 100 most similar users
    distances = np.zeros(len_rows)
    target_stars = M[target_uid]
    for irow in range(len_rows):
        u_stars = M[irow]
        if np.count_nonzero(u_stars) > 0:
            cos_dis = cosine_distance(target_stars, u_stars)
            distances[irow] = cos_dis
    sim_uids = np.argpartition(distances, -20)[-20:]
    # print(sim_uids)

    # first: make prediction with ratings of similar users
    m_stars = np.zeros((len(sim_uids)+1, 1))
    for idx in range(len(sim_uids)):
        uid = sim_uids[idx]
        avg = (avg_users[uid] + avg_items[target_mid])/2
        m_stars[idx] = new_M[uid][target_mid] + avg
    target_avg = (avg_users[target_uid] + avg_items[target_mid])/2
    m_stars[-1] = target_avg
    pred_stars = m_stars[np.nonzero(m_stars)].mean()

    # # second: remove effect of element with big difference from average
    # m_stars = np.zeros((len(sim_uids)+1, 1))
    # for idx in range(len(sim_uids)):
    #     uid = sim_uids[idx]
    #     avg = (avg_users[uid] + avg_items[target_mid])/2
    #     sim_avg = new_M[uid][target_mid] + avg
    #     if abs(sim_avg - pred_stars) >= 3:
    #         cnt += 1
    #         continue
    #     m_stars[idx] = sim_avg
    # pred_stars = m_stars[np.nonzero(m_stars)].mean()

    # write output
    info[2] = str(pred_stars)
    gg.write(",".join(info))
print("cnt = %d" %cnt)




# # compute average ratings of I for similar users
# # v1. make prediction with normalized values
# # v2. make prediction with original values

# stars = np.zeros(len_cols)
# for icol in range(len_cols):
#     m_stars = np.copy(new_M[sim_uids,icol]) # v1.
#     # m_stars = np.copy(original_M[sim_uids,icol]) # v2.
#     if np.count_nonzero(m_stars) < 10:
#         continue
#     avg_stars = m_stars[np.nonzero(m_stars)].mean()
#     target_avg = (avg_users[target_uid] + avg_items[icol])/2
#     stars[icol] = avg_stars + target_avg # v1.
#     # stars[icol] = avg_stars # v2.
# rcmd_mids = np.argpartition(stars, -100)[-100:]
# print(stars[rcmd_mids])

# # recommend 10 movies
# user_base = []
# recommend_star = []
# for mid in rcmd_mids:
#     user_base.append((mid, stars[mid]))
#     recommend_star.append(stars[mid])
# final_sum = 0
# for s in recommend_star:
#     final_sum += s
# print("final_star = %f" %(float(final_sum)/float(len(recommend_star))))
# user_base.sort(key=lambda x: (-x[1],x[0]))
# for i in range(50):
#     ele = user_base[i]
#     print("%d\t%f" %(ele[0], ele[1]))

# # find similar 100 items and take average ratings of them