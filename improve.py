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

class Star:
    def __init__(self, uid, mid, star):
        self.uid = uid
        self.mid = mid
        self.star = star

star_arr = []
# data preprocessing
for line in lines:
    info = line.split(',')
    uid, mid, star = (int(info[0]), int(info[1]), float(info[2]))
    uid_set.add(uid)
    mid_set.add(mid)
    ums.append((uid, mid, star))
    star_arr.append(Star(uid,mid,star))

for line in lines_tf:
    info = line.split(',')
    pred_info.append(info)
    uid, mid = (int(info[0]), int(info[1]))
    uid_set.add(uid)
    mid_set.add(mid)

# get average of all ratings
sum_star = 0
num = 0
for star in star_arr:
    sum_star += star.star
    num += 1
average_all = float(sum_star)/float(num)



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

def rmse(M,UdotV):
    nz_M = np.where(M != 0)
    return np.sqrt(np.sum(np.square((M-UdotV)[nz_M])))/np.sqrt(len(M[nz_M]))

r = 30
U = np.zeros((len_rows, r)) + 3
V = np.zeros((len_cols, r)) + 3

lrate = 0.001
regularizer = 0.1
old_rmse = 1000000
threshold = 0.001
for x in range(1000):
    for i in range(len_rows):
        M_i = np.copy(original_M[i])
        for j in np.nonzero(M_i)[0]:
            # print j
            # print j
            # print(U[i])
            # print(V[j])
            err = original_M[i,j] - np.dot(U[i], V[j])
            U[i] = U[i] + lrate*(err*V[j] - regularizer*U[i])
            V[j] = V[j] + lrate*(err*U[i] - regularizer*V[j])
    if x > 100:
        new_rmse = rmse(original_M, np.dot(U,V.T))
        print (new_rmse)
        if (old_rmse - new_rmse) < threshold:
            break
        old_rmse = new_rmse
# print (rmse(M, np.dot(U,V.T)))
# print(np.dot(U, V.T))
final_UV = np.dot(U, V.T)
gg = open("output_testfinal.txt", 'w')
for info in pred_info:
    uid = int(info[0])
    mid = int(info[1])
    # add = (avg_users[uid] + avg_items[mid])/2
    pred_star = final_UV[uid][mid]
    # if uid == 551:
    #     print(final_UV[uid][mid])
    #     print(add)
    info[2] = str(pred_star)
    gg.write(",".join(info))


# def optimum_u(M,U,V,i,x):
#     numer = 0
#     v_x = np.copy(V[x])
#     m_i = np.copy(M[i])
#     u_i = np.copy(U[i])
#     nz_col = np.where(m_i != 0)[0]
#     if len(nz_col) <= 0:
#         return 0
#     for col in nz_col:
#         for k in range(r):
#             v_k = np.copy(V[k])
#             if k == x:
#                 continue
#             numer += v_x[col] * (m_i[col] - u_i[k]*v_k[col])
#     denom = np.sum(np.square(v_x[nz_col]))
#     # print(denom)
#     opt_u = numer/denom
#     U[i][x] = opt_u

#     return opt_u
# def optimum_v(M,U,V,x,j):
#     numer = 0
#     u_x = np.copy(U[:,x])
#     m_j = np.copy(M[:,j])
#     v_j = np.copy(V[:,j])
#     nz_row = np.where(m_j != 0)[0]
#     if len(nz_row) <= 0:
#         return 0
#     for row in nz_row:
#         for k in range(r):
#             u_k = np.copy(U[:,k])
#             if k == x:
#                 continue
#             numer += u_x[row] * (m_j[row] - u_k[row]*v_j[k])
#     denom = np.sum(np.square(u_x[nz_row]))
#     opt_v = numer/denom
#     V[x][j] = opt_v

#     return opt_v



# # UV decomposition
# # repeat UV decomposition and take the average of the results
# # start, last, interval = -0.025, 0.025, 0.0051
# # reps = (last-start)/interval
# # r = 37 # total number of genres
# r = 2
# final_U = np.zeros((len_rows, r))
# final_V = np.zeros((r, len_cols))
# # perturb = np.arange(start,last, interval) # uniform perturbation
# # reps = len(perturb)
# # print(reps)
# # order the optimization of the elements of U and V
# permutation = []
# for i in range(len_rows):
#     for j in range(r):
#         permutation.append((0,i,j))
#         # print(permutation)
# for i in range(r):
#     for j in range(len_cols):
#         permutation.append((1,i,j))
# np.random.shuffle(permutation)

# # # version 3
# # # optimize all elements
# # # initialize U and V with small perturbation
# # # M = original_M



# # for i in range(2):
# U = np.zeros((len_rows, r)) + 1
# V = np.zeros((r, len_cols)) + 1
# # print(U)
# # print(V)
# # if i == 1:
# #     U -= 0.0002
# #     V -= 0.0002
# iwant = len(permutation)
# standard = len(permutation)
# for perm in permutation:
#     tar = list(perm)
#     tar = list(map(str, perm))
    
#     standard -= 1
#     print(standard)
#     # print(rmse(M, np.dot(U, V)))
#     # print(perm)
#     if perm[0] == 0: # optimize u_ix
#         i = perm[1]
#         x = perm[2]
#         # print (i, x)
#         ret = optimum_u(M,U,V,i,x)
#         # if ret <= -30:
#         #     if rmse(M, np.dot(U,V)) < 0.9:
#         #         break
#         #     new_rmse = rmse(M, np.dot(U,V))
#         #     t.write("standard = " + str(iwant - standard) + "\n")
#         #     t.write(",".join(tar))
#         #     t.write('\n')
#         #     t.write("rmse = " + str(new_rmse) + "\n")
#         #     t.write(str(ret) + '\n')

#     else: # optimize v_xj
#         x = perm[1]
#         j = perm[2]
#         # print(x,j)
#         ret = optimum_v(M,U,V,x,j)
#         # if ret <= -30:
#         #     if rmse(M, np.dot(U,V)) < 0.9:
#         #         break
#         #     new_rmse = rmse(M, np.dot(U,V))
#         #     t.write("standard = " + str(iwant - standard) + "\n")
#         #     t.write(",".join(tar))
#         #     t.write('\n')
#         #     t.write("rmse = " + str(new_rmse) + "\n")
#         #     t.write(str(ret) + '\n')

#     # print("for each perm")
#     # print(U)
#     # print(V)
# # final_U += U
# # final_V += V

# final_U = final_U/float(2)
# # final_V = final_V/float(2)
# final_UV = np.dot(U, V)
# r_rmse = rmse(M, final_UV)
# print("r = %d" %r)
# print("final rmse = %f" %r_rmse)

# result = np.zeros((len_rows, len_cols))
# for row in range(len_rows):
#     for col in range(len_cols):
#         result[row][col] = (avg_users[row] + avg_items[col])/2

# f_rmse = rmse(original_M, result)
# print("final final rmse = %f" %f_rmse)



# # write result with predicted star
# g = open("output_testUV.txt", 'w')
# for info in pred_info:
#     uid = int(info[0])
#     mid = int(info[1])
#     add = (avg_users[uid] + avg_items[mid])/2
#     f_pred_star = final_UV[uid][mid]
#     # if uid == 551:
#     #     print(M[uid][mid])
#     #     print(add)
#     info[2] = str(f_pred_star)
#     g.write(",".join(info))


# # write result with predicted star
# g = open("output_poor.txt", 'w')
# # for info in pred_info:
# #     uid = int(info[0])
# #     mid = int(info[1])
# #     add = (avg_users[uid] + avg_items[mid])/2
# #     pred_star = final_UV[uid][mid] + add
# #     if uid == 551:
# #         print(final_UV[uid][mid])
# #         print(add)
# #     info[2] = str(pred_star)
# #     g.write(",".join(info))

# ## test result
# # result = np.zeros((len_rows, len_cols))
# # for row in range(len_rows):
# #     for col in range(len_cols):
# #         result[row][col] = M + (avg_users[uid] + avg_items[mid])/2
# # print(U)
# # print(V)
# # print(final_UV)
# # print(M)
# # print(result)
# # print(original_M)

