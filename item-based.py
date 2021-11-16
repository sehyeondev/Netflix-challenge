import sys
# import numpy as np

file = sys.argv[1]
f = open(file, 'r')
lines = f.readlines()
test_file = sys.argv[2]
tf = open(test_file, 'r')
lines_tf = tf.readlines()

# uid_set = set()
# mid_set = set()
# ums = []
pred_info = [] # store information to be predicted

class Info:
    def __init__(self, uid, mid, star=0.0):
        self.uid = uid
        self.mid = mid
        self.star = star

# info_arr = []
# target_movies=[]
# # data preprocessing
# for line in lines:
#     info = line.split(',')
#     uid, mid, star = (int(info[0]), int(info[1]), float(info[2]))
#     uid_set.add(uid)
#     mid_set.add(mid)
#     ums.append((uid, mid, star))
#     info_arr.append(Info(uid,mid,star))

# for line in lines_tf:
#     info = line.split(',')
#     pred_info.append(info)
#     uid, mid = (int(info[0]), int(info[1]))
#     uid_set.add(uid)
#     mid_set.add(mid)
#     target_movies.append(Info(uid, mid))

# # make utility matrix M
# len_rows = max(uid_set)+1
# len_cols = max(mid_set)+1
# M = np.empty((len_rows, len_cols))
# M[:] = np.nan
# M_arr = [[] for _ in range(len_rows)]

# for info in info_arr:
#     uid, mid, star = info.uid, info.mid, info.star
#     M[uid, mid] = star
#     M_arr[uid].append(info)

# original_M = np.copy(M)
# # print(M)

# # if np.sum(~np.isnan(user_avg)) > 0:
# # user_avg = np.nanmean(M, axis=1)
# # if np.sum(~np.isnan(item_avg)) > 0:
# # item_avg = np.nanmean(M, axis=0)
# # # print(user_avg)
# # # print(item_avg)

# # norm = np.empty((len_rows, len_cols))
# # # normalize M
# # for i in range(len_rows):
# #     for j in range(len_cols):
# #         norm[i][j] = (user_avg[i] + item_avg[j])/float(2)
# # # print(norm)
# # M -= norm
# # # print(M)

# avg_users = np.empty(len_rows) # store average stars of each user
# avg_items = np.empty(len_cols) # store average stars of each movie
# avg_users[:] = np.nan
# avg_items[:] = np.nan

# # get average stars of movies
# for icol in range(len_cols):
#     m_stars = np.copy(M[:,icol])
#     if np.sum(~np.isnan(m_stars)) <= 0:
#         continue
#     # nonzero_index = np.nonzero(m_stars)
#     avg_stars = np.nanmean(m_stars)
#     avg_items[icol] = avg_stars

# # get average stars of users
# for irow in range(len_rows):
#     u_stars = np.copy(M[irow])
#     if np.sum(~np.isnan(u_stars)) <= 0:
#         continue
#     # nonzero_index = np.nonzero(u_stars)
#     avg_stars = np.nanmean(u_stars)
#     avg_users[irow] = avg_stars
#     # if irow == target_uid:
#     #     target_avg = avg_stars
#     # normalize M
#     # by subracting average of average stars of item and user
# norm = np.empty((len_rows, len_cols))
# norm[:] = np.nan
# # print(norm)
# # normalize M
# for i in range(len_rows):
#     for j in range(len_cols):
#         # print(norm[i][j])
#         norm[i][j] = (avg_users[i] + avg_items[j])/float(2)
# # print(norm)
# M -= norm
# # # print(M)
# #     u_stars[nonzero_index] -= (avg_stars + avg_items[nonzero_index])/2
# #     M[irow] = u_stars

# def cosine_distance(a, b):
#     mag_a = np.linalg.norm(a[np.nonzero(~np.isnan(a))])
#     # print(mag_a)
#     mag_b = np.linalg.norm(b[np.nonzero(~np.isnan(b))])
#     # print(np.nansum(a*b))
#     ret = np.nansum(a*b)/(mag_a*mag_b)
#     return ret

# g = open("known_output.txt", 'w')
# # ===================================
# # item-based collaborative filtering
# # ===================================
# # find 10 similar movies to each movie
# # and compute average ratings
# # v1. make prediction with normalized values
# # v2. make prediction with original values
# # len_tmovies = len(target_movies)
# # avg_ratings = np.empty(len_cols)
# # avg_ratings[:] = np.nan
# standard = len(target_movies)
# # for movie in target_movies: # for movie 1 to 1000
# for i in range(len(pred_info)):
#     info = pred_info[i]
#     movie = target_movies[i]
#     print("%d/%d" %(standard, len(target_movies)))
#     standard -= 1
#     target_mid = movie.mid
#     target_uid = movie.uid
#     # print(target_uid, target_mid)
#     movie_stars = np.copy(M[:,target_mid])
#     if sum(~np.isnan(movie_stars)) <= 0: # if movie doesn't exist
#         continue
#     m_distances = np.empty(len_cols) # store distances between target movie and all other movies
#     m_distances[:] = np.nan
#     for jcol in range(len_cols):
#         j_stars = np.copy(M[:,jcol])
#         if sum(~np.isnan(j_stars)) <= 0:
#             continue
#         cos_dis = cosine_distance(movie_stars, j_stars)
#         # print(cos_dis)
#         if target_mid == jcol:
#             # print("!!!!")
#             # print(movie_stars)
#             # print(j_stars)
#             # print(cos_dis)
#             continue
#         m_distances[jcol] = cos_dis
#     # isnan = 
#     # notnan = 
#     real_distances = m_distances[np.nonzero(~np.isnan(m_distances))]
#     # print(m_distances)
#     # print("except nan")
#     # print(real_distances)
#     ## here is a problem real distance is not useful
#     c = np.nonzero(np.isnan(m_distances))
#     m_distances[c] = -10**9
#     sim_mids = np.argpartition(m_distances, -10)[-10:] # 10 similar movies
#     # indices = sim_mids[np.argsort(-m_distances)[sim_mids]]
#     # print(indices)
#     # print(sim_mids)
#     # print(m_distances[sim_mids])
#     # print(m_distances[sim_mids])
#     # tuser_stars: ratings by target user for 10 similar movies
#     # tuser_stars = M[target_uid,sim_mids] # v1. 
#     tuser_stars = original_M[target_uid, sim_mids] # v2.
#     nz_idx = np.nonzero(~np.isnan(tuser_stars))[0]
#     if sum(~np.isnan(tuser_stars)) > 0: # if target user rates at least one movie among them
#         # print("stars that target rated among 10")
#         # print(tuser_stars[nz_idx])
#         avg_star = np.nanmean(tuser_stars) # take average the ratings
#         # avg_ratings[icol] = avg_star + target_avg # v1.
#         # if len(nz_idx) > 2:
#         # avg_ratings[icol] = avg_star # v2.
#         movie.star = avg_star
#         print(avg_star)
#     pred_star = target_movies[i].star
#     info[2] = str(pred_star)
#     g.write(",".join(info))
    
# # top20_movies = np.argpartition(avg_ratings, -20)[-20:]

# # g = open("my_output.txt", 'w')
# # # recommend 5 movies
# # for i in range(len(pred_info)):
# #     info = pred_info[i]
# #     pred_star = target_movies[i].star
# #     info[2] = str(pred_star)
# #     g.write(",".join(info))

# # item_base = []
# # for mid in top20_movies:
# #     item_base.append((mid, avg_ratings[mid]))
# # item_base.sort(key=lambda x: (-x[1],x[0]))
# # for i in range(5):
# #     ele = item_base[i]
# #     print("%d\t%f" %(ele[0], ele[1]))

import sys
import numpy as np
    
uid_set = set()
mid_set = set()
ums = []

info_arr = []
target_movies=[]
# data preprocessing
for line in lines:
    info = line.split(',')
    uid, mid, star = (int(info[0]), int(info[1]), float(info[2]))
    uid_set.add(uid)
    mid_set.add(mid)
    ums.append((uid, mid, star))
    info_arr.append(Info(uid,mid,star))

for line in lines_tf:
    info = line.split(',')
    pred_info.append(info)
    uid, mid = (int(info[0]), int(info[1]))
    uid_set.add(uid)
    mid_set.add(mid)
    target_movies.append(Info(uid, mid))

# make utility matrix M
len_rows = max(uid_set)+1
len_cols = max(mid_set)+1
M = np.zeros((len_rows, len_cols))
for info in ums:
    uid, mid, star = info
    M[uid, mid] = star
original_M = np.copy(M)

# normalize M
for irow in range(len_rows):
    u_stars = np.copy(M[irow])
    if np.count_nonzero(u_stars) > 0:
        nonzero_index = np.nonzero(u_stars)
        avg_stars = u_stars[nonzero_index].mean()
        # if irow == target_uid:
        #     target_avg = avg_stars
        u_stars[nonzero_index] -= avg_stars
        M[irow] = u_stars

def cosine_distance(a, b):
    mag_a = np.linalg.norm(a)
    mag_b = np.linalg.norm(b)
    # ret = np.dot(a, b)/(mag_a*mag_b)
    ret = np.matmul(a, np.transpose(b))/(mag_a*mag_b)
    return ret

# ===================================
# item-based collaborative filtering
# ===================================
# find 10 similar movies to each movie
# and compute average ratings
# v1. make prediction with normalized values
# v2. make prediction with original values
# avg_ratings = np.zeros(1001)

g = open("known_output2.txt", 'w')

standard = len(target_movies)
# for movie in target_movies: # for movie 1 to 1000
for i in range(len(pred_info)):
    info = pred_info[i]
    movie = target_movies[i]
    print("%d/%d" %(standard, len(target_movies)))
    standard -= 1
    target_mid = movie.mid
    target_uid = movie.uid
# for icol in range(1001): # for movie 1 to 1000
    movie_stars = np.copy(M[:,target_mid])
    if np.count_nonzero(movie_stars) <= 0: # if movie doesn't exist
        continue
    m_distances = np.zeros(len_cols) - 10**9 # store distances between icol movie and all other movies
    # m_distances[:] = -10**9 # very small values
    for jcol in range(len_cols):
        j_stars = np.copy(M[:,jcol])
        if np.count_nonzero(j_stars) <= 0:
            continue
        if target_mid == jcol:
            continue
        cos_dis = cosine_distance(movie_stars, j_stars)
        m_distances[jcol] = cos_dis
    sim_mids = np.argpartition(m_distances, -100)[-100:] # 10 similar movies
    # tuser_stars: ratings by target user for 10 similar movies
    # tuser_stars = M[target_uid,sim_mids] # v1. 
    tuser_stars = original_M[target_uid, sim_mids] # v2.
    if np.count_nonzero(tuser_stars) > 0: # if target user rates at least one movie among them
        avg_star = tuser_stars[np.nonzero(tuser_stars)].mean() # take average the ratings
        # avg_ratings[icol] = avg_star + target_avg # v1.
        # avg_ratings[ta] = avg_star # v2.
        movie.star = avg_star
        print(avg_star)
    pred_star = target_movies[i].star
    info[2] = str(pred_star)
    g.write(",".join(info))
