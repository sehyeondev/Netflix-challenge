import sys
import numpy as np

file = sys.argv[1]
f = open(file, 'r')
lines = f.readlines()
target_uid = 600
target_avg = 0

uid_set = set()
mid_set = set()
ums = []
# data preprocessing
for line in lines:
    info = line.split(',')
    uid, mid, star = (int(info[0]), int(info[1]), float(info[2]))
    uid_set.add(uid)
    mid_set.add(mid)
    ums.append((uid, mid, star))

# make utility matrix M
len_rows = max(uid_set)+1
len_cols = max(mid_set)+1
M = np.zeros((len_rows, len_cols))
for info in ums:
    uid, mid, star = info
    M[uid, mid] = star

# normalize M
for irow in range(len_rows):
    u_stars = M[irow]
    if np.count_nonzero(u_stars) > 0:
        avg_stars = u_stars[np.nonzero(u_stars)].mean()
        if irow == target_uid:
            target_avg = avg_stars
        for icol in range(len_cols):
            if (M[irow][icol]) != 0:
                M[irow][icol] -= avg_stars

def cosine_distance(a, b):
    mag_a = np.linalg.norm(a)
    mag_b = np.linalg.norm(b)
    ret = np.dot(a, b)/(mag_a*mag_b)
    return ret

# =====================================
# user-based collaborative filtering
# =====================================
# find 10 most similar users
distances = np.zeros(len_rows)
target_stars = M[target_uid]
for irow in range(len_rows):
    u_stars = M[irow]
    if np.count_nonzero(u_stars) > 0:
        cos_dis = cosine_distance(target_stars, u_stars)
        distances[irow] = cos_dis
sim_uids = np.argpartition(distances, -10)[-10:]
# print(sim_uids)

# compute average ratings of I for similar users
stars = np.zeros(1001)
for icol in range(1001):
    m_stars = M[sim_uids,icol]
    if np.count_nonzero(m_stars) > 0:
        avg_stars = m_stars[np.nonzero(m_stars)].mean()
        stars[icol] = avg_stars
rcmd_mids = np.argpartition(stars, -5)[-5:]
rcmd_stars = stars[rcmd_mids] + target_avg

# recommend 5 movies
user_base = []
for i in range(5):
    user_base.append((rcmd_mids[i], rcmd_stars[i]))
user_base.sort(key=lambda x: -x[1])
for ele in user_base:
    print("%d\t%f" %(ele[0], ele[1]))

# ===================================
# item-based collaborative filtering
# ===================================
# find 10 similar movies to each movie
# and compute average ratings
avg_ratings = np.zeros(1001)
for icol in range(1001): # for movie 1 to 1000
    i_stars = M[:,icol]
    if np.count_nonzero(i_stars) <= 0: # if movie doesn't exist
        continue
    m_distances = np.zeros(len_cols) # store distances between icol movie and all other movies
    for jcol in range(len_cols):
        j_stars = M[:,jcol]
        if np.count_nonzero(j_stars) <= 0:
            continue
        cos_dis = cosine_distance(i_stars, j_stars)
        m_distances[jcol] = cos_dis
    sim_mids = np.argpartition(m_distances, -10)[-10:] # 10 similar movies
    tuser_stars = M[target_uid,sim_mids] # ratings by target user for 10 similar movies
    if np.count_nonzero(tuser_stars) > 0: # if target user rates at least one movie among them
        avg_star = tuser_stars[np.nonzero(tuser_stars)].mean() # take average the ratings
        avg_ratings[icol] = avg_star
        top20_movies = np.argpartition(avg_ratings, -20)[-20:]

# recommend 5 movies
item_base = []
for mid in top20_movies:
    item_base.append((mid, avg_ratings[mid] + target_avg))
item_base.sort(key=lambda x: (-x[1],x[0]))
for i in range(5):
    ele = item_base[i]
    print("%d\t%f" %(ele[0], ele[1]))