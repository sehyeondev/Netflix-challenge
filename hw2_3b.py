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
original_M = np.copy(M)

# normalize M
for irow in range(len_rows):
    u_stars = np.copy(M[irow])
    if np.count_nonzero(u_stars) > 0:
        nonzero_index = np.nonzero(u_stars)
        avg_stars = u_stars[nonzero_index].mean()
        if irow == target_uid:
            target_avg = avg_stars
        u_stars[nonzero_index] -= avg_stars
        M[irow] = u_stars

def cosine_distance(a, b):
    mag_a = np.linalg.norm(a)
    mag_b = np.linalg.norm(b)
    # ret = np.dot(a, b)/(mag_a*mag_b)
    ret = np.matmul(a, np.transpose(b))/(mag_a*mag_b)
    return ret

# =====================================
# user-based collaborative filtering
# =====================================
# find 10 most similar users
distances = np.zeros(len_rows) - 10**9 # very small values
target_stars = M[target_uid]
for irow in range(len_rows):
    u_stars = M[irow]
    if np.count_nonzero(u_stars) > 0:
        if irow == target_uid:
            continue
        cos_dis = cosine_distance(target_stars, u_stars)
        distances[irow] = cos_dis
sim_uids = np.argpartition(distances, -10)[-10:]

# compute average ratings of I for similar users
# v1. make prediction with normalized values
# v2. make prediction with original values
stars = np.zeros(1001)
for icol in range(1001):
    # m_stars = np.copy(M[sim_uids,icol]) # v1.
    m_stars = np.copy(original_M[sim_uids,icol]) # v2.
    if np.count_nonzero(m_stars) > 0:
        avg_stars = m_stars[np.nonzero(m_stars)].mean()
        # stars[icol] = avg_stars + target_avg # v1.
        stars[icol] = avg_stars # v2.
rcmd_mids = np.argpartition(stars, -20)[-20:]

# recommend 5 movies
user_base = []
for mid in rcmd_mids:
    user_base.append((mid, stars[mid]))
user_base.sort(key=lambda x: (-x[1],x[0]))
for i in range(5):
    ele = user_base[i]
    print("%d\t%f" %(ele[0], ele[1]))


# ===================================
# item-based collaborative filtering
# ===================================
# find 10 similar movies to each movie
# and compute average ratings
# v1. make prediction with normalized values
# v2. make prediction with original values
avg_ratings = np.zeros(1001)
for icol in range(1001): # for movie 1 to 1000
    i_stars = np.copy(M[:,icol])
    if np.count_nonzero(i_stars) <= 0: # if movie doesn't exist
        continue
    m_distances = np.zeros(len_cols) - 10**9 # store distances between icol movie and all other movies
    # for jcol in range(1001, len_cols): # v3.
    for jcol in range(len_cols):
        j_stars = np.copy(M[:,jcol])
        if np.count_nonzero(j_stars) <= 0:
            continue
        if icol == jcol:
            continue
        cos_dis = cosine_distance(i_stars, j_stars)
        m_distances[jcol] = cos_dis
    sim_mids = np.argpartition(m_distances, -10)[-10:] # 10 similar movies
    # tuser_stars: ratings by target user for 10 similar movies
    # tuser_stars = M[target_uid,sim_mids] # v1. 
    tuser_stars = original_M[target_uid, sim_mids] # v2.
    if np.count_nonzero(tuser_stars) > 0: # if target user rates at least one movie among them
        avg_star = tuser_stars[np.nonzero(tuser_stars)].mean() # take average the ratings
        # avg_ratings[icol] = avg_star + target_avg # v1.
        avg_ratings[icol] = avg_star # v2.
top20_movies = np.argpartition(avg_ratings, -20)[-20:]

# recommend 5 movies
item_base = []
for mid in top20_movies:
    item_base.append((mid, avg_ratings[mid]))
item_base.sort(key=lambda x: (-x[1],x[0]))
for i in range(5):
    ele = item_base[i]
    print("%d\t%f" %(ele[0], ele[1]))