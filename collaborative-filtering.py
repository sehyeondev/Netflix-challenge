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
# print("original len_cols = %f" %len_cols)
M = np.zeros((len_rows, len_cols))
for info in ums:
    uid, mid, star = info
    M[uid, mid] = star
# print (M.__repr__())

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
# print (M.__repr__())

def cosine_distance(a, b):
    mag_a = np.linalg.norm(a)
    mag_b = np.linalg.norm(b)
    ret = np.dot(a, b)/(mag_a*mag_b)
    # print (ret)
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
recommend = []
for i in range(5):
    recommend.append((rcmd_mids[i], rcmd_stars[i]))
recommend.sort(key=lambda x: -x[1])
for ele in recommend:
    print("%d\t%f" %(ele[0], ele[1]))

# ===================================
# item-based collaborative filtering
# ===================================
