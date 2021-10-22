import sys
import numpy as np

file = sys.argv[1]
f = open(file, 'r')
lines = f.readlines()

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
# print (M.__repr__())

# normalize M
for irow in range(len_rows):
    u_stars = M[irow]
    if np.count_nonzero(u_stars) > 0:
        avg_stars = u_stars[np.nonzero(u_stars)].mean()
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

# a = np.array([3,4])
# b = np.array([6,8])
# print(cosine_distance(a,b))

# =====================================
# user-based collaborative filtering
# =====================================
target_uid = 600
dis_dic = {}
target_stars = M[target_uid]
# compute cosine distances between target user and all users
for irow in range(len_rows):
    if irow == target_uid:
        continue
    u_stars = M[irow]
    if np.count_nonzero(u_stars) > 0:
        cos_dis = cosine_distance(M[target_uid], u_stars)
        if cos_dis not in dis_dic.keys():
            dis_dic[cos_dis] = []
        dis_dic[cos_dis].append(irow)
# print (dis_dic)

# find 10 most similar users
top_ten_uid = []
dis_list = dis_dic.keys()
dis_list.sort()
dis_list.reverse()
while len(top_ten_uid) < 10:
    max_dis = dis_list.pop(0)
    sim_uids = dis_dic[max_dis]
    for uid in sim_uids:
        top_ten_uid.append(uid)
print(top_ten_uid)

# compute average ratings of I for similar users
stars = {}
sim_matrix = np.zeros((10,len_cols))
i = 0
for uid in top_ten_uid:
    sim_matrix[i] = M[uid]
    i+=1
# print(sim_matrix.__repr__())

for icol in range(len_cols):
    m_stars = sim_matrix[:,icol]
    if np.count_nonzero(m_stars) > 0:
        avg_stars = m_stars[np.nonzero(m_stars)].mean()
        if avg_stars not in stars.keys():
            stars[avg_stars] = []
        stars[avg_stars].append(icol)
# print(stars)

recommend = []
star_list = stars.keys()
star_list.sort()
print(max(star_list))
# star_list.reverse()
# print(star_list)
while len(recommend) < 5:
    max_star = star_list.pop()
    movies = stars[max_star]
    print(max_star)
    print(movies)
    for mid in movies:
        recommend.append(mid)
print(recommend)

# ===================================
# item-based collaborative filtering
# ===================================
