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

# print(min(uid_set))
# print(min(mid_set))
# make utility matrix M
len_rows = max(uid_set)+1
len_cols = max(mid_set)+1
M = np.zeros((len_rows, len_cols))
for info in ums:
    uid, mid, star = info
    M[uid, mid] = star
print (M.__repr__())

# normalize M
for irow in range(len_rows):
    u_stars = M[irow,:]
    nonzero_values = np.nonzero(u_stars)
    if np.count_nonzero(u_stars) > 0:
        avg_stars = u_stars[nonzero_values].mean()
        # avg_stars = np.sum(u_stars)/np.count_nonzero(u_stars)
        # print(np.sum(u_stars))
        # print(len(nonzero_values))
        # print (avg_stars)
    # avg_stars = u_stars[np.nonzero(u_stars)].mean()
    # avg_stars = np.nanmean(u_stars[u_stars!=0])
        for icol in range(len_cols):
            if (M[irow][icol]) != 0:
                M[irow][icol] -= avg_stars
print (M.__repr__())