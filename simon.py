import sys
import numpy as np

file = sys.argv[1]
f = open(file, 'r')
lines = f.readlines()
test_file = sys.argv[2]
tf = open(test_file, 'r')
lines_tf = tf.readlines()

uid_set = set()
mid_set = set()
pred_info = [] # store information to be predicted

class Info:
    def __init__(self, uid, mid, star):
        self.uid = uid
        self.mid = mid
        self.star = star
info_arr = []

# data preprocessing
for line in lines:
    info = line.split(',')
    uid, mid, star = (int(info[0]), int(info[1]), float(info[2]))
    uid_set.add(uid)
    mid_set.add(mid)
    info_arr.append(Info(uid,mid,star))

for line in lines_tf:
    info = line.split(',')
    pred_info.append(info)
    uid, mid = (int(info[0]), int(info[1]))
    uid_set.add(uid)
    mid_set.add(mid)

len_rows = max(uid_set)+1
len_cols = max(mid_set)+1

def predict(uid,mid,U,V):
    p = np.dot(U[uid], V[mid]) # predict Mij with U and V
    if p > 5:
        return 5
    elif p < 1:
        return 1
    return p

def optimize(k,U,V):
    denom = 0
    lrate = 0.035
    reg = 0.01
    for info in info_arr:
        error = info.star - predict(info.uid, info.mid, U, V) # error between original rating and predicted rating
        denom += 1
        uTemp = U[info.uid][k]
        vTemp = V[info.mid][k]
        U[info.uid][k] += lrate*(error*vTemp - reg*uTemp)
        V[info.mid][k] += lrate*(error*uTemp - reg*vTemp)
    return np.sqrt((error**2)/denom)

# ===========================
# UV decomposition
# ===========================
# constants for optimization
old_error = 10**6
r = 37
max_epoch = 30
threshold = 0.0001

# get average of all ratings
sum_star = 0
num = 0
for info in info_arr:
    sum_star += info.star
    num += 1
average_all = float(sum_star)/float(num)

# initialization
init = np.sqrt(average_all/r)
U = np.zeros((len_rows, r)) + init
V = np.zeros((len_cols, r)) + init

# optimize U and V matrix
for k in range(r):
    for epoch in range(max_epoch):
        error = optimize(k,U,V)
        if abs(old_error - error) < threshold:
            break
        old_error = error

# wirte output
final_UV = np.dot(U, V.T)
g = open("output.txt", 'w')
for info in pred_info:
    uid = int(info[0])
    mid = int(info[1])
    pred_star = final_UV[uid][mid]
    info[2] = str(pred_star)
    g.write(",".join(info))