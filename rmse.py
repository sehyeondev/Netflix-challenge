import numpy as np

f = open("compare.txt", 'r')
gname = "output_testfinal.txt"
g = open(gname, 'r')

f_lines = f.readlines()
g_lines = g.readlines()

diff_sum = 0
denom = len(f_lines)
for i in range(len(f_lines)):
    nf = f_lines[i].split(',')[2]
    ng = g_lines[i].split(',')[2]
    if ng == "nan":
        denom -= 1
        continue
    diff_sum += np.square(float(nf) - float(ng))
print(gname)
print(np.sqrt(diff_sum/float(denom)))

