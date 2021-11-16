file = open("compare.txt", 'r')
g = open("known_test.txt", 'w')

lines = file.readlines()
for line in lines:
    l = line.split(',')
    del l[2]
    l[1] += ','
    g.write(','.join(l))

