k = 0
x = k
for i in range(0, 1000000):
    x += 0.000001
x -= k
print('finally, x=%.8f' % x)