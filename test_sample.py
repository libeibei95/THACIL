import random
from time import time

t1 = time()
for i in range(100):
    random.sample(list(range(700000)), 5000)
print(time()-t1)

# t2 = time()
# for i in range(100):
#     random.choice(list(range(10000)))
# print(time()-t2)