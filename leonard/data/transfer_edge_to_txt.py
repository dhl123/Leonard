import numpy as np
a=np.load('edges200m.npy',allow_pickle=True)
for i in range(len(a)):
    a[i]=str(a[i])
f=open('edges200m.txt','w')
f.write('\n'.join(a))
