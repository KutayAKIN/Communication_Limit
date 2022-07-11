import numpy as np
n_class =15
n_device =20
#iter_con = 3
C_mat = np.zeros((n_class,n_class,n_device))
for i in range(n_device):
    for j in range(n_class):
        for k in range(n_class):
            if j != k:
                C_mat[k,j,i] = np.absolute(np.random.normal(0,1,1))
            else:
                continue
A = np.sum(C_mat, axis=0)
for i in range(n_device):
    for j in range(n_class):
        for k in range(n_class):
            if j == k:
                while C_mat[k,j,i] < A[j,i]:
                    C_mat[k,j,i] = np.absolute(np.random.randint(np.ceil(A[j,i])-2,high = 5*np.ceil(A[j,i]), size = 1))
            else:
                continue
B = np.sum(C_mat, axis=0)
for i in range(n_device):
    for j in range(n_class):
        for k in range(n_class):
            C_mat[k,j,i] = C_mat[k,j,i] / B[j,i]

best_selections = np.zeros((n_device-1,n_device), dtype=np.int8)
for i in range(n_device):
    used = np.array([i])
    base = C_mat[:,:,i]
    besti = np.empty([n_device-1], dtype=np.int8)
    for k in range(n_device-1):
        
        max_score = np.array([0])
        for j in range(n_device):
            if j in used:
                continue
            else:
                mat = base + C_mat[:,:,j]
                score = np.linalg.det(np.transpose(mat))/np.math.factorial(n_class)
                if score > max_score:
                    max_mat = C_mat[:,:,j]
                    max_score = score
                    index = j
        used = np.append(used, index)
        besti[k] = index
        base = base + C_mat[:,:,index]
    best_selections[:,i] = besti
print(np.transpose(best_selections))
np.savetxt('test.text', best_selections, delimiter=',')