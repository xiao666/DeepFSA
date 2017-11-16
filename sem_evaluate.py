

import numpy as np

def sem_eval(G,P):
    
    sum_GxP=0
    sum_G_square=0
    sum_P_square=0
    for i in range(len(G)):
        temp0=G[i]*P[i]
        temp1=G[i]*G[i]
        temp2=P[i]*P[i]
        sum_GxP=sum_GxP+temp0
        sum_G_square=sum_G_square+temp1
        sum_P_square=sum_P_square+temp2

    cosine_G_P=sum_GxP/(np.sqrt(sum_G_square)*np.sqrt(sum_P_square))

    return cosine_G_P