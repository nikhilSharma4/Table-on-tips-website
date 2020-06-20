#!/usr/bin/env python
# coding: utf-8

# In[43]:


from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import random
import math


# In[44]:


def first_no(m,n,X):
    
    '''
    selecting first no:
    
    inputs
      -m(int): required sample contain numbers between 1 and m
      -n(int): number of numbers required in sample
      -counts(list of lists or np array..): contains counts of 
       each of previous right or wrong submissions
    outputs
      -sample1: list of required numbers. size should be n
    '''
    
    # Exception
    if n> (m*10):
        
        return [i for i in range(1,m+1) for j in range(1,11)]
        #return all possible combinations
    else:
        # Choosing only m questions
        X = X[:m]
    
    
    # Normalizing the data
    X.append(30)
    X = np.array(X).reshape(-1,1)
    mms = MinMaxScaler()
    counts = (mms.fit_transform(X)).reshape(1,X.shape[0])
    
    
    # list Flatten 2D to 1D
    counts = [j for sub in counts for j in sub]
    
    
    # Increasing probablity for questions of 0 chance
    counts = [x or 0.02 for x in counts]     # 2% chance of appearing
    
    
    # Generating n values from probability distribution of the values
    '''    count_ = counts.copy()
        sample = []
        for i in range(0,n):
            rand = (random.choices(population=count_,weights=count_,k=1))[0]
            sample.append(counts.index(rand)+1) 
            count_.remove(rand) 
    '''
    count_ = counts.copy()
    count_ = count_[:-1]
    rand = (random.choices( population=range(1,len(count_)+1) , weights=count_ , k=n )) 
#    sample = [counts.index(j)+1 for j in rand]
    
    return rand 
        


# In[45]:


def normal_choice(lst, mean=None, stddev=None):
    if mean is None:
        # if mean is not specified, use center of list
        mean = (len(lst) - 1) / 2

    if stddev is None:
        # if stddev is not specified, let list be -3 .. +3 standard deviations
        stddev = len(lst) / 6
    if len(lst) == 0:
        return 0

    while True:
        index = int(random.normalvariate(mean, stddev) + 0.5)
        if 0 <= index < len(lst):
            return lst[index]


# In[46]:


def second_no(n,count,first):
    
    '''
    selecting second no:
    
    inputs
      -n(int): number of numbers required in sample
      -counts(list of lists or np array..): contains counts 
       of each of previous right or wrong submissions
    outputs
      -sample2: list of required numbers. size should be n.
       each of the numbers should be between 1 & 10
    '''
    if len(first) < n:
        return [j for i in range(1,int(len(first)/10)+1) for j in range(1,11)]
    else:
        X = count.copy() 
        # Normalizing the data
        X = np.array(X).reshape(-1,1)
        mms = MinMaxScaler()
        counts = (mms.fit_transform(X)).reshape(1,X.shape[0])

        # list Flatten 2D to 1D
        counts = [j for sub in counts for j in sub]
        #print(counts)
        # Increasing probablity for questions of 0 chance
        #counts = [x or 0.02 for x in counts]     # 2% chance of appearing
        j= 0.02
        for (i,x) in enumerate(counts):
            if (x == 0.0):
                counts[i] = j
                j+=0.001

        # Use this for non-unique values
        sample =[]
        final =[]
        for i in range(0,n):
            count_ = (counts.copy())

            if (count_.count(count_[0])> 2) and (count_.count(count_[1])> 2) and  (count_.count(count_[2])> 2) and (count_.count(count_[3])> 2):
                ran =[1,2,3,4,5,6,7,8,9,10]
                z = (random.choices( population=ran , k=1 ))
                while (((first[i],z[0])) in final):
                    ran = [i for i in ran if (i != z[0])]
                    z = (random.choices( population=ran , k=1 ))
                final.append((first[i],z[0]))
                sample.append(z[0])    

            else:
                rand = normal_choice(count_)
                z = math.ceil(((rand)/max(count_))*10)
                while ((first[i],z) in final):
                    count_ = [i for i in count_ if (i != rand)]
                    rand = normal_choice(count_)
                    if (rand == 0):
                        ran =[1,2,3,4,5,6,7,8,9,10]
                        z = (random.choices( population=ran , k=1 ))
                        while (((first[i],z[0])) in final):
                            ran = [i for i in ran if (i != z[0])]
                            z = (random.choices( population=ran , k=1 ))   
                        z = z[0] 
                    else:
                        z =counts.index(rand) +1    
                sample.append(z)
                final.append((first[i],z))


        return sample        


# In[47]:


def str_to_int(list_):
    list_ = list_.split(',')
    return [int(i) for i in list_]


# In[48]:


def combined_pairs(X1,X2):
    return list(zip(X1,X2))


# In[53]:


X1 = first_no(4,50,[10, 0, 5, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
print(X1)


# In[54]:


#X2 = second_no(10,[0]*10,X1)
X2 = second_no(50,[8, 10, 6, 0, 3, 7, 0, 0, 7, 0],X1)
print(X2)


# In[55]:


X3 = str_to_int('1,2,3,4,5')
print(X3)


# In[52]:


print(combined_pairs(X1,X2))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




