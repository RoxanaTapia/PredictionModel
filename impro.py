
# coding: utf-8

# In[2]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[3]:


import numpy as np
import pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LinearRegression


# In[4]:


c1 = np.loadtxt('c1_matrix.txt')
c2 = np.loadtxt('c2_matrix.txt')
h1 = np.loadtxt('h1_matrix.txt')
h2 = np.loadtxt('h2_matrix.txt')


# In[6]:


get_ipython().run_cell_magic('time', '', "for i in range(len(c1)):\n    y=c1[i]\n    #select the value between 10 and 130\n    cond=np.where(((y>130)|(y<10)),-1,y)\n    y=np.delete(cond,np.argwhere(cond==-1))\n    \n    x=np.arange(len(y))\n    model = LinearRegression()\n    model.fit(np.reshape(x,[len(x),1]), np.reshape(y,[len(y),1]))\n    yy = model.predict(np.reshape(x,[len(x),1]))\n    \n    w = model.coef_[0][0] # parameters of model\n    b = model.intercept_[0] #intercept of model\n    \n    g1=np.where(y>w*x+b,y,-1)\n    g11=np.delete(g1,np.argwhere(g1==-1))\n    g11_mean=np.mean(g11)\n\n    g2=np.where(y<w*x+b,y,-1)\n    g21=np.delete(g2,np.argwhere(g2==-1))\n    g21_mean=np.mean(g21)\n    \n    print('Patient',i+1, 'in C1:','%0.2f'%g11_mean,'%0.2f'%g21_mean,len(g11),len(g21))")


# In[13]:


c2 = np.loadtxt('c2_matrix.txt')
for i in range(len(c2)):
    y=c2[i]
    #select the value between 10 and 130
    cond=np.where(((y>130)|(y<10)),-1,y)
    y=np.delete(cond,np.argwhere(cond==-1))
    
    x=np.arange(len(y))
    model = LinearRegression()
    model.fit(np.reshape(x,[len(x),1]), np.reshape(y,[len(y),1]))
    yy = model.predict(np.reshape(x,[len(x),1]))
    
    w = model.coef_[0][0] # parameters of model
    b = model.intercept_[0] #intercept of model
    
    g1=np.where(y>w*x+b,y,-1)
    g11=np.delete(g1,np.argwhere(g1==-1))
    g11_mean=np.mean(g11)

    g2=np.where(y<w*x+b,y,-1)
    g21=np.delete(g2,np.argwhere(g2==-1))
    g21_mean=np.mean(g21)
    
    print('Patient',i+1, 'in C2:','%0.2f'%g11_mean,'%0.2f'%g21_mean,len(g11),len(g21))


# In[14]:


h1 = np.loadtxt('h1_matrix.txt')
for i in range(len(h1)):
    y=h1[i]
    #select the value between 10 and 130
    cond=np.where(((y>130)|(y<10)),-1,y)
    y=np.delete(cond,np.argwhere(cond==-1))
    
    x=np.arange(len(y))
    model = LinearRegression()
    model.fit(np.reshape(x,[len(x),1]), np.reshape(y,[len(y),1]))
    yy = model.predict(np.reshape(x,[len(x),1]))
    
    w = model.coef_[0][0] # parameters of model
    b = model.intercept_[0] #intercept of model
    
    g1=np.where(y>w*x+b,y,-1)
    g11=np.delete(g1,np.argwhere(g1==-1))
    g11_mean=np.mean(g11)

    g2=np.where(y<w*x+b,y,-1)
    g21=np.delete(g2,np.argwhere(g2==-1))
    g21_mean=np.mean(g21)
    
    print('Patient',i+1, 'in H1:','%0.2f'%g11_mean,'%0.2f'%g21_mean,len(g11),len(g21))


# In[15]:


h2 = np.loadtxt('h2_matrix.txt')
for i in range(len(h2)):
    y=h2[i]
    #select the value between 10 and 130
    cond=np.where(((y>130)|(y<10)),-1,y)
    y=np.delete(cond,np.argwhere(cond==-1))
    
    x=np.arange(len(y))
    model = LinearRegression()
    model.fit(np.reshape(x,[len(x),1]), np.reshape(y,[len(y),1]))
    yy = model.predict(np.reshape(x,[len(x),1]))
    
    w = model.coef_[0][0] # parameters of model
    b = model.intercept_[0] #intercept of model
    
    g1=np.where(y>w*x+b,y,-1)
    g11=np.delete(g1,np.argwhere(g1==-1))
    g11_mean=np.mean(g11)

    g2=np.where(y<w*x+b,y,-1)
    g21=np.delete(g2,np.argwhere(g2==-1))
    g21_mean=np.mean(g21)
    
    print('Patient',i+1, 'in H2:','%0.2f'%g11_mean,'%0.2f'%g21_mean,len(g11),len(g21))


# In[34]:


y=c1[0]
#select the value between 10 and 130
cond=np.where(((y>130)|(y<10)),-1,y)
y=np.delete(cond,np.argwhere(cond==-1))
y.shape


# In[35]:



#the first patient in c1
x=np.arange(len(y))
x.shape
model = LinearRegression()
model.fit(np.reshape(x,[len(x),1]), np.reshape(y,[len(y),1]))
yy = model.predict(np.reshape(x,[len(x),1]))
w = model.coef_[0][0] # parameters of model
b = model.intercept_[0] #intercept of model

plt.figure()
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True) 
plt.plot(x,y,'b') 
plt.plot(x,yy,'g-') 
plt.show()


# In[36]:


g1=np.where(y>w*x+b,y,-1)
g11=np.delete(g1,np.argwhere(g1==-1))
g11_mean=np.mean(g11)

g2=np.where(y<w*x+b,y,-1)
g21=np.delete(g2,np.argwhere(g2==-1))
g21_mean=np.mean(g21)

print('two means and size of first patient in C1: \n',g11_mean,g21_mean,len(g11),len(g21))

