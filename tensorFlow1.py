#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


from tensorflow import keras
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[3]:


keras.datasets.mnist.load_data()


# In[4]:


(X_train, y_train),(X_test, y_test) = keras.datasets.mnist.load_data()


# In[5]:


len(X_train)


# In[6]:


len(X_test)


# In[7]:


X_train[0].shape


# In[8]:


X_train[0]


# In[11]:


plt.matshow(X_train[5])


# In[12]:


y_train[:5]


# In[22]:


X_train = X_train / 255
X_test = X_test / 255


# In[23]:


X_train[0]


# In[24]:


#flatten the images from 28x28 to single dimensional array 
X_train_flattened = X_train.reshape(len(X_train),28*28)
X_train_flattened.shape


# In[25]:


#flatten the images from 28x28 to single dimensional array 
X_test_flattened = X_test.reshape(len(X_test),28*28)
X_test_flattened.shape


# In[26]:


X_train_flattened


# In[27]:


model = keras.Sequential([
    keras.layers.Dense(10,input_shape=(784,),activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

model.fit(X_train_flattened, y_train,epochs=5)


# In[28]:


model.fit(X_test_flattened, y_test,)


# In[29]:


plt.matshow(X_test[0])


# In[30]:


y_predicted = model.predict(X_test_flattened)
y_predicted[0]


# In[31]:


np.argmax(y_predicted[0])


# In[33]:


plt.matshow(X_test[7])


# In[32]:


np.argmax(y_predicted[7])


# In[34]:


plt.matshow(X_test[77])


# In[35]:


np.argmax(y_predicted[77])


# In[36]:


y_predicted_labels = [np.argmax(i) for i in y_predicted]
y_predicted_labels[:5]


# In[37]:


y_test[:5]


# In[38]:


confusionMatrix = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)


# In[39]:


confusionMatrix


# In[40]:


import seaborn as sns
plt.figure(figsize = (10,7))
sns.heatmap(confusionMatrix, annot=True,fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[41]:


#adding hidden layer to improve performance
model = keras.Sequential([
    keras.layers.Dense(100,input_shape=(784,),activation='relu'),
    keras.layers.Dense(10,activation='sigmoid')

])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

model.fit(X_train_flattened, y_train,epochs=5)


# In[42]:


model.evaluate(X_test_flattened, y_test)


# In[44]:


y_predicted = model.predict(X_test_flattened)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
confusionMatrix = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)

plt.figure(figsize = (10,7))
sns.heatmap(confusionMatrix, annot=True,fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[ ]:


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28*28)),
    keras.layers.Dense(100,activation='relu'),
    keras.layers.Dense(10,activation='sigmoid')

])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

model.fit(X_train,y_train,epochs=5)

