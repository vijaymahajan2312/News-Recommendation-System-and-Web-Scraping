#!/usr/bin/env python
# coding: utf-8

# # News-Recommendation and Web Scraping

# # Import Libraries

# In[1]:


import pandas as pd
from requests import get
from bs4 import BeautifulSoup


# # Extracting details of web-page

# In[2]:


url = "https://www.indiatoday.in/india?page=1"


# In[3]:


# Specify number of pages upto which I want to scrape articles.
noPages=5


# In[4]:


Urllinks =[]
for i in range(1,noPages+1):
    print("Processing Page: ", i)
    url = "https://www.indiatoday.in/india?page="+str(i)+"/"
    Urllinks.append(url)


# In[5]:


Headlines = []
for i in Urllinks:
    data = get(i)
    soup = BeautifulSoup(data.content,'html.parser')
    for i in soup.find_all('div',class_="catagory-listing"):
        Headlines.append(i.text)
        
Headlines


# In[6]:


len(Headlines)


# # Cleaning The Data

# In[7]:


p_art =[]
for i in Headlines:
    q = i.upper()
    import re
    q = re.sub("[^A-Z0-9 ]","",q)
    from nltk.stem import PorterStemmer
    tk_q = q.split(" ")
    sent = ""
    for j in tk_q:
        ps = PorterStemmer()
        sent = sent + " " + ps.stem(j).upper()
    p_art.append(sent)


# In[8]:


p_art


# # Cluster for News-Recommendation 

# In[9]:


from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer()
A = tf.fit_transform(p_art).toarray()


# In[10]:


from sklearn.cluster import KMeans
km = KMeans(n_clusters=5)
cl_res = km.fit(A)


# In[11]:


cl_res.labels_


# In[12]:


Q = pd.DataFrame(p_art,columns=["Article"])
Q['Cluster']=cl_res.labels_


# In[13]:


Q.head(15)


# # Categories for Clusters

# In[14]:


E = {1 : "Technology",
2 : "Politics",
3 : "Entertainment",
4 : "Geopolotics",
0 : "Sport"}


# In[15]:


R = []
for i in Q.Cluster:
    R.append(E[i])

Q['category'] = R


# In[16]:


Q

