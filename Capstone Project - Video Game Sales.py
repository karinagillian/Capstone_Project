#!/usr/bin/env python
# coding: utf-8

# ## Overview
# 
# This project is an opportunity to use the concepts and techniques I have learned in throughout this course and apply it using real world data. Through exploratory data analysis and multiple linear regression this report will generate insights into what genres & platorms have the highest number of sales globally. 
# 
# ![games-banner-1140x400-1.png](attachment:games-banner-1140x400-1.png)

# ## Business Problem
# 
# The primary objective of this project is to analyze historical video game sales data, exploring patterns, identifying successful genres, platforms, and publishers, and deriving meaningful conclusions to inform strategic decision-making within the gaming industry.
# 
# The dependent variable is Global Sales and will be the main focus to make predictions through the use of multiple linear regression models. 
# 
# This data set does not have quantitative independent variables for regression analysis. Regional sales are just the sub-totals of global sales. They are highly correlated with the dependent variable. Thus, we need to convert our categorical variables, Platform, Genre and Publisher, into numerical variables and use them in the regression models.
# 
# 
# In this data set there are 11 columns. Their names and data types as follows:
# 
# Rank - Ranking of overall sales
# 
# Name - The title of the game
# 
# Platform - Platform the game was relased on (i.e. PC,PS4, etc.)
# 
# Year - Year of the game's release
# 
# Genre - Genre of the game (i.e. action, role-play, sports, etc.)
# 
# Publisher - Publisher of the game  
# 
# NA_Sales - Sales in North America (in millions)
# 
# EU_Sales - Sales in Europe (in millions)
# 
# JP_Sales - Sales in Japan (in millions)
# 
# Other_Sales - Sales in the rest of the world (in millions)
# 
# Global_Sales - Total worldwide sales.

# In[1]:


# Import relevant packages
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objs as go

import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf

import scipy.stats as stats

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
get_ipython().run_line_magic('matplotlib', 'inline')


# # Import & Clean the Data

# In[2]:


df = pd.read_csv ('vgsales.csv', index_col=0)


# In[3]:


display(df.head())
display(df.tail())


# In[4]:


df.describe().style.background_gradient(cmap = 'Blues')


# In[5]:


df.columns


# In[6]:


df.describe(include = 'O').T


# In[7]:


df.info()


# This dataset contains 11 variables & 16,598 observations

# In[8]:


df.isnull().sum()


# This dataset has 271 NaN values for Year and 58 NaN values for Publisher

# In[9]:


df.isnull().sum()


# We have 271 NaN values for Year and 58 NaN values for Publisher

# In[10]:


# Removing the missing value rows in the dataset
df = df.dropna(axis=0, subset=['Year','Publisher'])


# In[11]:


df.isnull().values.any()


# In[12]:


df.Year=df.Year.astype('int64')
df.info()


# In[13]:


len(df)-len(df.drop_duplicates())


# ## Identifying Categorical Variables

# In[14]:


df[['Platform', 'Genre', 'Year' , 'Publisher']].nunique()


# In[15]:


count_of_ones = df['Publisher'].value_counts()[1]
count_of_twos = df['Publisher'].value_counts()[2]
count_of_threes = df['Publisher'].value_counts()[3]
count_of_fours = df['Publisher'].value_counts()[4]
count_of_fives = df['Publisher'].value_counts()[5]

print("Number of occurrences of 1 in column 'Publisher':", count_of_ones)
print("Number of occurrences of 2 in column 'Publisher':", count_of_twos)
print("Number of occurrences of 3 in column 'Publisher':", count_of_threes)
print("Number of occurrences of 4 in column 'Publisher':", count_of_fours)
print("Number of occurrences of 5 in column 'Publisher':", count_of_fives)


# Given that 'Publisher' has such a high number of unique values and with 4,346 of these only publishing 5 games or less, so while it will included in this model, it would be worthwhile to note that Platform and Genre will be better for drawing conclusions

# In[16]:


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16,3))

for xcol, ax in zip(['Platform', 'Genre', 'Year'], axes):
    df.plot(kind='scatter', x=xcol, y='Global_Sales', ax=ax, alpha=0.4, color='b')


# In[17]:


display(df.Platform.unique())
display(df.Genre.unique())
display(df.Year.unique())


# Searching the "2600" showed that this is referring to the "Atari 2600", so for clarity I will rename this to Atari

# ![Atari.png](attachment:Atari.png)

# In[18]:


df['Platform'].replace('2600', 'Atari', inplace=True)
display(df.Platform.unique())


# In[19]:


df["Year"].value_counts()


# In[20]:


#Given the counts of year 2017 and 2020 are very few, they will be dropped
df=df[df["Year"]<2017]


# ## Exploratory Data Analysis with Visualisation

# In[21]:


import random
colors=sns.color_palette("rocket_r", n_colors=10)
# random.shuffle(colors)
palette=['#264653', '#1f766b', '#2e9e95', '#3fe9d4', '#f7c9aa', '#e9c46a', '#dea721', '#cead5c', '#cc6147']
c=["#FF5733", "#007ACC", "#00CC7A", "#FFD733", "#2e9e95", "#FFA500", "#008000", "#f7c9aa", "#FFD700", "#264653"]


# In[22]:


ax=plt.figure(figsize=(10,6))
sns.distplot(df['Year'],color='green')


# 2009 has the highest numberof game releases
# 
# Which genre game has sold the most in a single year?

# In[23]:


year_sale_dx = df.groupby(by=['Year', 'Genre'])['Global_Sales'].sum().reset_index()
year_sale = year_sale_dx.groupby(by=['Year'])['Global_Sales'].transform(max) == year_sale_dx['Global_Sales']
year_sale_max = year_sale_dx[year_sale].reset_index(drop=True)
genre = year_sale_max['Genre']
plt.figure(figsize=(30, 18))
g = sns.barplot(x='Year', y='Global_Sales', data=year_sale_max)
index = 0
for value in year_sale_max['Global_Sales']:
    g.text(index, value + 1, str(genre[index] + '----' +str(round(value, 2))), color='#000', size=14, rotation= 90, ha="center")
    index += 1

plt.xticks(rotation=90)
plt.show()


# 2009 Action is 139.36 million and 2008 Action is 136.39 miliion .

# In[24]:


df['Genre'].value_counts()


# In[25]:


df['Platform'].value_counts()


# In[26]:


# Grouping the data by platform and summing up the global sales
platform_sales = df.groupby('Platform')['Global_Sales'].sum().sort_values(ascending=False)

# Grouping the data by genre and summing up the global sales
genre_sales = df.groupby('Genre')['Global_Sales'].sum().sort_values(ascending=False)

# Creating subplots for platforms and genres
fig, ax = plt.subplots(1, 2, figsize=(18, 6))

# Plotting the sales by platform
sns.barplot(x=platform_sales.values, y=platform_sales.index, ax=ax[0])
ax[0].set_title('Global Video Game Sales by Platform')
ax[0].set_xlabel('Global Sales (in millions)')
ax[0].set_ylabel('Platform')

# Plotting the sales by genre
sns.barplot(x=genre_sales.values, y=genre_sales.index, ax=ax[1])
ax[1].set_title('Global Video Game Sales by Genre')
ax[1].set_xlabel('Global Sales (in millions)')
ax[1].set_ylabel('Genre')

plt.tight_layout()
plt.show()


# In[27]:


genre_counts = df['Genre'].value_counts().nlargest(10)

# Plotting
plt.figure(figsize=(10, 8))
genre_counts.plot(kind='bar', color= palette)
plt.xlabel('Genre')
plt.ylabel('Global_Sales')
plt.title('Top 10 Genres')
plt.xticks(rotation=45)
plt.show()


# In[28]:


platform_counts = df['Platform'].value_counts().nlargest(10)

# Plotting
plt.figure(figsize=(10, 8))
platform_counts.plot(kind='bar', color= palette)
plt.xlabel('Platform')
plt.ylabel('Global_Sales')
plt.title('Top 10 Platforms')
plt.xticks(rotation=45)
plt.show()


# Sales comparison by genre

# In[29]:


comp_genre = df[['Genre', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]
comp_map = comp_genre.groupby(by=['Genre']).sum()
comp_table = comp_map.reset_index()
comp_table = pd.melt(comp_table, id_vars=['Genre'], value_vars=['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'], var_name='Sale_Area', value_name='Sale_Price')
plt.figure(figsize=(15, 10))
sns.barplot(x='Genre', y='Sale_Price', hue='Sale_Area', data=comp_table)


# Total profit by region

# In[30]:


top_sale_reg = df[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]
top_sale_reg = top_sale_reg.sum().reset_index()
top_sale_reg = top_sale_reg.rename(columns={"index": "region", 0: "sale"})
top_sale_reg


# In[31]:


GSales_Year = df.groupby('Year')[['Global_Sales','NA_Sales','EU_Sales','JP_Sales','Other_Sales']].sum()
GSales_Year.plot(figsize = (20,10))


# In[32]:


df.plot(kind = "box" , subplots = True , figsize = (18,18), layout = (3,5))


# ## Identifying Multicollinearity

# In[33]:


corr_matrix = df.corr(numeric_only=True)
corr_matrix


# In[34]:


corr_matrix.describe()


# In[35]:


f,ax = plt.subplots(figsize=(16, 16))
sns.heatmap(corr_matrix.corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax)
plt.show()


# We can see correlation of global sales between NA,EU,JP,other sales are strong, which is to be expected.
# The strongest relation between NA_sales (North American/ USA Sales) and Global sales which indicates NA_sales take a large part of total Global sales

# In[36]:


jp_sales=df['JP_Sales'].sum()
na_sales=df['NA_Sales'].sum()
eu_sales=df['EU_Sales'].sum()
other_sales=df['Other_Sales'].sum()


# In[37]:


labels=['Japan','North America (NA)','European Union (EU)','Other countries']
vales=[jp_sales, na_sales, eu_sales, other_sales]
vales


# In[38]:


import plotly.express as px

fig = px.pie( values=vales, names=labels,  color_discrete_sequence=c, title='Percentage of Global Sales by Region')

fig.show()


# As we assumed from the Heatmap, the USA accounts for the highest percentage of global sales with just under 50% of all sales coming out of that one country 

# In[39]:


#Here,we can see with regression line 
sns.pairplot(corr_matrix,kind="reg")


# ## Dealing with Outliers

# In[40]:


df1=df.head(100)


# In[41]:


trace1 = go.Scatter(
                    x = df1.Global_Sales,
                    y = df1.NA_Sales,
                    mode = "markers",
                    name = "North America",
                    marker = dict(color = 'rgba(28, 149, 249, 0.8)',size=8),
                    text= df.Name)

trace2 = go.Scatter(
                    x = df1.Global_Sales,
                    y = df1.EU_Sales,
                    mode = "markers",
                    name = "Europe",
                    marker = dict(color = 'rgba(249, 94, 28, 0.8)',size=8),
                    text= df1.Name)
trace3 = go.Scatter(
                    x = df1.Global_Sales,
                    y = df1.JP_Sales,
                    mode = "markers",
                    name = "Japan",
                    marker = dict(color = 'rgba(150, 26, 80, 0.8)',size=8),
                    text= df.Name)
trace4 = go.Scatter(
                    x = df1.Global_Sales,
                    y = df1.Other_Sales,
                    mode = "markers",
                    name = "Other",
                    marker = dict(color = 'lime',size=8),
                    text= df.Name)
                    

data = [trace1, trace2,trace3,trace4]
layout = dict(title = 'North America, Europe, Japan and Other Sales of Top 100 Video Games',
              xaxis= dict(title= 'Global_Sales',ticklen= 5,zeroline= False,zerolinewidth=1,gridcolor="white"),
              yaxis= dict(title= 'Sales(In Millions)',ticklen= 5,zeroline= False,zerolinewidth=1,gridcolor="white",),
              paper_bgcolor='rgb(243, 243, 243)',
              plot_bgcolor='rgb(243, 243, 243)' )
fig = dict(data = data, layout = layout)
iplot(fig)


# The above scatter plot indicates that there is one clear outlier with significantly higher sales in all markets. The game is Wii Sports, so this game was not only highest selling Globally but also the highest selling in each individual market by a large margin. 
# 
# ![Wii%20Sports.webp](attachment:Wii%20Sports.webp)
# 
# Given that the purpose of this model is to find out which genres have the highest rate of sales, I think it is important to remove this outlier as it may be skewing the data into showing sports as a more popular genre all over when it is really mainly coming from this one game. Wii Sports was one of the first interactive simluation games that allowed players to feel like they were actually playing the game, this is most likely the reason for it's popularity rather than the fact that it was a sport game. 

# The above bar graphs clearly illustrate the skewness that Wii Sports was having on the data. With that one game included, it was showing that Sports was the second highest genre, however after removing this outlier we can now see 'Sports' dropped to 7th place.

# In[42]:


g = sns.regplot(x='Global_Sales', y='EU_Sales', data=df, ci=None, scatter_kws={"color": "r", "s": 9})
plt.xlim(-2, 85)
plt.ylim(bottom=0)


# In[43]:


if 1 in df.index:
    df = df.drop([1], axis=0)
else:
    print("Row index 1 not found in the DataFrame.")


# In[44]:


g = sns.regplot(x='Global_Sales', y='EU_Sales', data=df, ci=None, scatter_kws={"color": "r", "s": 9})
plt.xlim(-2, 85)
plt.ylim(bottom=0)


# ## Iteration 1 (Baseline Model)
# 
# Now that I am happy with my EDA, I will generate a baseline model. This model will contain all data and no transformations. It will be compared to subsequent iterations to observe the effect of the transformations.
# 
# Our dependent variable is global sales.We will make predictions through different linear regression models.
# 
# I need to convert our categorical variables, Platform, Genre and Publisher, into numerical variables and use them in the regression models.Because Regional sales are just the sub-totals of global sales.

# In[45]:


from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import statsmodels.api as sm 
import statsmodels.formula.api as smf 
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[46]:


from sklearn.preprocessing import LabelEncoder
# label encoding of categorical variables
lbe = LabelEncoder()
df['Genre_Cat'] = lbe.fit_transform(df['Genre'])
df['Platform_Cat'] = lbe.fit_transform(df['Platform'])
df['Publisher_Cat'] = lbe.fit_transform(df['Publisher'])
df.sample(3)


# In[47]:


df1 = df.loc[:,'Global_Sales':]
df1.head()


# ## Model Using the Raw Features 

# In[48]:


outcome = 'Global_Sales'
x_cols = ['Genre_Cat', 'Platform_Cat', 'Publisher_Cat']
predictors = '+'.join(x_cols)
formula = outcome + '~' + predictors
model = ols(formula=formula, data=df).fit()
model.summary()


# ##  Normalisation 

# In[49]:


df2 = df.loc[:,'Global_Sales':]
df2.head()


# In[50]:


# Defining independent and dependent variables and splitting the data into two groups as train and test data
df2 = preprocessing.normalize(df2)
x = df2[:,1:]
y = df2[:,0]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state= 42)


# ##  Statmodels
# ML Regression model with statmodels and model summary

# In[51]:


lm = sm.OLS(y_train, x_train)
model = lm.fit()
model.summary()


# ##  Sci Kit Learn

# In[52]:


# Multilinear Regression model with skilearn 
lm1 = LinearRegression()
model1 = lm1.fit(x_train, y_train)


# In[53]:


# Coefficients
model1.coef_ 


# In[54]:


# Intercept
model1.intercept_


# In[55]:


# R2 score
model1.score(x,y)


# In[56]:


# RMSE score of train data
rmse = np.sqrt(mean_squared_error(y_train, model1.predict(x_train)))
rmse


# In[57]:


# RMSE score of test data
rmse = np.sqrt(mean_squared_error(y_test, model1.predict(x_test)))
rmse


# ##  Cross Validation

# In[58]:


# RMSE average score of train data after cross-validation
np.sqrt(-cross_val_score(model1, 
                x_train, 
                y_train, 
                cv = 10, 
                scoring = "neg_mean_squared_error")).mean()


# In[59]:


# R2 average for differents situation since each time the algorithm selects different %80 as train data 
cross_val_score(model1, x_train, y_train, cv = 10, scoring = "r2").mean()


# In[60]:


# RMSE average score of test data after cross-validation
reg_final_rmse = np.sqrt(-cross_val_score(model1, 
                x_test, 
                y_test, 
                cv = 10, 
                scoring = "neg_mean_squared_error")).mean()
reg_final_rmse


# In[61]:


# R2 average of test data after cross validation
reg_final_r2 = cross_val_score(model1, x_test, y_test, cv = 10, scoring = "r2").mean()
reg_final_r2


# ## Principal Component Analysis

# In[62]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import scale 


# In[63]:


# PCA model instantiation and transformation for PCA
pca = PCA()
x_reduced_train = pca.fit_transform(scale(x_train))


# In[64]:


# PCA components 
x_reduced_train[0:1,:]


# In[65]:


# Cumulative percentage of explained variance as we add each component
np.cumsum(np.round(pca.explained_variance_ratio_, decimals = 4)*100)[0:5]


# In[66]:


# PCA instantiation of regression model
lm2 = LinearRegression()
pcr_model = lm2.fit(x_reduced_train, y_train)


# In[67]:


# PCA model intercept
pcr_model.intercept_


# In[68]:


# PCA model coefficients
pcr_model.coef_


# In[69]:


# PCA Regression model with statmodels
lm3 = sm.OLS(y_train, x_reduced_train)
model2 = lm3.fit()
model2.summary()


# In[70]:


# PCA model prediction
y_pred = pcr_model.predict(x_reduced_train)


# In[71]:


# PCA RMSE score for train data
np.sqrt(mean_squared_error(y_train, y_pred))


# In[72]:


# PCA R2 for train data
r2_score(y_train, y_pred)


# In[73]:


# PCA instantiation of model for test data
pca2 = PCA()
x_reduced_test = pca2.fit_transform(scale(x_test))


# In[74]:


# PCA prediction with test data
y_pred = pcr_model.predict(x_reduced_test)


# In[75]:


# PCA RMSE score for test data
pca_final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
pca_final_rmse


# In[76]:


# PCA R2 for test data
pca_final_r2 = r2_score(y_test, y_pred)
pca_final_r2


# Model Tuning

# In[77]:


from sklearn import model_selection


# In[78]:


# Illustraion of change in RMSE score as we add each component into the model.

cv_10 = model_selection.KFold(n_splits = 10,
                             shuffle = True,
                             random_state = 1)

lm4 = LinearRegression()

RMSE = []

for i in np.arange(1, x_reduced_train.shape[1] + 1):
    
    score = np.sqrt(-1*model_selection.cross_val_score(lm4, 
                                                       x_reduced_train[:,:i], 
                                                       y_train.ravel(), 
                                                       cv=cv_10, 
                                                       scoring='neg_mean_squared_error').mean())
    RMSE.append(score)


# It is clear that the RMSE score decreases as we add all three components into the model. Given this, it makes sense to keep all three components in the model

# In[79]:


plt.plot(RMSE, '-v')
plt.xlabel('Number of Components')
plt.ylabel('RMSE')
plt.title('PCR Model Tuning');


# ## PLS (Partial Least Squares) Regression

# In[80]:


from sklearn.cross_decomposition import PLSRegression, PLSSVD


# In[81]:


# PLS model instantiation
pls_model = PLSRegression().fit(x_train, y_train)


# In[82]:


# PLS model coefficients
pls_model.coef_


# In[83]:


# PLS model predictions based on train data
y_pred = pls_model.predict(x_train)


# In[84]:


# PLS RMSE score for train data
np.sqrt(mean_squared_error(y_train, y_pred))


# In[85]:


# PLS R2 for train data
r2_score(y_train, y_pred)


# In[86]:


# PLS prediction based on test data
y_pred = pls_model.predict(x_test)


# In[87]:


# PLS RMSE test score
np.sqrt(mean_squared_error(y_test, y_pred))


# In[88]:


# PLS R2 for test data
r2_score(y_test, y_pred)


# PLS Cross Validation

# In[89]:


# Illustraion of change in RMSE score as the model adds one additional component to the model in each loop.
cv_10 = model_selection.KFold(n_splits=10, shuffle=True, random_state=1)


RMSE = []

for i in np.arange(1, x_train.shape[1] + 1):
    pls = PLSRegression(n_components=i)
    score = np.sqrt(-1*cross_val_score(pls, x_train, y_train, cv=cv_10, scoring='neg_mean_squared_error').mean())
    RMSE.append(score)

plt.plot(np.arange(1, x_train.shape[1] + 1), np.array(RMSE), '-v', c = "r")
plt.xlabel('Number of Components')
plt.ylabel('RMSE')
plt.title('Components and RMSE');


# In[90]:


# PLS model with two components
pls_model2 = PLSRegression(n_components = 3).fit(x_train, y_train)


# In[91]:


# PLS prediction based on test data after cross validation
y_pred2 = pls_model2.predict(x_test)


# In[92]:


# PLS RMSE test score after cross validation
pls_final_rmse = np.sqrt(mean_squared_error(y_test, y_pred2))
pls_final_rmse


# In[93]:


pls_final_r2 = r2_score(y_test, y_pred2)
pls_final_r2


# In[94]:


print(f"""Multilinear Regression RMSE: {reg_final_rmse}, R2: {reg_final_r2}
PCA Regression RMSE: {pca_final_rmse}, R2: {pca_final_r2}
PLS Regression RMSE: {pls_final_rmse}, R2: {pls_final_r2}""")


# ## Comparison of RMSE and R2 values across different models with normalization
# Multilinear Regression RMSE: 0.015001624864342897, R2: 0.07397442844946144
# 
# PCA Regression RMSE : 0.01674123699336149, R2: 0.09509144293622152
# 
# PLS Regression RMSE : 0.016741898509377524, R2: 0.09501992810961701
# 
# ## Comparison of RMSE and R2 values across different models without normalization
# 
# Multilinear Regression RMSE: 1.7105964805711504, R2: -0.015022400745020936
# 
# PCA Regression RMSE: 2.0668894045476174, R2: -0.00018219613950942737
# 
# PLS Regression RMSE: 2.0646621120360353, R2: 0.0019722471705171385

# ![game%20end-2.jpg](attachment:game%20end-2.jpg)
# 
# ## Conclusion
# The data is not the most desired data since we do not have quantitative independent variables and we have to encode categorical variables. 
# 
# Multilinear Regression has better RMSE test scores and the rest of the approaches have very close RMSE test scores. In this respect, Multilinear Regression model appears to be the best model compared to others.

# In[ ]:




