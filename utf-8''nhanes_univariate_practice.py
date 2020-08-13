
# coding: utf-8

# # Practice notebook for univariate analysis using NHANES data
# 
# This notebook will give you the opportunity to perform some univariate analyses on your own using the NHANES.  These analyses are similar to what was done in the week 2 NHANES case study notebook.
# 
# You can enter your code into the cells that say "enter your code here", and you can type responses to the questions into the cells that say "Type Markdown and Latex".
# 
# Note that most of the code that you will need to write below is very similar to code that appears in the case study notebook.  You will need to edit code from that notebook in small ways to adapt it to the prompts below.
# 
# To get started, we will use the same module imports and read the data in the same way as we did in the case study:

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
import numpy as np

da = pd.read_csv("nhanes_2015_2016.csv")
da.describe()


# ## Question 1
# 
# Relabel the marital status variable [DMDMARTL](https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/DEMO_I.htm#DMDMARTL) to have brief but informative character labels.  Then construct a frequency table of these values for all people, then for women only, and for men only.  Then construct these three frequency tables using only people whose age is between 30 and 40.

# In[2]:


da['DMDMARTL2'] = da['DMDMARTL'].replace({1:'Married', 2:'Widowed',3:'Divorced',4:'Separated',
                                         5:'Never married',6:'Living with partner',77:'Refused',99:'dont know'})
da['DMDMARTL2'].value_counts()

da["RIAGENDRx"] = da.RIAGENDR.replace({1: "Male", 2: "Female"})

a = da.groupby(da["RIAGENDRx"])['DMDMARTL2'].value_counts()

da["agegrp"] = pd.cut(da.RIDAGEYR, [30,40])

b = da.groupby([da["RIAGENDRx"],da['agegrp']])['DMDMARTL2'].value_counts()

b.loc['Male',:].unstack()


# __Q1a.__ Briefly comment on some of the differences that you observe between the distribution of marital status between women and men, for people of all ages.

# __Q1b.__ Briefly comment on the differences that you observe between the distribution of marital status states for women between the overall population, and for women between the ages of 30 and 40.

# __Q1c.__ Repeat part b for the men.

# ## Question 2
# 
# Restricting to the female population, stratify the subjects into age bands no wider than ten years, and construct the distribution of marital status within each age band.  Within each age band, present the distribution in terms of proportions that must sum to 1.

# In[3]:


da['agegrp'] = pd.cut(da.RIDAGEYR, [20, 30, 40, 50, 60, 70, 80])
da["RIAGENDRx"] = da.RIAGENDR.replace({1: "Male", 2: "Female"})

da = da[da.RIAGENDRx == 'Female']

# eliminate rare/missing values
dx = da.loc[~da.DMDMARTL2.isin(['dont know','Refused']),:]

dx = dx.groupby('agegrp')["RIAGENDRx"].value_counts()
dx = dx.unstack()
dx = dx.apply(lambda x: x /x.sum())
print(dx.to_string(float_format="%.3f"))


# __Q2a.__ Comment on the trends that you see in this series of marginal distributions.

# __Q2b.__ Repeat the construction for males.

# In[4]:


# insert your code here


# __Q2c.__ Comment on any notable differences that you see when comparing these results for females and for males.

# ## Question 3
# 
# Construct a histogram of the distribution of heights using the BMXHT variable in the NHANES sample.

# In[5]:


da = pd.read_csv("nhanes_2015_2016.csv")
da['agegrp'] = pd.cut(da.RIDAGEYR, [20, 30, 40, 50, 60, 70, 80])
da["RIAGENDRx"] = da.RIAGENDR.replace({1: "Male", 2: "Female"})
sns.distplot(da.BMXHT.dropna(),bins = 50,kde = False)


# __Q3a.__ Use the `bins` argument to [distplot](https://seaborn.pydata.org/generated/seaborn.distplot.html) to produce histograms with different numbers of bins.  Assess whether the default value for this argument gives a meaningful result, and comment on what happens as the number of bins grows excessively large or excessively small. 

# __Q3b.__ Make separate histograms for the heights of women and men, then make a side-by-side boxplot showing the heights of women and men.

# In[6]:


da = pd.read_csv("nhanes_2015_2016.csv")
da['agegrp'] = pd.cut(da.RIDAGEYR, [20, 30, 40, 50, 60, 70, 80])
da["RIAGENDRx"] = da.RIAGENDR.replace({1: "Male", 2: "Female"})

fe = da[da.RIAGENDRx == 'Female']
# fe = da.loc[da.RIAGENDRx.isin(['Female'],:)]
mm = da[da.RIAGENDRx == 'Male']

fig1 = plt.figure()
sns.distplot(fe.BMXHT.dropna())
fig2 = plt.figure()
sns.distplot(mm.BMXHT.dropna())

fig3 = plt.figure(figsize = (12,5))
sns.boxplot(x = 'RIAGENDRx', y = 'BMXHT',data = da )


# __Q3c.__ Comment on what features, if any are not represented clearly in the boxplots, and what features, if any, are easier to see in the boxplots than in the histograms.

# ## Question 4
# 
# Make a boxplot showing the distribution of within-subject differences between the first and second systolic blood pressure measurents ([BPXSY1](https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/BPX_I.htm#BPXSY1) and [BPXSY2](https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/BPX_I.htm#BPXSY2)).

# In[19]:


da = pd.read_csv("nhanes_2015_2016.csv")
da['agegrp'] = pd.cut(da.RIDAGEYR, [20, 30, 40, 50, 60, 70, 80])
da["RIAGENDRx"] = da.RIAGENDR.replace({1: "Male", 2: "Female"})
di = da.BPXSY1 - da.BPXSY2
_ = sns.boxplot(di)
_.set_title('Difference between first and second SBP')
_.set_xlabel('Differnce')


# __Q4a.__ What proportion of the subjects have a lower SBP on the second reading compared to the first?

# In[31]:


a = (da.BPXSY1) - (da.BPXSY2)
b = (da.BPXSY1 - da.BPXSY2).dropna()

print(np.mean(a > 0))
print(np.mean(b > 0))


# __Q4b.__ Make side-by-side boxplots of the two systolic blood pressure variables.

# In[23]:


_ = sns.boxplot(data = da.loc[:,['BPXSY1','BPXSY2']]).set_title('Two systolic blood pressure variables')


# __Q4c.__ Comment on the variation within either the first or second systolic blood pressure measurements, and the variation in the within-subject differences between the first and second systolic blood pressure measurements.

# ## Question 5
# 
# Construct a frequency table of household sizes for people within each educational attainment category (the relevant variable is [DMDEDUC2](https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/DEMO_I.htm#DMDEDUC2)).  Convert the frequencies to proportions.

# In[17]:


dx = da.groupby('DMDEDUC2')['DMDHHSIZ'].value_counts().unstack()
dx = dx.apply(lambda x: x / x.sum(),axis =1 )
print(dx.to_string(float_format = '%.3f'))


# __Q5a.__ Comment on any major differences among the distributions.

# __Q5b.__ Restrict the sample to people between 30 and 40 years of age.  Then calculate the median household size for women and men within each level of educational attainment.

# In[20]:


da["RIAGENDRx"] = da.RIAGENDR.replace({1: "Male", 2: "Female"})
da["agegrp"] = pd.cut(da.RIDAGEYR, [30,40])
da.groupby(['RIAGENDRx','agegrp'])['DMDHHSIZ'].median()


# ## Question 6
# 
# The participants can be clustered into "maked variance units" (MVU) based on every combination of the variables [SDMVSTRA](https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/DEMO_I.htm#SDMVSTRA) and [SDMVPSU](https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/DEMO_I.htm#SDMVPSU).  Calculate the mean age ([RIDAGEYR](https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/DEMO_I.htm#RIDAGEYR)), height ([BMXHT](https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/BMX_I.htm#BMXHT)), and BMI ([BMXBMI](https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/BMX_I.htm#BMXBMI)) for each gender ([RIAGENDR](https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/DEMO_I.htm#RIAGENDR)), within each MVU, and report the ratio between the largest and smallest mean (e.g. for height) across the MVUs.

# In[42]:


da = pd.read_csv("nhanes_2015_2016.csv")
da["RIAGENDRx"] = da.RIAGENDR.replace({1: "Male", 2: "Female"})
a = da.groupby(['SDMVSTRA', 'SDMVPSU','RIAGENDRx'])['RIDAGEYR'].mean().unstack()
b = da.groupby(['SDMVSTRA', 'SDMVPSU','RIAGENDRx'])['BMXHT'].mean().unstack()
c = da.groupby(['SDMVSTRA', 'SDMVPSU','RIAGENDRx'])['BMXBMI'].mean().unstack()
ratio_a = a.max() / a.min()
ratio_b = b.max() / b.min()
ratio_c = c.max() / c.min()

print('RatioA:', ratio_a)
print('RatioB:', ratio_b)
print('RatioC:', ratio_c)


# __Q6a.__ Comment on the extent to which mean age, height, and BMI vary among the MVUs.

# __Q6b.__ Calculate the inter-quartile range (IQR) for age, height, and BMI for each gender and each MVU.  Report the ratio between the largest and smalles IQR across the MVUs.

# In[52]:


mm = da[da["RIAGENDRx"] == 'Male']
ff = da[da["RIAGENDRx"] == 'Female']

iqr_age_m_grp = mm.groupby(['SDMVSTRA', 'SDMVPSU'])['RIDAGEYR']
iqr_age_m = (iqr_age_m_grp.quantile(0.75) -  iqr_age_m_grp.quantile(0.25)).max()
iqr_age_f = (iqr_age_m_grp.quantile(0.75) -  iqr_age_m_grp.quantile(0.25)).min()
#iqr_age_m_grp
iqr_age_m - iqr_age_f


# __Q6c.__ Comment on the extent to which the IQR for age, height, and BMI vary among the MVUs.
