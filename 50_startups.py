import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#loading the data

startup = pd.read_csv('C:\\Users\\nidhchoudhary\\Desktop\\Assignment\\MULTI_LINEAR_REGRESSION\\50_startups.csv')

print(startup.head())

##File Contains non.numerical data so dropping the entire column of state


startup_new = pd.get_dummies(startup,columns= ['State'],drop_first=True)

#comp_new=pd.get_dummies(startup,columns=['cd','multi','premium'],drop_first=True)


print(startup_new.head())

##find correlation

print(startup_new.corr())

# ##correlation between R&D and marketing spend is 0.72 which is quite high
startup_new.columns = [c.replace(' ', '_') for c in startup_new.columns]
import statsmodels.formula.api as smf
model1 = smf.ols('Profit~Spend+Administration+Marketing_Spend+State_Florida+State_New_York',data= startup_new).fit()

#print(model1.summary())

### Pvalue  for  Marketing spend State_Florida and State_New_York  is high
# ##r2 is 0.951

import seaborn as sns
sns.pairplot(startup_new)
#plt.show()

# ##Check if there any data of which influencing the other data

import statsmodels.api as sm
sm.graphics.influence_plot(model1)
#plt.show()

# ##Observed that few of the records has r&D expense very low or nil but have some profit which is not an ideal scenario
# ##So dropping those records

model2_new = startup_new.drop(startup_new.index[[45,46,47,48,49]],axis=0)

# ##Check the P Values after dropping some records

model2_new_sls = smf.ols('Profit~Spend+Administration+Marketing_Spend+State_Florida+State_New_York',data= model2_new).fit()

print(model2_new_sls.summary())


# ##Preparing Model between Profit and Administration spend

model_Admin = smf.ols('Profit~Administration',data = model2_new ).fit()

#print(model_Admin.summary())

# ##P Value 0.368>0.05 insignificant

# ##Preparing Model between profit Administration

model_Spend = smf.ols('Profit~Marketing_Spend',data = model2_new ).fit()
#print(model_Spend.summary())

# ## P value is 0.00< 0.05  significant



model_State_Florida = smf.ols('Profit~State_Florida',data = model2_new ).fit()

#print(model_State_Florida.summary())

# ##P Value id 0.626>0.05 insignificant


model_State_New_York = smf.ols('Profit~State_New_York',data = model2_new ).fit()

#print(model_State_New_York.summary())


# ##P Value id 0.745>0.05 insignificant


# # calculating VIF's values of independent variables

r_spend = smf.ols('Spend~Administration+Marketing_Spend+State_Florida+State_New_York',data = model2_new).fit().rsquared
vif_spend = 1/(1-r_spend)
#print(vif_spend)###2.38

r_administartion = smf.ols('Administration~Spend+Marketing_Spend+State_Florida+State_New_York',data =model2_new).fit().rsquared
vif_administartion = 1/(1-r_administartion)
#print(vif_administartion)###1.24

r_Marketing_Spend = smf.ols('Marketing_Spend~Spend+Administration+State_Florida+State_New_York',data =model2_new).fit().rsquared
vif_Marketing_spend = 1/(1-r_Marketing_Spend)

#print(vif_Marketing_spend)#####2.337



r_state_Florida = smf.ols('State_Florida~Spend+Administration+Marketing_Spend+State_New_York',data =model2_new).fit().rsquared
vif_state_Florida = 1/(1-r_state_Florida)

##print(r_state_Florida)##2.68


r_state_New_York = smf.ols('State_New_York~Spend+Administration+Marketing_Spend+State_Florida',data =model2_new).fit().rsquared
vif_state_New_York = 1/(1-r_state_New_York)
print(vif_state_New_York)###1.365

# ## SInce Vif for administraion is quite so wull ignore this column and will predict the final mdoel
final_model = smf.ols('Profit~Spend+Marketing_Spend+State_Florida+State_New_York',data= model2_new).fit()

print(model2_new_sls.summary())

final_predict = final_model.predict(model2_new)
print(final_predict)






