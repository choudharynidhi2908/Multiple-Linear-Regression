import pandas as pd
import numpy as np 
import matplotlib.pyplot as py

comp = pd.read_csv('C:/Users/nidhchoudhary/Desktop/Assignment/MULTI_LINEAR_REGRESSION/computer_Data.csv')
print('Without Drop')
print(comp.head())

###Drop all the index columns as they can't be read as multilinear regression and create dummy for ctegorcial data

comp_new=pd.get_dummies(comp,columns=['cd','multi','premium'],drop_first=True)

#comp_new = comp.drop(comp.columns[[0,6,7,8]],axis= 1)

#print(comp_new.head())

#print(comp_new.corr())

##Correlation between HD and RAM is high 0.777

import statsmodels.formula.api as smf


model1 = smf.ols('price~speed+hd+ram+screen+ads+trend+cd_yes+multi_yes+premium_yes',data=comp_new).fit()###R2 = 0.776

#print(model1.summary())

##Check if there is any influential data

import statsmodels.api as sm
sm.graphics.influence_plot(model1)
py.show()## drop 5959 and 1699 index 

comp_new1 = comp_new.drop(comp_new.index[[5959,1699,3782,4476,1100,899]],axis = 0)


model2 = smf.ols('price~speed+hd+ram+screen+ads+trend+cd_yes+multi_yes+premium_yes',data=comp_new1).fit()##0.775 
#print(model2.summary())

##Exponential model

model3 = smf.ols('price~np.log(speed+hd+ram+screen+ads+trend+cd_yes+multi_yes+premium_yes)',data=comp_new1).fit()##0.210 
print(model3.summary())


###Exponential Model has high r2 number

model4 = smf.ols('np.log(price)~speed+hd+ram+screen+ads+trend+cd_yes+multi_yes+premium_yes',data=comp_new1).fit()###0.783

print(model4.summary())

##Quardtic Model

comp_new1["xsquare"] =  (comp_new1.speed+comp_new1.hd+comp_new1.ram+comp_new1.screen+comp_new1.ads+comp_new1.trend+comp_new1.cd_yes
	+comp_new1.multi_yes+comp_new1.premium_yes)
#print('xsquare',xsquare)

model_quad =  smf.ols('price~xsquare+speed+hd+ram+screen+ads+trend+cd_yes+multi_yes+premium_yes',data=comp_new1).fit()
#print(model_quad.summary())


pred = model_quad.predict(comp_new1)

print(pred)

##Plot the final model

sm.graphics.plot_partregress_grid(model_quad)
py.show()
