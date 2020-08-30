# Multilinear Regression
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt



# loading the data
Toyotta = pd.read_csv("C:\\Users\\nidhchoudhary\\Desktop\\Assignment\\MULTI_LINEAR_REGRESSION\\ToyotaCorolla.csv",encoding='ANSI')
##Creeating Data Set
Toyotta_Cars = Toyotta[["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]


##Correlation values


#print(Toyotta_Cars.corr())
import seaborn as sns
sns.pairplot(Toyotta_Cars)

#plt.show()

import statsmodels.formula.api as smf 

model1 = smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data= Toyotta_Cars).fit()##0.864
print(model1.params)

print(model1.summary())
##Calculate RMSE Value

predicted_value = model1.predict(Toyotta_Cars)
Error = Toyotta_Cars.Price - predicted_value ###1338.25
#print((np.sqrt(np.mean(Error*Error)),'Rmse Values'))

import statsmodels.api as sm
sm.graphics.influence_plot(model1)
#plt.show()

##From Influence Plot we saw 80 has highest radius so dropping index 80 row

Toyotta_Cars_new  = Toyotta_Cars.drop(Toyotta_Cars.index[80],axis=0)

model2 = smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data= Toyotta_Cars_new).fit()##0.864
##print(model2.summary())


final_predict = model2.predict(Toyotta_Cars_new)

#print(final_predict)




##Third model 


model3 = smf.ols('Price~np.log(Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight)',data= Toyotta_Cars_new).fit()##0.4375
print(model3.summary())






model4= smf.ols('np.log(Price)~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data= Toyotta_Cars_new).fit()##0.869
print(model2.summary())



final_predict = model4.predict(Toyotta_Cars_new)


