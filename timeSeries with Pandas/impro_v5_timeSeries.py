
# coding: utf-8

# # Time series Analysis 

# In[86]:


from pandas import Series
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from datetime import datetime
rcParams['figure.figsize'] = 15, 6


# In[80]:


data=pd.read_csv('a40282n.csv',header=None,index_col=None)
print(data.head())
# print(data.dtypes)


# In[81]:


#extract previous 5 columns
data=data[list(range(5))]


# In[82]:


# format the timestamp
rawtime=[]
timeformat=[]
for i in range(len(data)):
    rawtime.append(data[0][i][2:21])
    timeformat.append(pd.to_datetime(rawtime[i],format='%H:%M:%S %d/%m/%Y'))
data[0]=timeformat


# In[83]:


#rename the columns
data.columns=['Time and date','HR','ABPSys','ABPDias','ABPMean']


# In[84]:


data.head()


# In[91]:


ts=data['ABPMean']
# ts[datetime(2016,6,20,10,0,0)]
# plt.plot(data['ABPMean'])


# In[93]:


print(data['Time and date'][0])


# In[96]:


data['Time and date']=pd.to_datetime(data['Time and date'],infer_datetime_format=True)
indexedDataset=data.set_index(['Time and date'])


# In[97]:


indexedDataset.head()


# In[115]:


plt.xlabel('Date')
plt.ylabel('test')
plt.grid(True)
plt.plot(indexedDataset)


# In[118]:


ABPm=indexedDataset[['ABPMean']]
plt.plot(ABPm)


# In[121]:


#Determing rolling statistics
#30 minutes
rolmean=ABPm.rolling(window=30).mean()
rolstd=ABPm.rolling(window=30).std()
print(rolmean,rolstd)


# In[122]:


#plot rolling statistics
orig=plt.plot(ABPm,color='b',label='Original')
mean=plt.plot(rolmean,color='red',label='Rolling Mean')
std=plt.plot(rolstd,c='k',label='Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.show(block=False)


# ## Stationarity
# the standard deviation is not constant, so our data is not stationary
# 只用肉眼是分不清是否真的平稳的，因此，我们有必要引入数学方法对平稳进行形式化的检验。
# 
# dickey-fuller test是用来检验序列是否平稳的方式，可以测试一个自回归模型是否存在单位根（unit root）
# 单位根检验是指检验序列中是否存在单位根，因为存在单位根就是非平稳时间序列了。
# 单位根就是指单位根过程，可以证明，序列中存在单位根过程就不平稳，会使回归分析中存在伪回归。
# 
# 而迪基-福勒检验（Dickey-Fuller test）和扩展迪基-福勒检验（Augmented Dickey-Fuller test可以测试一个自回归模型是否存在单位根（unit root）。
# http://www.pengfoo.com/post/machine-learning/2017-01-24

# In[125]:


#Perform Dickey-Fuller test
from statsmodels.tsa.stattools import adfuller
print('Results of Dickey-Fuller Test:')
dftest=adfuller(ABPm['ABPMean'],autolag='AIC')

dfoutput=pd.Series(dftest[0:4],index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for k,v in dftest[4].items():
    dfoutput['Critical Value (%s)'%k]=v
    
print(dfoutput)


# ## Note
# p-value should be always small, here is close to 0, it's good
# 
# Critical Value should be always more than Test Statistic, it's good shows that it can reject the hypothesis and we can say the data is stationary 

# In[126]:


#Estimating trend
ABPm_logScale=np.log(ABPm)
plt.plot(ABPm_logScale)


# In[127]:


movingAverage=ABPm_logScale.rolling(window=12).mean()
movingSTD=ABPm_logScale.rolling(window=12).mean()
plt.plot(ABPm_logScale)
plt.plot(movingAverage,c='red')


# ## Note
# ### some standard ways to make a time series stationary
# make it stationary like take log, take a square, cube roots all depends on data what it holds so here we're going to log scale 

# In[130]:


datasetLogScaleMinusMovingAverage=ABPm_logScale-movingAverage
datasetLogScaleMinusMovingAverage.head(12)

#remove Nan Values
datasetLogScaleMinusMovingAverage.dropna(inplace=True)
datasetLogScaleMinusMovingAverage.head(10)


# In[134]:


from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    #Determing rolling statistics
    movingAverage=timeseries.rolling(window=12).mean()
    movingSTD=timeseries.rolling(window=12).std()
    
    #plot rolling statistics
    orig=plt.plot(timeseries,color='b',label='Original')
    mean=plt.plot(movingAverage,c='red',label='Rolling Mean')
    std=plt.plot(movingSTD,c='k',label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test
    from statsmodels.tsa.stattools import adfuller
    print('Results of Dickey-Fuller Test:')
    dftest=adfuller(timeseries['ABPMean'],autolag='AIC')

    dfoutput=pd.Series(dftest[0:4],index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for k,v in dftest[4].items():
        dfoutput['Critical Value (%s)'%k]=v

    print(dfoutput)


# In[135]:


test_stationarity(datasetLogScaleMinusMovingAverage)


# In[136]:


#weighted mean
exponentialDecayWeightedAverage=ABPm_logScale.ewm(halflife=12,min_periods=0,adjust=True).mean()
plt.plot(ABPm_logScale)
plt.plot(exponentialDecayWeightedAverage)


# In[137]:


#substract weighted mean instead of simple mean
#and then check for stationarity
datasetLogScaleMinusMovingExponentialDecayAverage=ABPm_logScale-exponentialDecayWeightedAverage
test_stationarity(datasetLogScaleMinusMovingExponentialDecayAverage)


# ## Note
# the std is quite flat and in fact you can also say that this doesn't have any trend
# 
# use the function called a shift to shift all of these values
# below we take a lag滞后 of 1 so here we just shift the values by 1
# 
# ### ARIMA model
# ARIMA model have three models in it:
# * AR model: stand for auto regressive
# * MA model: for moving average
# * IS model: for integration
# 
# so Arima model basically takes three parameters

# In[138]:


#shift the value into series so that we can use it in the forecasting
datasetLogDiffShifting=ABPm_logScale-ABPm_logScale.shift()
plt.plot(datasetLogDiffShifting)


# In[139]:


datasetLogDiffShifting.dropna(inplace=True)
test_stationarity(datasetLogDiffShifting)


# In[148]:


from statsmodels.tsa.seasonal import seasonal_decompose
# sm.tsa.seasonal_decompose(ABPm_logScale.G.values, freq=1440)
#freq is calculated based on the time window, which is 10 mins, 
#so the freqency actually is half a day. freq=6*12 
#frep=6*10 ten hours???
decomposition = seasonal_decompose(ABPm_logScale,freq=60)
trend=decomposition.trend
seasonal=decomposition.seasonal
residual=decomposition.resid

plt.subplot(411)
plt.plot(ABPm_logScale,label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend,label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual,label='Residuals')
plt.legend(loc='best')
plt.tight_layout()


# ## Note
# residuals is nothing, the irregularities that is present in your data
# 
# so they don't have any shape any size and it cannot find out what is going to happen next
# it's quite regular in nature
# 
# so now we are going to check the noise if it's stationary or not

# In[149]:


decomposeLogData=residual
decomposeLogData.dropna(inplace=True)
test_stationarity(decomposeLogData)


# In[150]:


#determine how to calculate p-value and Q-value(left graph) using acf 
#according to the graph we need to check that what is the value where the graph cuts off 
#or you can set drops to zero for the first time 

'''
it touches the confidence level over here so here if you see the p-values (right graph) 
almost around 1 or 2（下降最陡的那个点）
左边那个图用来计算Q value，从左边看，it cut 在和第一条线的第一个交点，
drops to zero在跟0相交的那个点，然后Q value是第三条线的交点，也在1附近

'''

#ACF and PACF plots
from statsmodels.tsa.stattools import acf,pacf
lag_acf=acf(datasetLogDiffShifting,nlags=20)
lag_pacf=pacf(datasetLogDiffShifting,nlags=20,method='ols')

#Plot ACF
plt.subplot(211)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',c='gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',c='gray')
plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',c='gray')
plt.title('Autocorrelation Function')

#plot PACF
plt.subplot(212)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',c='gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',c='gray')
plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',c='gray')
plt.title('Partial Autocorrelation Function')

plt.tight_layout()


# ## Note
# we can simply substitute these values in the ARIMA model 
# 
# with respect to AR that is your auto regressive part

# In[188]:


from statsmodels.tsa.arima_model import ARIMA
# AR Model
#(2,1,2)means P value is 2, differenced it 1, so D value becomes 1, Q value is again 2
#(1,1,0)(0,1,1)
model= ARIMA(ABPm_logScale,order=(2,1,0))
results_AR=model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_AR.fittedvalues,color='red')
#RSS is the residual sum of squares
# the greater the RSS the bad is for you
plt.title('RSS: %.4f'%sum((results_AR.fittedvalues-datasetLogDiffShifting['ABPMean'])**2))
print('Plotting AR model')


# In[187]:


#MA model
model=ARIMA(ABPm_logScale,order=(0,1,2))
results_MA=model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_MA.fittedvalues,color='red')
plt.title('RSS: %.4f'%sum((results_MA.fittedvalues-datasetLogDiffShifting['ABPMean'])**2))
print('Plotting MA model')


# ## Note
# here we conclude that with respect to auto regressive part, we have the RSS as 1.5867
# with respect to moving average we have the RSS 1.4886 and if we combine both of them and make a ARIMA out of it that is (2,1,2)

# In[189]:


model=ARIMA(ABPm_logScale,order=(2,1,2))
results_ARIMA=model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_ARIMA.fittedvalues,color='red')
plt.title('RSS: %.4f'%sum((results_ARIMA.fittedvalues-datasetLogDiffShifting['ABPMean'])**2))
print('Plotting ARIMA model')


# In[190]:


#do some fitting on the time series on what data we have
predictions_ARIMA_diff=pd.Series(results_ARIMA.fittedvalues,copy=True)
print(predictions_ARIMA_diff.head())


# In[191]:


#convert to cumulative sum
predictions_ARIMA_diff_cumsum=predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum.head())


# In[201]:


# finally we are going to have the predictions done for the fitted values
predictions_ARIMA_log=pd.Series(ABPm_logScale['ABPMean'].iloc[0],index=ABPm_logScale.index)
predictions_ARIMA_log=predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()


# In[202]:


#come back to the original form
# plot the actual values to how our model has fited
predictions_ARIMA=np.exp(predictions_ARIMA_log)
plt.plot(ABPm)
plt.plot(predictions_ARIMA,c='r')


# In[203]:


ABPm_logScale


# In[210]:


#to predict next 60 points
results_ARIMA.plot_predict(1,660)


# ## Note
# the blue is the forecasted value and this gray part is the confidence level so now whatever happens or however you do the foreasting this value will not exceed the confidence level so it means for the next 1 hour you have the prediction somewhat like this

# In[211]:


x=results_ARIMA.forecast(steps=60)


# In[206]:


x[1]


# In[207]:


len(x[1])


# In[208]:


np.exp(x[1])

