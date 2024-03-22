!pip install retry --q
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
tqdm.pandas()
from retry import retry #https://pypi.org/project/retry/
#from tenacity import wait_exponential, retry, stop_after_attempt
#@retry(wait=wait_exponential(multiplier=2, min=2, max=30),  stop=stop_after_attempt(5))

def calc_col_weighted_avg(measure_v, measure_c):
  #print(len(measure_v))
  #print(type(measure_v))
  #print(measure_v)
  try:
         sum_product = sum(measure_v * measure_c)
         w_avg = sum_product / sum(measure_c)
  except Exception as error:
         print("An exception occurred:", type(error).__name__, "–", error)
  return w_avg*100

@retry(ZeroDivisionError, tries=3, delay=2)
def df_summary(df):
  df_atr = []
  df_wavg = []
  for col in df:
    if "_score" in col:
      #print(col)
      records = col.replace(col[len(col) - 5:], "records")
      df_atr.append(col)
      weighted_avg = float(format(calc_col_weighted_avg(df[col], df[records]),".2f"))
      #print(type(weighted_avg))
      df_wavg.append(weighted_avg)
      #print(f"Weighted average for {col}: {'%.2f' % weighted_avg}")
  retruned_df = pd.DataFrame(zip(df_atr,df_wavg), columns= ["variable name","Weighted avg"])
  dict_param_values = {df_atr[i]: df_wavg[i] for i in range(len(df_atr))}
  return retruned_df, dict_param_values

def calc_overall_weighted_avg(dict_param_weights,dict_param_values):
  return sum(dict_param_weights[k]*dict_param_values[k] for k in dict_param_weights)

@retry(tries=3)
def simulate_adge(df,adge_n,param_n,score_v):
  df_simulated =  df.copy(deep=True)
  #print(df_simulated[""+param_n+""])
  #df_simulated[""+param_n+""] = 0
  #print(df_simulated)
  try:
    #df_simulated[""+param_n+""] = np.where(df_simulated['ADGE']==adge_n,score_v,df_simulated[""+param_n+""])
    df_simulated[param_n] = np.where(df_simulated['ADGE']==adge_n,score_v,df_simulated[param_n])
  except:
     print(param_n + " doesn't exist")
  param_n_records = param_n.replace(param_n[len(param_n) - 5:], "records")
  #print(param_n_test)
  return df_simulated,format(calc_col_weighted_avg(df_simulated[param_n],df_simulated[param_n_records]),".2f")
