import pandas as pd
import numpy as np
from tqdm.auto import tqdm
tqdm.pandas()
from retry import retry #https://pypi.org/project/retry/
#from tenacity import wait_exponential, retry, stop_after_attempt
#@retry(wait=wait_exponential(multiplier=2, min=2, max=30),  stop=stop_after_attempt(5))

def calc_col_weighted_avg(measure_v, measure_c):
  """
  Calculates the param_n weighted average

  Paramters:
    measure_v: A series of the measure values
    measure_c: A series of the measure # of records

  Returnes:
    float: the calculated weighted average of the column

  """
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
  """
    Summarize the DataFrame by calculating weighted averages for columns containing '_score'.

    Parameters:
        df (DataFrame): The DataFrame to be summarized.

    Returns:
        dict: A dictionary containing column names as keys and their corresponding weighted averages as values.
  """

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
  #retruned_df = pd.DataFrame(zip(df_atr,df_wavg), columns= ["variable name","Weighted avg"])
  dict_param_values = {df_atr[i]: df_wavg[i] for i in range(len(df_atr))}
  return dict_param_values #,retruned_df

def calc_overall_weighted_avg(dict_param_weights,dict_param_values):
  """
  Calculate the overall index weighted average

  Parameters:
    dict_param_weights: A set of predefined weights for each param_n, in a dictionary format
    dict_param_values: A set of calculated weighted average for each parameter generated from df_summary() function

  Returns:
    flot: the overall TAMM Quality Index weighted average

  """
  return sum(dict_param_weights[k]*dict_param_values[k] for k in dict_param_weights)

@retry(tries=3)
def simulate_adge(df, adge_n, param_n, score_v):
    """
    Simulate changes based on a given condition.

    Parameters:
        df (DataFrame): The DataFrame containing the data.
        adge_n (str): The name of 'ADGE' to update the selected parameter.
        param_n (str): The name of the column to update based on the new score_v.
        score_v (int or float): The value to assign to 'param_n' for the selected adge.

    Returns:
        DataFrame: A copy of the original DataFrame with simulated changes.
        str: The weighted average of the 'param_n' column in the DataFrame, formatted to two decimal places.
    """

    df_simulated = df.copy(deep=True)
    # print(df_simulated[""+param_n+""])
    # df_simulated[""+param_n+""] = 0
    try:
        # df_simulated[""+param_n+""] = np.where(df_simulated['ADGE']==adge_n,score_v,df_simulated[""+param_n+""])
        df_simulated[param_n] = np.where(
            df_simulated["ADGE"] == adge_n, score_v, df_simulated[param_n]
        )
    except:
        print(param_n + " doesn't exist")
    param_n_records = param_n.replace(param_n[len(param_n) - 5 :], "records")
    # print(param_n_test)
    updated_weighted_avg = format(
        calc_col_weighted_avg(df_simulated[param_n], df_simulated[param_n_records]),
        ".2f")
    return df_simulated, updated_weighted_avg
    )
