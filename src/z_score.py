import numpy as np
import pandas as pd
import math
from scipy.stats import norm

## calculate zscore
def func_zscore(df,var_name,l_name,m_name,s_name,newvar_name1,newvar_name2,newvar_name3):
    df.ix[(df[var_name] > 0) & (abs(df[l_name]) >= 0.01),newvar_name1] = ((df[var_name] / df[m_name]) ** df[l_name] -1)/(df[l_name] * df[s_name])
    df.ix[(df[var_name] > 0) & (abs(df[l_name]) < 0.01),newvar_name1] = np.log(df[var_name] / df[m_name])/df[s_name]
    df[newvar_name2] = norm.cdf(df[newvar_name1]) * 100
    df['sdl'] = ((df[m_name] - df[m_name] * (1-2 * df[l_name] * df[s_name]) ** (1 / df[l_name])) / 2)
    df['sdh'] = ((df[m_name] * (1 + 2 * df[l_name] * df[s_name]) ** (1 / df[l_name]) - df[m_name]) / 2)
    df[newvar_name3] = (df[var_name] - df[m_name]) / df['sdl']
    df.ix[(df[var_name] > 0) & (df[var_name] < df[m_name]),newvar_name3] = (df[var_name] - df[m_name]) / df['sdl']
    df.ix[(df[var_name] > 0) & (df[var_name] >= df[m_name]),newvar_name3] = (df[var_name] - df[m_name]) / df['sdh']

    return df

if __name__=='__main__':
    zscore()
