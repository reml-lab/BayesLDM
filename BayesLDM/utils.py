import numpy as np
import pandas as df
import collections

def get_imputed_df(model, old_df, sample_indices=[], method=None, rescale_df_info=None):
  # select sample_indices for imputation, or set method to "mean" or "mode"
  # if mean or mode is used, then get_imputed_df() returns imputed_df,
  # otherwise, get_imputed_df() returns imputed_dfs_dict, with the sample_indices as keys
  posterior_samples = post_process_samples(model, model.unprocessed_samples)
  imputed_dfs_dict = {}
  if len(sample_indices) > 0:
    check_range = len(sample_indices)
  else:
    check_range = 1
  for i in range(check_range):     
    imputed_df = old_df.copy()    
    for var_name in imputed_df.columns:
      if rescale_df_info is not None:
        df_mean = rescale_df_info['df_mean']
        df_sd   = rescale_df_info['df_sd']
        k = var_name        
        imputed_df[var_name] = np.array(imputed_df[var_name]) * df_sd[k] + df_mean[k]
        for name in posterior_samples.keys():
          if name.find(k) >= 0:
            posterior_samples[name] = np.array(posterior_samples[name]) * df_sd[k] + df_mean[k]   
      found_posterior_nan_indices = []
      if var_name in list(posterior_samples.keys()):
        if len(sample_indices) > 0:
          sample_index = sample_indices[i]           
          imputed_samples = posterior_samples[var_name][:,sample_index]
          imputed_dfs_dict[sample_index] = imputed_df
        elif method == "mode":
          last_dim_index = len(posterior_samples[var_name].shape)-1
          imputed_samples = np.array(stats.mode(posterior_samples[var_name], axis=last_dim_index)[0].flatten())
        else:
          imputed_samples = posterior_samples[var_name].mean(axis=-1)            
        data_nan_indices=list(np.argwhere(np.isnan(imputed_df[var_name]).values).flatten())
        #found_posterior_nan_indices=list(np.argwhere(~np.isnan(imputed_samples)).flatten())
        #assert(found_posterior_nan_indices == data_nan_indices), "nan indices in df and samples do not match"
        imputed_df[var_name].values[data_nan_indices] = imputed_samples[data_nan_indices]
  if len(sample_indices) > 0:
    return imputed_dfs_dict
  else:  
    return imputed_df

def get_df_statistics(model, input_df, statistics):
  #Set statistics to "mean", "sd", or for percentile set to "p=2.5", "p=97.5" 
  posterior_samples = post_process_samples(model, model.unprocessed_samples)    
  df_statistics = input_df.copy()
  for var_name in df_statistics.columns:           
    data_obs_indices=list(np.argwhere(~np.isnan(df_statistics[var_name]).values).flatten())
    if statistics != "mean":
      df_statistics[var_name].values[data_obs_indices] = 0        
    found_posterior_nan_indices = []
    if var_name in list(posterior_samples.keys()):
      if statistics == "sd":         
        imputed_samples = posterior_samples[var_name].std(axis=-1)          
      elif statistics.find("p=") >= 0:
        percentile = float(statistics.replace("p=",""))
        imputed_samples = np.percentile(posterior_samples[var_name], q=percentile, axis=-1)          
      else:
        imputed_samples = posterior_samples[var_name].mean(axis=-1)       
      data_nan_indices=list(np.argwhere(np.isnan(df_statistics[var_name]).values).flatten())                     
      #found_posterior_nan_indices=list(np.argwhere(~np.isnan(imputed_samples)).flatten())       
      #assert(found_posterior_nan_indices == data_nan_indices), "nan indices in df and samples do not match"
      df_statistics[var_name].values[data_nan_indices] = imputed_samples[data_nan_indices]
  return df_statistics
  
def post_process_samples(model, posterior_samples):
    #Process posterior samples, reshape in (n,t,num_samples)
    post_process_dict = collections.defaultdict(list)
    multi_index_array = collections.defaultdict(list)    
    b_find_multi_index = False
    reshape_tuples = {}
    count_var = collections.defaultdict(int)
    for k,v in posterior_samples.items():
      v = np.array(v)    
      pos = k.find("[")
      if pos > 0:
        #Variable name contains [ ]
        k_only = k[:pos]
        pos_end = k.find("]")
        mid = k[pos+1:pos_end].strip()
        if k_only in model.obs:
          #Found missing data
          corresponding_df = model.find_df(k_only)
          if k_only not in list(post_process_dict.keys()):
            for i in range(corresponding_df[k_only].shape[0]):
              post_process_dict[k_only].append([np.nan for _ in range(model.num_samples)])
          found_index = int(mid)
          post_process_dict[k_only][found_index] = v
        else:
          post_process_dict[k_only].append(v)
    
        if k.find(",") >= 0:
          b_find_multi_index = True  
          mid_array = mid.split(',')
          mid_array = [int(i) for i in mid_array]
          multi_index_array[k_only].append(mid_array)
        else:
          count_var[k_only] += 1          
          reshape_tuples[k_only] = tuple([count_var[k_only]])      
      else:
        #Variable name does not contain [ ]
        reshape_tuples[k] = v.shape[1:]
        
    if b_find_multi_index:
      #Get largest tuple of indices
      for k_only, v_array in multi_index_array.items():
        reshape_tuples[k_only] = tuple([max(index)+1 for index in zip(*v_array)])        

    if len(post_process_dict) < 1:
      #Did not find other format, so using format of original samples
      for k,v in posterior_samples.items():
        post_process_dict[k] = np.array(v)
    else:
      #Check for any variable name without [ ] but with dim > 1
      for k,v in posterior_samples.items():
        if (k not in post_process_dict) and (k.find("[") < 0):
          post_process_dict[k] = np.array(v.T)        

    #Reshape in (n,t,num_samples)  
    for k,v in post_process_dict.items():
      if k not in model.obs:
        v = np.array(v)
        post_process_dict[k] = v
        if k in reshape_tuples:
          reshape_array = list(reshape_tuples[k])
          reshape_array.append(model.num_samples)
          reshape_tuple = tuple(reshape_array)
          if len(reshape_tuple) > 1:
            post_process_dict[k] = v.reshape(reshape_tuple)
      else:
        post_process_dict[k] = np.array(v)
    return post_process_dict
