import numpy as np
import pandas as df
import numpyro
from numpyro import handlers
from jax import random, vmap
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

def compute_predictions(BayesLDM_model, input_name, posterior_samples, data_dict, seed=0, b_display=True):  
  def get_predictions(rng_key, input_name, input_model, posterior_samples, *args, **kwargs):
    model_handler = handlers.seed(handlers.condition(input_model, posterior_samples), rng_key)
    model_trace = handlers.trace(model_handler).get_trace(*args, **kwargs)
    return_value = None
    if input_name in model_trace:
      return_value = model_trace[input_name]['value']
    else:
      if b_display:
        print('warning: {} is not a model parameter'.format(input_name))
    return return_value
  model   = BayesLDM_model.model_code
  rng_key = random.PRNGKey(seed)
  num_samples = BayesLDM_model.num_samples   
  predict_fn = vmap(lambda rng_key, params: get_predictions(rng_key, input_name, model, params, **data_dict))
  predictions = predict_fn(random.split(rng_key, num_samples), posterior_samples)  
  return predictions

def compute_log_likelihood(chosen_model, unprocessed_samples=None, data_dict=None, participants=None, n_index_name=None):
  '''This returns a dictionary with keys: log_likelihood_sum, likelihood, log_likelihood_dict, aic, bic.'''
  #For multi-index, with many participants, set participants and n_index_name (default from df index[0])
  #for example: participants=['10232', '10054'] and n_index_name='n'
  if data_dict is None:
    data_dict = chosen_model.data_dict
  model  = chosen_model.model_code
  b_scan = (chosen_model.chosen_implementation == 'scan')
  dfs    = chosen_model.data
  if unprocessed_samples is None:
    samples = chosen_model.unprocessed_samples
  else:
    samples = unprocessed_samples
  
  df_plots = {}
  if "df" in data_dict:
    df_plots["df"] = data_dict["df"]
  for df_i in dfs:
    if df_i.name not in df_plots:
      df_plots[df_i.name] = df_i
  if df_plots == {}:
    print('please enter the dataframes in compute_log_likelihood: dfs=[df1, df2...]')
    return {'log_likelihood_sum':None, 'log_likelihood':None, 'aic':None, 'bic':None}
  else:
    for k,v in df_plots.items():
      df_i = v
      df_i_indices = df_i.index.names   
      if len(df_i_indices) > 1:
        n_index_name = df_i_indices[0]        
        participants = list(set(df_i.index.get_level_values(n_index_name)))
        return chosen_model.compute_log_likelihood_more_participants_(participants, n_index_name,
          data_dict=data_dict, samples=samples, model=model, b_scan=b_scan, dfs=dfs)

  #Below is for one participant
      
  log_likelihood_dict = numpyro.infer.log_likelihood(model, samples, **data_dict)
  
  parameter_names = list(samples.keys())
  obs_names = list(log_likelihood_dict.keys())

  log_likelihood_obs_dict = {}
  scan_order_p = 1
  obs_names_unique = []
  for obs_name in obs_names:
    pos = obs_name.find('[')
    if pos < 0:
      obs_names_unique.append(obs_name)
    else:
      obs_names_unique.append(obs_name[:pos])
      if b_scan:
        stop = obs_name.find(']')
        mid = obs_name[pos+1:stop]
        found_index = (int)(mid)+1
        if found_index > scan_order_p:
          scan_order_p = found_index
  obs_names_unique = set(obs_names_unique)
  n_unique = len(obs_names_unique)
  
  log_likelihood_sums = np.zeros((n_unique))
  counts = np.zeros((n_unique), dtype=int)
  if b_scan:
    for i, name in enumerate(obs_names_unique):
      for j in range(scan_order_p):
        likelihood_name = name+'['+str(j)+']'
        if likelihood_name in log_likelihood_dict:
          log_likelihood_sums[i] += log_likelihood_dict[likelihood_name].sum()
          counts[i] += 1
    for k,v in log_likelihood_dict.items():
      for i, name in enumerate(obs_names_unique):
        if k == name:
          if str(type(v)).find("BatchTracer") > 0:
            log_likelihood_sums[i] += v.val.sum()
            counts[i] += v.val.shape[1]
          else:
            log_likelihood_sums[i] += v.sum()
            counts[i] += v.shape[1]

    log_likelihood_sum = log_likelihood_sums.sum()

    #-----------------------------------------------------------------------  
     
    obs_indices = []
    found_obs_indices = []
    for i, name in enumerate(obs_names_unique):
      for df_i in df_plots.values():
        for col in list(df_i.columns):
          if name.find(col) >= 0:          
            found_obs_indices = np.argwhere(~np.isnan(df_i[col].values)).flatten()
            obs_indices.append(found_obs_indices)
            break
    log_likelihood_sums = np.zeros((n_unique))
    counts = np.zeros((n_unique), dtype=int) 
    for i, name in enumerate(obs_names_unique):
      for j in range(scan_order_p): 
        likelihood_name = name+'['+str(j)+']'       
        if likelihood_name in log_likelihood_dict:            
          log_likelihood_sums[i] += log_likelihood_dict[likelihood_name].sum()        
          counts[i] += 1
          log_likelihood_obs_dict[likelihood_name] = log_likelihood_dict[likelihood_name]
    df_obs_indices = []
    for i, name in enumerate(obs_names_unique):
      for j in range(scan_order_p): 
        obs_indices[i] = np.delete(obs_indices[i], 0)
      df_obs_indices.append(obs_indices[i])
      obs_indices[i] = obs_indices[i] - scan_order_p      
    for k,v in log_likelihood_dict.items():
      for i, name in enumerate(obs_names_unique):
        if k == name:
          if str(type(v)).find("BatchTracer") > 0:
            log_likelihood_sums[i] += v.val[:,obs_indices[i]].sum()              
            counts[i] += len(obs_indices[i])
            for obs_i, obs_indices_values in enumerate(obs_indices[i]):
              new_name = k+'['+str(df_obs_indices[i][obs_i])+']'
              log_likelihood_obs_dict[new_name] = v.val[:,obs_indices_values]              
          else:
            log_likelihood_sums[i] += v[:,obs_indices[i]].sum()
            counts[i] += len(obs_indices[i])
            for obs_i, obs_indices_values in enumerate(obs_indices[i]):
              new_name = k+'['+str(df_obs_indices[i][obs_i])+']'
              log_likelihood_obs_dict[new_name] = v[:,obs_indices_values]
  else:
    for k,v in log_likelihood_dict.items():
      for i, name in enumerate(obs_names_unique):
        if str(k).find(name) >= 0:
          if str(type(v)).find("BatchTracer") > 0:
            log_likelihood_sums[i] += v.val.sum()
          else:
            log_likelihood_sums[i] += v.sum()
          counts[i] += 1
          log_likelihood_obs_dict[k] = v
    
  log_likelihood_sum = log_likelihood_sums.sum()

  n_samples = 0
  for v in log_likelihood_dict.values():
    n_samples = len(v)
    break
  log_likelihood = log_likelihood_sum / max(1,n_samples)

  (aic, bic) = compute_aic_bic(chosen_model, samples, log_likelihood, df_plots, None)
  
  return {'log_likelihood_sum':log_likelihood_sum, 'log_likelihood':log_likelihood,
          'log_likelihood_dict':log_likelihood_obs_dict, 'aic':aic, 'bic':bic}
  
def compute_log_likelihood_more_participants_(chosen_model, participants, n_index_name,
                                               data_dict, samples, model, b_scan, dfs=[]):
  #Set participants names and set index name, for example: participants=['10232','10054'] and n_index_name="n"
  df_plots = {}
  if "df" in data_dict:
    df_plots["df"] = data_dict["df"]
  for df_i in dfs:
    if df_i.name not in df_plots:
      df_plots[df_i.name] = df_i
  if df_plots == {}:
    print('please enter the dataframes in compute_log_likelihood: dfs=[df1, df2...]')
    return {'log_likelihood_sum':None, 'log_likelihood':None, 'aic':None, 'bic':None}
  
  log_likelihood_dict = numpyro.infer.log_likelihood(model, samples, **data_dict)

  log_likelihood_obs_dict = {}
  parameter_names = list(samples.keys())
  obs_names = list(log_likelihood_dict.keys())
  scan_order_p = 1
  obs_names_unique = []
  for obs_name in obs_names:            
    pos = obs_name.find('[')
    if pos < 0:
      if b_scan:
        for participant in participants:
          if obs_name.find(participant) >= 0:
            obs_names_unique.append(obs_name)
      else: 
        obs_names_unique.append(obs_name)
    else:
      if b_scan:
        stop = obs_name.find(']')
        mid = obs_name[pos+1:stop].strip()
        obs_name = obs_name[:pos]
        if obs_name.find(participant) >= 0:
          obs_names_unique.append(obs_name)              
        found_index = (int)(mid)+1
        if found_index > scan_order_p:
          scan_order_p = found_index
      else:
        obs_names_unique.append(obs_name[:pos])
  obs_names_unique = list(set(obs_names_unique))
  n_unique = len(obs_names_unique)  

  log_likelihood_sums = np.zeros((n_unique))
  counts = np.zeros((n_unique), dtype=int)
  if b_scan:
    for i, name in enumerate(obs_names_unique):
      for j in range(scan_order_p):
        likelihood_name = name+'['+str(j)+']'
        if likelihood_name in log_likelihood_dict:
          log_likelihood_sums[i] += log_likelihood_dict[likelihood_name].sum()
          counts[i] += 1
    for k,v in log_likelihood_dict.items():      
      for i, name in enumerate(obs_names_unique):          
        if (k.find(name) >= 0) and (k.find("[") < 0):
          log_likelihood_sums[i] += v.sum()
          counts[i] += v.shape[1]

    log_likelihood_sum = log_likelihood_sums.sum()

    #-----------------------------------------------------------------------  
    
    obs_indices_dict = collections.defaultdict(list)
    for name in obs_names_unique:
      for df_i in df_plots.values():
        name_only = None        
        for col in list(df_i.columns):          
          if name.find(col) >= 0:
            name_only = col
            participant = name.replace(col,"").strip()
            b_keep = True
            break
        if name_only is not None:
          groups = df_i[name_only].groupby(n_index_name)
          df_i = groups.get_group(participant)
          found_obs_indices = np.argwhere(~np.isnan(df_i.values)).flatten()
          obs_indices_key = name
          obs_indices_dict[obs_indices_key].append(found_obs_indices)
          
    log_likelihood_sums = np.zeros((n_unique))
    counts = np.zeros((n_unique), dtype=int) 
    for i, name in enumerate(obs_names_unique):
      for j in range(scan_order_p): 
        likelihood_name = name+'['+str(j)+']'     
        if likelihood_name in log_likelihood_dict:
          log_likelihood_sums[i] += log_likelihood_dict[likelihood_name].sum()        
          counts[i] += 1
          log_likelihood_obs_dict[likelihood_name] = log_likelihood_dict[likelihood_name]
    for i, name in enumerate(obs_names_unique):
      if name in obs_indices_dict:
        indices_so_far = obs_indices_dict[name]        
        for j in range(scan_order_p):
          indices_so_far = np.delete(indices_so_far, 0)
        obs_indices_dict[name] = indices_so_far
        obs_indices_dict[name] = obs_indices_dict[name] - scan_order_p
  
    for k,v in log_likelihood_dict.items():
      for i, name in enumerate(obs_names_unique):
        if (k.find(name) >= 0) and (k.find("[") < 0):
          log_likelihood_sums[i] += v[:,obs_indices_dict[name]].sum()
          counts[i] += len(obs_indices_dict[name])
          log_likelihood_obs_dict[k] = v[:,obs_indices_dict[name]]
  else:  
    for k,v in log_likelihood_dict.items():
      for i, name in enumerate(obs_names_unique):
        if str(k).find(name) >= 0:
          if str(type(v)).find("BatchTracer") > 0:
            log_likelihood_sums[i] += v.val.sum()
          else:
            log_likelihood_sums[i] += v.sum()
          counts[i] += 1
          log_likelihood_obs_dict[k] = v

  log_likelihood_sum = log_likelihood_sums.sum()

  n_samples = 0
  for v in log_likelihood_dict.values():
    n_samples = len(v)
    break
  log_likelihood = log_likelihood_sum / max(1,n_samples)
  
  (aic, bic) = compute_aic_bic(chosen_model, samples, log_likelihood, df_plots, participants)
  
  return {'log_likelihood_sum':log_likelihood_sum, 'log_likelihood':log_likelihood,
          'log_likelihood_dict':log_likelihood_obs_dict, 'aic':aic, 'bic':bic}

def compute_aic_bic(chosen_model, posterior_samples, log_likelihood, dfs, participants):
  #participants = None if there is no index n, if multi-index with n, then add participants
  n_all = 0
  n_obs = 0
  for df_i in dfs.values():
    n_all += df_i.shape[0] * df_i.shape[1]
    n_obs += df_i.count().sum()
  parameter_names = list(posterior_samples.keys())
  k_all = len(parameter_names)
  obs_names = chosen_model.obs
  aic_bic_params = []
  for param_name in parameter_names:
    check_name = param_name
    b_check = True
    pos = param_name.find("[")
    if pos > 0:
      check_name = param_name[:pos]
      if participants is not None:
        b_check = False
        for participant in participants:
          check_name = check_name.replace(str(participant),"")
    for obs_name in obs_names:
      if (check_name != obs_name) and b_check:
        if check_name not in aic_bic_params:
          check_shape = posterior_samples[param_name].shape
          if (param_name.find("[") < 0) and (len(check_shape) > 1):
            dim_param = int(posterior_samples[param_name].shape[1])
            for d in range(dim_param):
              aic_bic_params.append(check_name+"["+str(d)+"]")
          else:
            aic_bic_params.append(check_name)
  aic_bic_params = list(set(aic_bic_params))
  k_only = len(aic_bic_params)
  aic = -2*log_likelihood + 2 * k_only
  bic = -2*log_likelihood + np.log(n_obs) * (k_only)
  return (aic, bic)
