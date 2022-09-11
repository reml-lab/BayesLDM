import numpy as np
import pandas as pd
import numpyro
import numpyro.distributions as dist
import numpyro.diagnostics as diagnostics
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.contrib.control_flow import scan
from jax import random
import jax.numpy as jnp
from jax.numpy import log, exp
from jax.scipy.special import logsumexp, expit
from textx import metamodel_from_str
import matplotlib.pyplot as plt
from scipy import stats
import networkx as nx
import collections
import itertools
import textwrap
import timeit
import re

class Parser:
  def __init__(self, dsl_text):
    self.distribution_names = [ 'BernoulliSample', 'BernoulliLogitsSample', 'BetaSample', 'BinomialSample', 
                                'BinomialLogitsSample', 'CauchySample', 'CategoricalSample', 'DirichletSample',
                                'ExponentialSample', 'GammaSample', 'GumbelSample', 'HalfCauchySample', 
                                'HalfNormalSample', 'InverseGammaSample', 'LogNormalSample', 'NormalSample', 
                                'MixtureSameFamilySample', 'MultivariateNormalSample', 'TruncatedNormalSample', 
                                'PoissonSample', 'StudentTSample', 'ZeroInflatedPoissonSample']
    dist_statement = '|'.join(self.distribution_names)

    grammar = """    
    ProbProgram:
      statements*= Statement
    ;
    Statement:
      Indices | Inputs | Outputs | ProgramName | Assignment | {}
    ;
    Index:
      variable=ID min_index=INT max_index=INT
    ;
    Indices:
      'Indices:' (variables+=Index[','])?
    ;
    Inputs:
      'Inputs:' (variables+=ID[','])?
    ;
    Outputs:
      'Outputs:' (variables+=ID[','])?
    ;
    ProgramName:
      'ProgramName:' program_name=ID
    ;
    BernoulliSample:
      variable=Variable '~' 'Ber(' theta=Expression ')'
    ;    
    BernoulliLogitsSample:
      variable=Variable '~' 'Berlogits(' logits=Expression ')'
    ;
    BetaSample:
      variable=Variable '~' 'Beta(' a=Expression ',' b=Expression ')'
    ;
    BinomialSample:
      variable=Variable '~' 'Binomial(' total_count=Expression ',' probs=Expression ')'
    ;
    BinomialLogitsSample:
      variable=Variable '~' 'BinomialLogits(' total_count=Expression ',' logits=Expression ')'
    ;    
    CategoricalSample:
      variable=Variable '~' 'Categorical(' probs=Ratios ')'
    ;
    CauchySample:
      variable=Variable '~' 'Cauchy(' mean=Expression ',' sd=Expression ')'
    ;
    DirichletSample:
      variable=Variable '~' 'Dirichlet(' concentration=Ratios ')'
    ;
    ExponentialSample:
      variable=Variable '~' 'Exp(' rate=Expression ')'
    ;
    GammaSample:
      variable=Variable '~' 'Gamma(' concentration=Expression ',' rate=Expression ')'
    ;
    GumbelSample:
      variable=Variable '~' 'Gumbel(' loc=Expression ',' scale=Expression ')'
    ;
    HalfCauchySample:
      variable=Variable '~' 'HalfCauchy(' scale=Expression ')'
    ;
    HalfNormalSample:
      variable=Variable '~' 'HalfNormal(' scale=Expression ')'
    ;
    InverseGammaSample:
      variable=Variable '~' 'InverseGamma(' concentration=Expression ',' rate=Expression ')'
    ;    
    LogNormalSample:
      variable=Variable '~' 'LogNormal(' mean=Expression ',' sd=Expression ')'
    ;    
    MixtureSameFamilySample:
      variable=Variable '~' 'MixtureSameFamily(' mixing=Expression ',' component=Expression ')'
    ;    
    MultivariateNormalSample:
      variable=Variable '~' 'MultivariateNormal(' mean=Expression ',' cov=Expression ')'
    ;
    NormalSample:
      variable=Variable '~' 'N(' mean=Expression ',' sd=Expression ')'
    ;
    TruncatedNormalSample:
      variable=Variable '~' 'TruncatedNormal(' loc=Expression ',' scale=Expression ',' low=Expression ',' high=Expression ')'
    ;
    PoissonSample:
      variable=Variable '~' 'Poisson(' rate=Expression ')'
    ;
    StudentTSample:
      variable=Variable '~' 'StudentT(' degreef=Expression ')'
    ;
    ZeroInflatedPoissonSample:
      variable=Variable '~' 'ZeroInflatedPoisson(' gate=Expression ',' rate=Expression ')'
    ;    
    Assignment:
      variable=LeftHandSide '=' value=Expression 
    ;    
    Expression:
      Sum | Difference | Term
    ;
    ExpressionInt:
      Sum | Difference | Function | Variable | INT
    ;
    Function:
      func=FuncName '(' (parameters+=Expression[','])? ')'
    ;
    Derivative:
      'd' ID '/dt'
    ;
    LeftHandSide:
      Derivative | Variable
    ;    
    Variable:
      Array | ID 
    ;
    Array:
      name=ID '[' (contents+=ExpressionInt[','])? ']'
    ;
    Ratios:
      SquareBrackets | Array | ID
    ;   
    SquareBrackets:
      '[' (contents+=Expression[','])? ']'
    ;
    Sum:
      term1=Term '+' term2=Expression
    ;
    Difference:
      term1=Term '-' term2=Expression
    ;
    Term:
      Product | Division | Factor
    ;
    Product:
      term1=Factor '*' term2=Term
    ;
    Division:
      term1=Factor '/' term2=Term
    ;
    Factor:
      Function | Variable | FLOAT | INT | Parenthesis
    ;    
    Parenthesis:
      '(' (contents+=Expression[','])? ')'
    ;    
    FuncName:
      ID('.'ID)*  
    ;
    """.format(dist_statement)
    
    meta = metamodel_from_str(grammar)
    dsl_parse = meta.model_from_str(dsl_text)
    self.statements = dsl_parse.statements

  def get_statements(self):
    return self.statements

  def get_header(self, statements):
    function_name = ''; indices_dict = {}; inputs = ''; outputs = ''
    for statement in statements:
      type_name = self.get_type_name(statement)
      if type_name == "ProgramName":  function_name = statement.program_name
      elif type_name == "Indices":    indices_dict = self.makeIndices(statement.variables)
      elif type_name == "Inputs":     inputs  = statement.variables
      elif type_name == "Outputs":    outputs = statement.variables
    header_dict = {'function_name': function_name, 'indices_dict': indices_dict, 'inputs':inputs, 'outputs':outputs}
    return header_dict

  def process_distribution(self, statement):
    # this is for both distribution and assignment
    type_name = self.get_type_name(statement); statement_dict = {}
    if type_name in self.distribution_names or type_name == "Assignment":
      before = []; after = []
      if type_name == "Assignment":                 parameters = [statement.value]
      elif type_name == "BernoulliSample":          parameters = [statement.theta]
      elif type_name == "BernoulliLogitsSample":    parameters = [statement.logits]; before=['logits=']
      elif type_name == "BetaSample":               parameters = [statement.a, statement.b]
      elif type_name == "BinomialSample":           parameters = [statement.total_count, statement.probs]
      elif type_name == "BinomialLogitsSample":     parameters = [statement.logits, statement.total_count]
      elif type_name == "CategoricalSample":        parameters = [statement.probs]; before=['jnp.array(']; after=[')']
      elif type_name == "CauchySample":             parameters = [statement.mean, statement.sd]
      elif type_name == "DirichletSample":          parameters = [statement.concentration]; before=['jnp.array(']; after=[')']
      elif type_name == "ExponentialSample":        parameters = [statement.rate]
      elif type_name == "GammaSample":              parameters = [statement.concentration, statement.rate]
      elif type_name == "GumbelSample":             parameters = [statement.loc, statement.scale]
      elif type_name == "HalfCauchySample":         parameters = [statement.scale]
      elif type_name == "HalfNormalSample":         parameters = [statement.scale]
      elif type_name == "InverseGammaSample":       parameters = [statement.concentration, statement.rate]          
      elif type_name == "LogNormalSample":          parameters = [statement.mean, statement.sd]
      elif type_name == "MixtureSameFamilySample":  parameters = [statement.mixing, statement.component]          
      elif type_name == "MultivariateNormalSample": parameters = [statement.mean, statement.cov]
      elif type_name == "NormalSample":             parameters = [statement.mean, statement.sd]          
      elif type_name == "TruncatedNormalSample":
        parameters = [statement.loc, statement.scale, statement.low, statement.high]
        before=['loc=','scale=','low=','high=']; after=['','','','']
      elif type_name == "PoissonSample":            parameters = [statement.rate]          
      elif type_name == "StudentTSample":           parameters = [statement.degreef]; before=['df=']   
      elif type_name == "ZeroInflatedPoissonSample":parameters = [statement.gate, statement.rate]
      else:
        print('\ncannot find {}: this type is not implemented yet'.format(type_name))
        assert(1==2), 'error: type not implemented yet'
      statement_dict =  {'type_name': type_name, 'parameters': parameters, 'before': before, 'after': after }
    return statement_dict

  def get_type_name(self, x):
    return type(x).__name__

  def makeIndices(self,statement):
    indices = {}
    for item in statement:
      indices[item.variable] = {"min_index":item.min_index, "max_index":item.max_index}
    return indices

  def makeExpression(self, expression):
    exp_type = self.get_type_name(expression)
    if exp_type == 'Sum': return self.makeSum(expression)
    if exp_type == 'Difference': return self.makeDifference(expression)
    if exp_type == 'Function': return self.makeFunction(expression)
    if exp_type == 'SquareBrackets':
      return "[" + ",".join(self.makeExpression(p) for p in expression.contents) + "]"
    else: return self.makeTerm(expression)

  def makeFunction(self, function):
    return function.func + "(" + ",".join(self.makeExpression(p) for p in function.parameters) + ")"

  def makeParenthesis(self, parenthesis):
    return "(" + ",".join(self.makeExpression(p) for p in parenthesis.contents) + ")"
  
  def makeArray(self, variable):
    str_array = variable.name + "[" + ",".join(self.makeExpression(p) for p in variable.contents) + "]"
    str_array.replace(".0,", ",").replace(".0]", "]")
    return str_array

  def makeTerm(self, term):
    exp_type = self.get_type_name(term)
    if exp_type=='Product': return self.makeProduct(term)
    elif exp_type=='Division': return self.makeDivision(term)
    else: return self.makeFactor(term)

  def makeProduct(self, product):
    term1 = self.makeTerm(product.term1)
    term2 = self.makeTerm(product.term2)
    return "{}*{}".format(term1,term2)
 
  def makeDivision(self, expr):
    term1 = self.makeTerm(expr.term1)
    term2 = self.makeTerm(expr.term2)
    return "{}/{}".format(term1,term2)

  def makeSum(self, sum):
    term1 = self.makeTerm(sum.term1)
    term2 = self.makeExpression(sum.term2)
    return "{}+{}".format(term1,term2)

  def makeDifference(self, diff):
    term1 = self.makeTerm(diff.term1)
    term2 = self.makeExpression(diff.term2)
    return "{}-{}".format(term1,term2)

  def makeFactor(self, factor):
    exp_type = self.get_type_name(factor)
    if exp_type in ['int', 'float']: return str(factor)
    elif exp_type == 'Array': return self.makeArray(factor)
    elif exp_type == 'Parenthesis': return self.makeParenthesis(factor)
    elif exp_type == 'Log': return self.makeLog(factor)
    elif exp_type == 'str': return(factor)
    elif exp_type == 'Function': return self.makeFunction(factor)
    else: return "[Err]"


class compile:
  def __init__(self, program_text, obs={}, data={}, b_graph=False, b_display=False, n_level=0, b_user_warning=True):
    # n_level can be: 0 default, 1 optimize using scan, 2 defaut with plate
    globals_dict = {}
    for k,v in globals().items():
      str_global_item = str(k)              
      if str_global_item.find("__") < 0:
        globals_dict[str_global_item] = v
    self.function_names = list(globals_dict.keys())
    self.globals_dict = globals_dict
    self.initialize(obs, data)
    self.n_level   = n_level
    self.b_graph   = b_graph
    self.b_display = b_display
    self.b_user_warning = b_user_warning
    self.program_text  = program_text
    self.parser = Parser(program_text)
    self.python_text = self.makeProgram(self.parser.get_statements())
    if self.b_display:
      print('program_text =\n', program_text); print()
      print('python_text =\n', self.python_text); print()

  def initialize(self, obs, data):
    self.obs = obs
    self.data = data
    self.function_name = 'function_name'
    self.inputs  = []
    self.outputs = []
    self.indices_dict = {}
    self.nodes = []
    self.edges = []
    self.topological_order = []
    self.array_inputs_dict = {}
    self.array_variables_dict = {}
    self.distributions_dict = {}
    self.tab1 = '  '
    self.tab2 = self.tab1+self.tab1
    self.data_dict = {}
    self.fitted_mcmc = None
    self.r_hat_warnings = ''
    self.chosen_implementation = ''

  def condition(self, obs, data):
    self.initialize(obs, data)
    self.python_text = self.makeProgram(self.parser.get_statements())
    
  def sample(self, n_seed=0, num_samples=1000, num_warmup=500, num_chains=1, b_show_numpyro_summary=True,
             b_post_process=False, b_display=False, **kwargs):
    if(self.python_text!=""):         
      exec_scope=self.globals_dict
      exec(self.python_text, exec_scope)
      model = exec_scope[self.function_name]
      self.model_code = model
 
      title = 'NUTS ' + self.chosen_implementation + ' (warmup={}, samples={})'.format(num_warmup, num_samples)
      if b_display:
        print('start sampling using {}...'.format(title))
      self.num_samples = num_samples
      self.num_warmup  = num_warmup
      self.num_chains  = num_chains
      kernel = NUTS(model)
      mcmc = MCMC(kernel, num_samples=num_samples, num_warmup=num_warmup, num_chains=num_chains, progress_bar=False)

      if(self.data is not None and len(self.obs)>0):
        for df_check in self.data:
          if df_check.name not in self.data_dict:
            self.data_dict[df_check.name] = df_check
      for var in self.inputs:
        if var in kwargs:
          self.data_dict[var] = kwargs[var]
        elif self.data is not None:
          for df_i in self.data:
            if var in df_i:
              self.data_dict[var] = df_i[var].values
        else:
          print("Input variable %s not found"%var)
          return None

      start_time = timeit.default_timer()

      mcmc.run(random.PRNGKey(n_seed), **self.data_dict)

      duration = timeit.default_timer() - start_time
      minutes = int(duration) // 60
      seconds = int(duration) - minutes * 60
      str_duration = title+' '+'duration = {} seconds ({} minutes {} seconds)'.format(duration, minutes, seconds)
      self.duration = duration
      self.str_duration = str_duration
      self.fitted_mcmc = mcmc
      if b_show_numpyro_summary:
        mcmc.print_summary()
      if b_display:
        print('\n'+str_duration+'\n')
      self.model_code = model
      samples = mcmc.get_samples()
      self.unprocessed_samples = samples
      if b_post_process:
        samples = self.post_process(samples)
      return samples
  
  def extract_indices(self, str_terms, b_brackets):
    # this outputs the list of indices found in str_terms
    # for example: input str_terms = 'a.y[t-1] + b.y[t]'
    # if b_bracket = True:  output = ['[t-1]', '[t]']
    # if b_bracket = False: output = ['t-1', 't']
    detail1 = ''; detail2 = ''
    for item in self.inputs:
      # insert index indicator (i.e.: A[n,t])
      detail2 += '|{}\[.*?\]'.format(item)
    indices_list = list(re.findall(r'\[.*?\]{}'.format(detail2), str_terms))    
    for i, item in enumerate(indices_list):
      item = str(item)
      for check_input in self.inputs:
        pos = item.find(check_input)
        if pos >= 0:
          if pos > 0:
            item = item[pos:]          
          item = '[{}]'.format(item)
      indices_list[i] = item.replace('.0]',']')  
    if not b_brackets:
      for i,item in enumerate(indices_list): 
        indices_list[i] = item[1:-1]    
    return indices_list

  def get_term_array(self, str_terms):
    # this outputs (array of non-numeric terms, clean str_terms)
    # for example: input str_terms = 'a*y[t-1.0]+b'
    # output = (['a', 'y[t-1]', 'b'], 'a*y[t-1]+b')
    # clean_str_terms = str_terms
    indices_list = self.extract_indices(str_terms, b_brackets=True)
    clean_str_terms = str_terms.replace('.0]',']')     
    replace_dict = {}; str_terms_check = clean_str_terms
    # mask indices before split via operators
    for i,old in enumerate(indices_list):
      new = 'INDEX'+str(i)
      replace_dict[new] = old
      str_terms_check = str_terms_check.replace(old, new)     
    delim = '_SPLIT_'; operators = ['+', '-', '*', '/', '**']
    for operator in operators:
      str_terms_check = delim.join(str_terms_check.split(operator))
    term_array = str_terms_check.split(delim)   
    # restore masked indices
    for j,item in enumerate(term_array):
      for old, new in replace_dict.items():  
        item = item.replace(old, new)
      term_array[j] = item    
    return term_array, clean_str_terms

  def draw_graph(self, nodes, edges, b_graph):
    str_nodes = ''
    for node_id in nodes:
      str_nodes += '\n\t<node id="{}"/>'.format(node_id)
    str_edges = ''
    for edge in edges:
      str_edges += '\n\t<edge source="{}" target="{}"/>'.format(edge['source'], edge['target'])
    graph_str = '''<?xml version='1.0' encoding='utf-8'?>
    <graphml>
      <graph edgedefault="directed">{}{}
      </graph>
    </graphml>'''.format(str_nodes, str_edges)
    f = open("graph.graphml","w"); f.write(graph_str); f.close()
    G = nx.read_graphml('graph.graphml')
    detail = ''
    if b_graph:
      if len(self.nodes) > 50:
        plt.figure(figsize=(10,10))
      else:
        plt.figure(figsize=(6,6))
      b_planar, _ = nx.check_planarity(G)
      if b_planar:
        pos = nx.planar_layout(G)
        nx.draw(G, pos=pos, with_labels=True, arrowsize=10, node_size=1000, width=2, font_size=10, node_color='skyblue')
      else:
        detail = ' (not planar)'
        nx.draw(G, with_labels=True, arrowsize=10, node_size=1000, width=2, font_size=10, node_color='skyblue')
      plt.show()
    topological_order = []
    if nx.is_directed_acyclic_graph(G):
      topological_order = list(nx.topological_sort(G))
      if self.b_display:
        if len(topological_order) > 50:
          print('graph is a DAG{}, found topological_order ='.format(detail))
          print('\n'.join(textwrap.wrap(str(topological_order),100))); print()
        else:
          print('graph is a DAG{}, found topological_order ='.format(detail))
          print('\n'.join(textwrap.wrap(str(topological_order),100))); print()
    else:
      print('\ngraph is not a DAG, found cycle = {}\n'.format(nx.find_cycle(G)))
      assert(1==2), 'error: graph is not a DAG'
    return topological_order, graph_str

  def remove_function_name(self, node_name):
    for check_function in ['pow(', 'log(', 'exp(']:
      pos = node_name.find(check_function)
      if (pos >= 0):
        if check_function == 'pow(':
          start = pos + len(check_function)
          end   = start + node_name[start:].find(',')
          node_name = node_name[start:end]
        else:
          start = pos + len(check_function)
          end   = start + node_name[start:].find(')')
          node_name = node_name[start:end]
    return node_name

  def insert_node(self, node_name):
    node_name = self.remove_function_name(node_name)
    if (node_name not in self.nodes) and (node_name not in list(self.indices_dict.keys())) and (
        node_name not in self.inputs) and (not node_name.replace('.','').isnumeric()):
      self.nodes.append(node_name)

  def insert_edge(self, source, target):
    source = self.remove_function_name(source)
    if (source not in list(self.indices_dict.keys())) and (source not in self.inputs) and (
        len(source) >= 0) and (not source.replace('.','').isnumeric()):
      edge = {'source':source, 'target':target }
      if (edge not in self.edges):            
        self.edges.append(edge)

  def expand_and_insert_edge_for_temporal_nodes(self):
    # expand nodes (i.e.: if y[0] exists then replace y[t] with y[1], y[2]..., y[max_index])
    init_nodes = self.nodes.copy()
    for node in init_nodes:
      if node.find('[') >= 0:
         # get start_index_list and max_length_list for combination_index_tuples
        short_name, current_index_name = self.extract_array_short_name_and_index(node)
        assigned_index = self.array_variables_dict[short_name]['assigned_index']
        index_so_far   = self.array_variables_dict[short_name]['index_so_far']
        current_index_list  = current_index_name.split(',')
        assigned_index_list = assigned_index.split(',')
        index_so_far_list   = str(index_so_far).split(',')
        start_index_list = []
        max_length_list  = []
        # if variable has index indicator (i.e.: A[n,t] or W[n,t]), do not include in combination_index_tuples
        # because we will use all the index indicator values when checking the parameters
        b_found_input_index = False
        for input_i in self.inputs:
          if node.find(input_i) >= 0:
            b_found_input_index = True
            break        
        if not b_found_input_index:          
          for index_dim, item in enumerate(current_index_list):            
            if item.isnumeric():
              # if index is numeric (i.e.: 0), then max length = 1
              start_index_list.append(str(item))
              max_length_list.append(str(int(item)+1))
            else:
              # if index is not numeric (i.e.: n), then max length = max_index+1
              start_index_list.append(str(int(index_so_far_list[index_dim])+1))  
              max_length_list.append(str(self.indices_dict[assigned_index_list[index_dim]]['max_index']+1))

          # get combination_index_tuples to expand nodes         
          if node in self.distributions_dict:         
            array_index_values = []
            n_index_dim = len(start_index_list)
            for index_dim in range(n_index_dim):
              start = int(start_index_list[index_dim]); end = int(max_length_list[index_dim])
              index_values = list(np.arange(start, end))
              array_index_values.append(index_values)   
            combination_index_tuples = list(itertools.product(*array_index_values))          

            # expand nodes using combination_index_tuples, and insert nodes, edges, distributions_dict
            for index_tuple in combination_index_tuples:
              index_tuple = str(index_tuple).replace(' ','').replace('(','').replace(')','')
              if n_index_dim <= 1:
                index_tuple = index_tuple.replace(',','')
              new_var_name = short_name+'['+index_tuple+']'
              self.insert_node(new_var_name)
              parameters = self.distributions_dict[node]['parameters']
              new_dist_params = []

              # check parameters to construct edge
              for param_dim, old_param_name in enumerate(parameters):
                param_term_list, new_param_str = self.get_term_array(old_param_name)                  
                for param_dim, param_term in enumerate(param_term_list):  
                  term_indices = self.extract_indices(param_term, b_brackets=False)
                  new_param_name = param_term
                  b_found_input_index = False
                  if str(param_term).find('[') >= 0:
                    # if parameter has indicator index (i.e.: wbb[A[n,t]]]) then: 
                    # replace parameter assigned_index with all its possible values (i.e.: wbb[a] where a=0 and 1)
                    for input_i in self.inputs:
                      if param_term.find(input_i) >= 0:                        
                        # found indicator index, then insert edge on all values of indicator index
                        param_short_term, param_index_name = self.extract_array_short_name_and_index(param_term)
                        if param_short_term not in self.inputs:
                          # for now, assume wcc[A[n,t]] has 1-dim index
                          assign_i = str(self.array_variables_dict[param_short_term]['assigned_index'])                 
                          for possible_values_i in range(self.indices_dict[assign_i]['min_index'], self.indices_dict[assign_i]['max_index']+1):
                            new_param_name = param_term.replace(param_index_name, str(possible_values_i))                          
                            self.insert_edge(source=new_param_name, target=new_var_name)
                        b_found_input_index = True
                        break                  
                  if not b_found_input_index:
                    # if parameter does not have indicator index then:
                    # replace parameter assigned_index with index_tuple from combination_index_tuples, then
                    # use eval (i.e.: if t=2 then replace y[n,t-1] with y[0,2-1], then y[0,1])                 
                    for old_index in term_indices:
                      index_tuple_list = index_tuple.split(',')
                      replaced_tuple = old_index                                          
                      for index_dim in range(len(assigned_index_list)):
                        replaced_tuple = replaced_tuple.replace(assigned_index_list[index_dim], index_tuple_list[index_dim])
                      replaced_list = replaced_tuple.split(',')
                      for replace_index, replaced_list_item in enumerate(replaced_list):
                        replaced_list[replace_index] = str(eval(replaced_list_item))
                      new_eval = ','.join(replaced_list)
                      new_param_name = new_param_name.replace(old_index, new_eval)
                      new_param_str  = new_param_str.replace(old_index,  new_eval)                             
                    if len(new_param_name) > 0:
                        if not new_param_name.isnumeric():
                          self.insert_edge(source=new_param_name, target=new_var_name)                        
                new_dist_params.append(new_param_str)
              
              # update distributions_dict
              if new_var_name not in self.distributions_dict:
                if new_var_name.find('[') >= 0:
                  new_short_name, new_index_name = self.extract_array_short_name_and_index(new_var_name)
                  new_index_name_list = new_index_name.split(',')
                  for param_dim, param_check in enumerate(new_dist_params):                
                    # replace any remaining non-numeric index with its value (i.e.: 3.0*n with 3.0*0, for y[0])
                    for index_dim in range(len(assigned_index_list)):                               
                      new_index = new_index_name_list[index_dim]                      
                      param_check = param_check.replace(assigned_index_list[index_dim], new_index)
                      new_dist_params[param_dim] = param_check
                distribution_name = self.distributions_dict[node]['distribution']
                before = self.distributions_dict[node]['before']
                after  = self.distributions_dict[node]['after']
                self.distributions_dict[new_var_name] = {'distribution': distribution_name, 
                                                         'parameters': new_dist_params,
                                                         'before':before, 'after':after}          
    if self.b_user_warning:
      # sanity check for Indices max mispecification
      for check_name in list(self.distributions_dict.keys()):
        if check_name.find('[') >= 0:
          short_name, current_index_name = self.extract_array_short_name_and_index(check_name)
          corresponding_df = self.array_variables_dict[short_name]['corresponding_df']
          if corresponding_df is not None:
            assigned_index_list = self.array_variables_dict[short_name]['assigned_index'].split(',')
            check_index_list = current_index_name.split(',')
            for index_dim, index_i in enumerate(check_index_list):
              assigned_index_i = assigned_index_list[index_dim]
              max_index_i_from_indices = self.indices_dict[assigned_index_i]['max_index']
              max_index_i_from_df = len(corresponding_df.index.get_level_values(index_dim))-1           
              if max_index_i_from_indices > max_index_i_from_df:
                detail = '\nplease fix Indices {} max to match {} {} max.\n'.format(assigned_index_i, corresponding_df.name, assigned_index_i)
                print('\nfound index max mismatch: cannot find variable {}, because Indices {} max = {} > {} index {} max = {}.{}'.format(
                      check_name, assigned_index_i, max_index_i_from_indices, corresponding_df.name, assigned_index_i, max_index_i_from_df, detail))
                assert(1==2), 'Indices and df index max mismatch'

    # remove nodes, edges, distributions that have non-numeric index (i.e.: y[t-1])
    for node in init_nodes:
      if node.find('[') >= 0:
        short_name, current_index_name = self.extract_array_short_name_and_index(node)
        check_index_list = current_index_name.split(',')
        for item in check_index_list:
          if not item.isnumeric():
            if node in self.nodes: 
              self.nodes.remove(node)
      for edge in self.edges:
        target_name = edge['target']
        pos = target_name.find('[')
        if pos >= 0:
          target_index = target_name[pos+1:-1]
          target_index_list = target_index.split(',')
          for item in target_index_list:
            if not item.isnumeric():
              if edge in self.edges:
                self.edges.remove(edge)

  def construct_array_variables_dict(self):
    for node in self.nodes:
      pos = node.find('[')
      if pos >= 0:
        short_name = node[:pos]
        if short_name not in self.array_variables_dict:
          # get corresponding_df
          corresponding_df = None
          for df_i in self.data:
            if short_name in list(df_i.columns):
              corresponding_df = df_i.copy()
              corresponding_df.name = df_i.name
              break
          
          # get assigned_index from df or input text
          n_levels = None
          assigned_index = None
          if corresponding_df is not None:
            assigned_index = corresponding_df.index.name
          if assigned_index is None:
            # find all the possible indices (i.e.: 'n,0', 'n,t'...), then choose ('n,t')
            check_nodes = []
            for check_node in self.nodes:
              if check_node.find(short_name) >= 0:
                check_nodes.append(check_node)
            indices = self.extract_indices('+'.join(check_nodes), b_brackets=False)
            for check_index in indices:
              if str(check_index).find(',') >= 0:
                # for multi-index (i.e: n,t)
                check_list = check_index.split(',')
                n_levels = len(check_list)
                n_count = 0
                for level, check_level in enumerate(check_list):
                  check_level = re.sub('[^a-zA-Z]+','',check_level)
                  check_list[level] = check_level
                  if len(check_level) > 0:
                    if not check_level.isnumeric():
                      n_count += 1
                if n_count == n_levels:
                  assigned_index = ','.join(check_list)
                  break
              else:
                # for 1 index (i.e.: t)
                index_name = re.sub('[^a-zA-Z]+','',check_index)
                if len(index_name) > 0:
                  if not index_name.isnumeric():
                    assigned_index = index_name
                    break

          # get index_so_far
          index_so_far = -1; index_so_far_list = []  
          for check_node in self.nodes:          
            if (check_node.find(short_name) >= 0) and (check_node in self.distributions_dict):
              if check_node.find('[') >= 0:
                _, current_index_name = self.extract_array_short_name_and_index(check_node)
                if str(current_index_name).find(',') >= 0:
                  # for multi-index (i.e: n,t)                  
                  current_index_list = current_index_name.split(',')
                  n_index_dim = len(current_index_list)
                  if len(index_so_far_list) == 0:
                    index_so_far_list = [str(-1) for _ in range(n_index_dim)]
                  for index_dim, item in enumerate(current_index_list):
                    if item.isnumeric():
                      if int(item) > (int)(index_so_far_list[index_dim]):
                        index_so_far_list[index_dim] = str(item)                
                else:
                  # for 1 index (i.e.: t)
                  if current_index_name.find(assigned_index) < 0:
                    if str(current_index_name).isnumeric():
                      if int(current_index_name) > int(index_so_far):
                        index_so_far = current_index_name
          if len(index_so_far_list) > 0:
            index_so_far = ','.join(index_so_far_list)
          self.array_variables_dict[short_name] = {'assigned_index':assigned_index,
                                                   'corresponding_df':corresponding_df,
                                                   'index_so_far':index_so_far}
  def makeDef(self, function_name, input_vars):
    def_list = []
    if(self.data is not None and len(self.obs)>0):
      check_duplicates = []
      for i, df_check in enumerate(self.data):
        if df_check.name not in check_duplicates:
          check_duplicates.append(df_check.name)
          def_list.append(df_check.name + "=None")
    if len(input_vars) > 0:
      str_vars = ",".join("{}=None".format(v) for v in input_vars)
      def_list.append(str_vars)
    python_text = "def {}({}):".format(function_name, ','.join(def_list))
    return python_text

  def makeReturn(self, output_vars):
    if len(output_vars)==0:
      output_str=""
    else:
      dict_list = ['\"{}\":{}'.format(s,s) for s in output_vars]
      details = ",".join(dict_list) 
      output_str= "{{{}}}".format(details)
    python_text = "\n{}return({})".format(self.tab1,output_str)
    return python_text

  def makeBody(self, topological_order):
    # choose implementation: default or scan
    scan_candidates  = self.get_transition_candidates()
    DBN_candidates   = self.get_transition_candidates(b_check_if_DBN=True)
    plate_candidates = self.get_plate_candidates()
    
    if self.n_level == 1:
      if len(DBN_candidates) > 0:
        body_text = self.construct_scan_code(self.topological_order, DBN_candidates, plate_candidates)
      else:
        self.n_level = 2

    if self.n_level == 2:
      if len(scan_candidates) > 0:
        body_text = self.construct_scan_code(self.topological_order, scan_candidates, plate_candidates)
      else:
        self.chosen_implementation = 'plate'
        body_text = ''; plate_init_so_far = []; plate_dist_so_far = []     
        for var_name in self.topological_order:
          body_text, plate_init_so_far, plate_dist_so_far = self.get_plate_text(
              var_name, plate_candidates, body_text, plate_init_so_far, plate_dist_so_far)
          
    if self.n_level == 0:
      self.chosen_implementation = 'default'
      body_text = ''; obs_array_dict = collections.defaultdict(list)
      for var_name in self.topological_order:
        body_text += self.get_full_distribution_text(var_name, self.tab1)
      # construct outputs
      if len(self.outputs) > 0:
        for var_name in self.topological_order:        
          if (var_name.find('[') >= 0):
            short_name, index_name = self.extract_array_short_name_and_index(var_name)
            check_short_name = short_name + '_' + '_'.join(str(index_name).split(','))
            if (short_name in self.outputs) and (body_text.find(check_short_name) >= 0):
              obs_array_dict[short_name].append(check_short_name)
        for output_obs in self.outputs:
          outputs_list = [str(x).replace("'",'') for x in str(obs_array_dict[output_obs]).split(',')]
          body_text += '\n{}'.format(self.tab1)+output_obs+'='+','.join(outputs_list)
    return body_text

  def makeProgram(self, statements):
    header_dict = self.parser.get_header(statements)
    self.function_name = header_dict['function_name']; self.indices_dict = header_dict['indices_dict']
    self.inputs = header_dict['inputs']; self.outputs = header_dict['outputs'] 
    for statement in statements:
      # construct edges, nodes and distributions_dict
      statement_type = self.parser.get_type_name(statement)    
      if (statement_type in self.parser.distribution_names) or (statement_type == 'Assignment'):
        parameters = self.parser.process_distribution(statement)['parameters']
        before = self.parser.process_distribution(statement)['before'] 
        after  = self.parser.process_distribution(statement)['after']
        variable_name = self.parser.makeFactor(statement.variable)
        parameter_expressions = []
        # check parameters and insert node and edge
        for param in parameters:
          param_type = self.parser.get_type_name(param)
          expression = self.parser.makeExpression(param)
          if param_type == 'str':
            self.insert_node(expression)
            self.insert_edge(source=expression, target=variable_name)
          elif param_type in ['Sum', 'Difference', 'Product', 'Division', 'Array']:
            terms = self.parser.makeTerm(expression)  
            terms_array, expression = self.get_term_array(terms)
            for term in terms_array:
              if (len(term) > 0):
                self.insert_node(term)
                self.insert_edge(source=term, target=variable_name)
                for short_input_name in self.inputs:
                  term = term.replace(' ','')
                  pos = term.find(short_input_name+'[')
                  if pos >= 0:
                    # construct array_inputs_dict to keep track of input arrays (i.e.: {A: A[n,t], W: W[n,t]} )
                    full_array_input_name = term[pos:term[pos:].find(']')+pos+1]
                    self.array_inputs_dict[short_input_name] = full_array_input_name                               
          parameter_expressions.append(expression.replace(' ','')) 
        self.insert_node(variable_name)
        if variable_name not in self.distributions_dict:   
          self.distributions_dict[variable_name] = {'distribution':statement_type.replace('Sample',''),
                                                    'parameters':parameter_expressions,
                                                    'before':before, 'after':after}          
    self.construct_array_variables_dict()  
    self.expand_and_insert_edge_for_temporal_nodes()
    self.topological_order, _ = self.draw_graph(self.nodes, self.edges, self.b_graph)

    def_text    = self.makeDef(self.function_name, self.inputs)
    body_text   = self.makeBody(self.topological_order)
    return_text = self.makeReturn(self.outputs)
    return def_text +"\n"+ body_text +"\n"+ return_text

  def get_distribution_text(self, dist_key, str_version, str_obs):
    # str_version can be: default, scan, plate
    text = ''
    if dist_key in self.distributions_dict:
      param_list = self.distributions_dict[dist_key]['parameters'].copy()
      dist_name  = self.distributions_dict[dist_key]['distribution']
      before_list = self.distributions_dict[dist_key]['before'].copy()
      after_list  = self.distributions_dict[dist_key]['after'].copy()
      short_name = dist_key[:dist_key.find('[')]
      index_name = dist_key[dist_key.find('[')+1:dist_key.find(']')]
      new_var_name  = dist_key
      new_mcmc_name = new_var_name      
      if (str_version == 'default'):
        # udapte variable names from 'y[n,t]' to 'y_n_t' 
        new_var_name  = dist_key.replace('[','_').replace(']','').replace(',','_')
        for i,item in enumerate(param_list):
          indices_list = self.extract_indices(item, b_brackets=True)
          sub_item_list, _ = self.get_term_array(item)          
          for sub_item in sub_item_list:
            pos = sub_item.find('[')
            if pos >= 0:
              check_name = sub_item[:pos]
              old_index  = sub_item[pos:]
              b_is_input = False
              if check_name in self.inputs:
                b_is_input = True
              if not b_is_input:               
                # update parameter from y[1-1] to y_0, if it is not an input variable
                if sub_item.find('pow(') >= 0:
                  # TODO check for multi-index
                  new_sub_item = sub_item.replace('[','_').replace(']','')
                else:
                  new_sub_item = sub_item.replace('[','_').replace(']','').replace(',','_')                
                new_index = new_sub_item[pos:]
                pos = new_index.find('_')
                if (new_index.find('-') >= 0) and (pos == 0):             
                  new_index_list = new_index[1:].split('_')
                  for dim_i, index_value in enumerate(new_index_list):
                    if (index_value.find('-') >= 0):
                      index_value = str(eval(index_value))
                      new_index_list[dim_i] = index_value
                  new_index = ''
                  for new_value in new_index_list:
                    new_index += '_' + new_value                           
                item = item.replace(check_name+old_index, check_name+new_index)
          param_list[i] = item        
        param_str = ','.join(param_list).replace(' ','') 
      elif (str_version == 'scan'):
        # update variable names from 'y[n,t]' to 'y_n_t' 
        new_var_name = dist_key.replace('[','_').replace(']','').replace(',','_')
        # update parameters
        for i,item in enumerate(param_list):
          indices_list = self.extract_indices(item, b_brackets=True)
          for old_index in indices_list:
            new_index = old_index
            array_inputs = list(self.array_inputs_dict.values())
            if len(array_inputs) > 0:
              for full_input_i in array_inputs:
                # if index indicator, then replace A[n,t] with A_t
                if old_index.find(full_input_i) >= 0:
                  old_index = old_index[1:-1]                
                  new_index = old_index[:old_index.find('[')]+'_t'
                  item = item.replace(old_index, new_index)
              else:
                # if not index indicator, then replace y[n,d-1] with y_n_dm1
                new_index = old_index.replace('[','_').replace(']','').replace(',','_').replace('-','m')              
              item = item.replace(old_index, new_index)  
            else:
              # if not index indicator, then replace y[n,d-1] with y_n_dm1
              new_index = old_index.replace('[','_').replace(']','').replace(',','_').replace('-','m')              
              item = item.replace(old_index, new_index)          
          param_list[i] = item     
      elif str_version == 'plate':
        new_var_name  = short_name
        new_mcmc_name = short_name
      if (str_version == 'scan'):
        new_mcmc_name = new_var_name.replace('[','_').replace(']','').replace(',','_')

      param_list_with_before_after = []
      for param_index, param_name in enumerate(param_list):
        before_param = ''; after_param = ''
        if len(before_list) > 0:
          before_param = before_list[param_index]
        if len(after_list) > 0:
          after_param = after_list[param_index]            
        param_list_with_before_after.append(before_param + param_name + after_param)        
      param_str = ','.join(param_list_with_before_after).replace(' ','')
      if dist_name == 'Assignment':
        text = '{}={}'.format(new_var_name, param_str)
      else:
        str_enumerate = ''
        if (dist_name.find('Ber') >= 0) or (dist_name.find('Binom') >= 0) or (
            dist_name.find('Poisson') >= 0 or dist_name.find('Cat') >= 0):
          str_enumerate = ',infer={"enumerate":"parallel"}'
        text = '{}=numpyro.sample("{}",dist.{}({}){}{})'.format(new_var_name, new_mcmc_name, 
                dist_name, param_str, str_enumerate, str_obs)
    return text

  def extract_array_short_name_and_index(self, input_name):
    short_name = None; index_name = None
    pos = input_name.find('[')
    if pos >= 0:
      short_name = input_name[:pos]
      index_name = input_name[pos+1:input_name.rfind(']')]
    return short_name, index_name

  def get_distribution_text_with_impute(self, dist_key, str_implementation, value_name):
    short_name, index_name = self.extract_array_short_name_and_index(dist_key)
    str_obs = ',obs=None if (np.isnan({}.loc[{}])) else {}.loc[{}]'.format(value_name, index_name, value_name, index_name)
    text = '{}'.format(self.get_distribution_text(dist_key, str_implementation, str_obs))
    return text

  def get_full_distribution_text(self, dist_key, str_tab, str_implementation='default'):
    # this inserts imputation: it calls get_get_distribution_text_with_impute and distribution_text
    if str_tab is None:
      str_tab = self.tab1
    short_name = dist_key[:dist_key.find('[')]
    corresponding_df = None
    if short_name in self.array_variables_dict:
      corresponding_df = self.array_variables_dict[short_name]['corresponding_df']
    b_use_impute = (corresponding_df is not None) and (short_name in self.obs)
    if b_use_impute:
      value_name = '{}["{}"]'.format(corresponding_df.name, short_name)  
      full_distribution_text = '\n{}{}'.format(str_tab, self.get_distribution_text_with_impute(dist_key, str_implementation, value_name))
    else:
      full_distribution_text = '\n{}{}'.format(str_tab, self.get_distribution_text(dist_key, str_implementation, ''))
    return full_distribution_text

  def get_transition_candidates(self, b_check_if_DBN=False):
    # search for variable with distribution that contains some lag (i.e.: y[t] ~ N(y[t-1],1))
    # set b_check_if_DBN=True to insert additional checks for DBN 
    transition_candidates = []
    if len(self.data) > 0:  # for now, we require that df exists to perform scan
      for node in list(self.distributions_dict.keys()):
        if str(node).find('[') >= 0:
          parameter_list = self.distributions_dict[node]['parameters']
          for param in parameter_list:
            if param.find('[') >= 0 and param.find('-') >= 0:
              b_keep = True
              if b_check_if_DBN:
                pass # TODO
              if b_keep:
                transition_candidates.append(node)
    return transition_candidates

  def get_plate_candidates(self):
    body_text = ''    
    # get distributions_only = variable names with distribution (i.e.: not Assignment)
    distributions_only = []
    for check_node in list(self.distributions_dict.keys()):
      if self.distributions_dict[check_node]['distribution'] != 'Assignment':
        distributions_only.append(check_node)

    # search for plate_candidates (i.e.: y[0] does not exist and y[n] ~ N(0,1) or N(mu,1) where mu=2)
    plate_candidates = []
    for k,v in self.distributions_dict.items():
      if k.find('[') >= 0:
        short_name, index_name = self.extract_array_short_name_and_index(k)
        index_so_far = self.array_variables_dict[short_name]['index_so_far']
        index_name_list   = str(index_name).split(',')
        index_so_far_list = str(index_so_far).split(',')
        b_index_so_far_m1 = True
        for index_so_far_i in index_so_far_list:
          # y[0] does not exist if index_so_far_i = -1
          if int(index_so_far_i) > -1:
            b_index_so_far_m1 = False
        if b_index_so_far_m1: 
          b_numeric = True
          for index_i in index_name_list:
            if not str(index_i).isnumeric():
              b_numeric = False
          if not b_numeric:
            # check all parameters do not contain any index (i.e.: t) nor any variables with distribution (i.e. mu=2)
            parameters_list = v['parameters']
            b_plate = True
            for param_i in parameters_list:
              for check_index in list(self.indices_dict.keys()):
                if str(param_i).find(check_index) >= 0:
                  b_plate = False
                  break
              for check_node in distributions_only:
                if (str(param_i).find(check_node) >= 0) and (check_node.find('[') >= 0):
                  b_plate = False
                  break
            if b_plate:
              plate_candidates.append(k)
            else:
              break      
          else:
            plate_candidates.append(k)
    return plate_candidates

  def get_plate_text(self, var_name, plate_candidates, body_text, plate_init_so_far, plate_dist_so_far):
    if var_name.find('[') >= 0:
      short_name, index_name = self.extract_array_short_name_and_index(var_name)
      if (str(index_name).find(',') < 0) and (var_name in plate_candidates) and (short_name not in self.obs):
        assigned_index = self.array_variables_dict[short_name]['assigned_index'] 
        max_index = self.indices_dict[assigned_index]['max_index']
        plate_text_init = '\n{}with numpyro.plate("{}_plate",{}):'.format(self.tab1, assigned_index, max_index+1)
        if plate_text_init not in plate_init_so_far:
          plate_init_so_far.append(plate_text_init)
          body_text += plate_text_init          
        plate_text_dist = self.get_full_distribution_text(var_name, self.tab2, str_implementation='plate')
        if plate_text_dist not in plate_dist_so_far:
          plate_dist_so_far.append(plate_text_dist)
          body_text += plate_text_dist 
      else:
        body_text += self.get_full_distribution_text(var_name, self.tab1)          
    else:
      body_text += self.get_full_distribution_text(var_name, self.tab1)
    return body_text, plate_init_so_far, plate_dist_so_far

  def construct_scan_code(self, topological_order, transition_candidates, plate_candidates):
    self.chosen_implementation = 'scan'
    str_tab1 = self.tab1; str_tab2 = str_tab1 + str_tab1; str_tab3 = str_tab2 + str_tab1; text = ''
    short_name_transition_candidates  = [str(x[:x.find('[')]) for x in transition_candidates]
      
    # get simple parameters before loop
    init_count = 0; plate_init_so_far = []; plate_dist_so_far = []
    for init_topo_order_so_far, var_name in enumerate(topological_order):      
      if var_name.find('[') < 0:
        text += self.get_full_distribution_text(var_name, str_tab1)        
        init_count += 1
      elif var_name in plate_candidates:
        text, plate_init_so_far, plate_dist_so_far = self.get_plate_text(
              var_name, plate_candidates, text, plate_init_so_far, plate_dist_so_far)
        init_count += 1        
    text += '\n'

    # get assigned_indices (i.e.: n,t) and transition_var_names (i.e.: construct order of transition_var_names according to topological order)
    transition_var_names = []; y_assigned_indices = []
    for var_name in topological_order:
      pos = var_name.find('[')
      if pos >= 0:
        short_name = var_name[:pos]
        if (short_name in short_name_transition_candidates ) and (short_name not in transition_var_names):
          assigned_index_i = self.array_variables_dict[short_name]['assigned_index']
          transition_var_names.append(short_name)
          y_assigned_indices.append(assigned_index_i)

    # check if multi-index. TODO for now assume x, y have indices n,t (chosen_y_dim = 0 for x)
    chosen_y_dim  = 0
    chosen_y_name = transition_var_names[chosen_y_dim]
    chosen_y_assigned_index_list = y_assigned_indices[chosen_y_dim].split(',')
    corresponding_df = self.array_variables_dict[chosen_y_name]['corresponding_df']
    n_multi_index_length = len(chosen_y_assigned_index_list)

    if n_multi_index_length > 1:
      # loop on left most index
      n_name = chosen_y_assigned_index_list[0]
      max_length = int(self.indices_dict[n_name]['max_index'])+1 
      if n_name not in corresponding_df.index.names:
        # sanity check for index name 'n' must match in Indices and data frame
        print('\ncannot find index {} in data frame {}. Please fix index name in Indices to match data frame index.\n'.format(n_name, corresponding_df.name))
        assert(1==2), 'error index name does not match in Indices and data frame'
      text += '\n{}for {},{}_ in enumerate({}.index.unique("{}")):'.format(str_tab1, n_name, n_name, corresponding_df.name, n_name)
      str_tab1 += str_tab1
      str_tab2 += str_tab1

    # get impute distributions for transition
    for y_dim, y_name in enumerate(transition_var_names):
      df_i = self.array_variables_dict[y_name]['corresponding_df']
      df_i_name = df_i.name
      assigned_index_i_list = str(y_assigned_indices[y_dim]).split(',')
      if n_multi_index_length > 1:
        text += '\n{}{}_values=np.array({}["{}"].loc[{}_,:])'.format(str_tab1, y_name, df_i_name, y_name, n_name)
      else:
        text += '\n{}{}_values=np.array({}["{}"].values)'.format(str_tab1, y_name, df_i_name, y_name)
    
    # get inputs for transition
    array_input_names = list(self.array_inputs_dict.keys())
    for input_name in array_input_names:
      _, index_input_name = self.extract_array_short_name_and_index(self.array_inputs_dict[input_name])
      index_input_name_list = index_input_name.split(',')
      if n_multi_index_length > 1:
        if index_input_name_list[0] == n_name:
          text += '\n{}{}_values=np.array({}[{},:])'.format(str_tab1, input_name, input_name, n_name)
        else:
          text += '\n{}{}_values=np.array({})'.format(str_tab1, input_name, input_name)
      else:
        text += '\n{}{}_values=np.array({})'.format(str_tab1, input_name, input_name)

    # get init distributions and carry start
    carry_start_list = []; max_index_so_far = 0
    dist_text_so_far = []
    for topo_order_so_far, var_name in enumerate(topological_order[init_count:]):
      if var_name.find('[') < 0:
        text += self.get_full_distribution_text(var_name, str_tab1)
      else:
        short_name, index_name = self.extract_array_short_name_and_index(var_name)
        index_so_far = self.array_variables_dict[short_name]['index_so_far']
        index_so_far_list = str(index_so_far).split(',')        
        index_name_list = str(index_name).split(',')
        current_left_most  = index_name_list[0]
        current_right_most = index_name_list[-1]
        if n_multi_index_length > 1:
          # for multi-index, search for y[n,0] (note: we perform scan on right most index)
          for dist_name in list(self.distributions_dict.keys()):            
            if (dist_name.find(short_name+'[') >= 0) and (dist_name.find(current_right_most) >= 0):
              dist_short_name, dist_index_name = self.extract_array_short_name_and_index(dist_name)
              dist_index_name_list = str(dist_index_name).split(',')
              dist_left_most  = dist_index_name_list[0]
              dist_right_most = dist_index_name_list[-1]
              if (not str(dist_left_most).isnumeric()) and (int(dist_right_most) <= int(index_so_far_list[-1])):
                str_obs = ',obs=None if (np.isnan({}_values[{}])) else {}_values[{}]'.format(
                           short_name, dist_right_most, short_name, dist_right_most)
                dist_text = '\n{}{}'.format(str_tab1, self.get_distribution_text(dist_name, 'default', str_obs))
                old_name = dist_name.replace('[','_').replace(']','').replace(',','_')
                new_var_name  = short_name+'_'+str(dist_right_most)
                new_mcmc_name = '"{}"+str({}_)+str([{}])'.format(short_name, dist_left_most, dist_right_most)
                dist_text = dist_text.replace(old_name, new_var_name).replace('"'+dist_name+'"', new_mcmc_name)
                if dist_text not in dist_text_so_far:
                  text += dist_text
                  dist_text_so_far.append(dist_text)
                  carry_start_list.append(new_var_name)
        else:
          # not multi-index
          for index_dim, index_i in enumerate(index_name_list):
            index_so_far_i = int(index_so_far_list[index_dim])
            if str(index_i).isnumeric():
              if int(index_i) <= index_so_far_i:
                str_obs = ',obs={}_values[{}] if (not np.isnan({}_values[{}])) else None'.format(
                           short_name, index_name, short_name, index_name)
                text += '\n{}{}'.format(str_tab1, self.get_distribution_text(var_name, 'default', str_obs))
                carry_start_list.append(var_name.replace('[','_').replace(']','').replace(',','_'))
        right_index_so_far = index_so_far_list[-1]
        if str(right_index_so_far).isnumeric():
          if int(right_index_so_far) > int(max_index_so_far):
            max_index_so_far = int(right_index_so_far)

    start_scan_index = max_index_so_far + 1

    y_nan_indices_list = []
    for y_name in transition_var_names:
      text += '\n{}{}_impute=jnp.zeros((len({}_values)))'.format(str_tab1, y_name, y_name)
    for y_name in transition_var_names:
      text += '\n{}{}_nan_indices=[nan_id for nan_id in np.argwhere(np.isnan({}_values)).flatten() if nan_id >= {}]'.format(
               str_tab1, y_name, y_name, start_scan_index)
    for y_dim, y_name in enumerate(transition_var_names):
      text += '\n{}for i in {}_nan_indices: '.format(str_tab1, y_name)
      detail = '' if (n_multi_index_length <= 1) else 'str({}_)+'.format(str(y_assigned_indices[y_dim]).split(',')[0])
      text += '{}_impute={}_impute.at[i].set(numpyro.sample("{}"+{}str([i]),dist.Normal(0.0,10.0).mask(False)))'.format(
               y_name, y_name, y_name, detail)

    # construct carry in transition and return
    carry_transition_list = []; carry_return_list = []
    for carry_index, carry_name in enumerate(carry_start_list):
      pos = carry_name.find('_')+1
      carry_transition_index = int(carry_name[pos:]) - start_scan_index
      carry_transition_list.append(carry_name[:pos] + str(carry_transition_index).replace('-','tm'))
      carry_return_index = carry_transition_index + 1
      carry_return_list.append(carry_name[:pos] + str(carry_return_index).replace('0','t').replace('-','tm'))    
    carry_start_names  = ",".join(carry_start_list) 
    carry_return_names = ",".join(carry_return_list)
    node_return_names1 = ",".join([x+'_t' for x in transition_var_names])
    node_return_names2 = ",".join(transition_var_names)

    # construct transition
    text += '\n\n{}def transition(carry,value):'.format(str_tab1)
    for carry_index, carry_name in enumerate(carry_transition_list):
      text += '\n{}{}=carry[{}]'.format(str_tab2, carry_name, carry_index)
    text += '\n{}t_carry=carry[{}]'.format(str_tab2, len(carry_transition_list))
    for y_dim, y_name in enumerate(transition_var_names):
      value_detail = ''
      if (len(transition_var_names) > 1) or (len(array_input_names) >= 1):
        value_detail = '[{}]'.format(y_dim)
      text += '\n{}{}_t_obs=jnp.where(jnp.any(t_carry==jnp.array({}_nan_indices)),{}_impute[t_carry],value{})'.format(
              str_tab2, y_name, y_name, y_name, value_detail)
    count = y_dim+1
    for input_name in array_input_names:  
      text += '\n{}{}_t=value[{}]'.format(str_tab2, input_name, count)
      count += 1
    for y_name in transition_var_names:
      assigned_index = self.array_variables_dict[y_name]['assigned_index']
      var_name = y_name + '[' + assigned_index + ']'
      str_obs = ',obs={}_t_obs'.format(y_name)
      dist_name = self.get_distribution_text(var_name, 'scan', str_obs)
      if n_multi_index_length > 1:
        assigned_index_list = assigned_index.split(',')
        n_name = assigned_index_list[0]
        # if multi-index, remove '_n' in parameter y_n_tm1 to get y_tm1
        dist_name = dist_name.replace('_{}'.format(n_name),'')
        mcmc_old = '"{}_t"'.format(y_name); mcmc_new_name = '"{}"+str({}_)'.format(y_name, n_name)
        dist_name = dist_name.replace(mcmc_old, mcmc_new_name)
      dist_name = dist_name.replace('_'+assigned_index+'=', '_t=').replace('_'+assigned_index+'m1', '_tm1')
      dist_name = dist_name.replace('_'+assigned_index+'+', '_t+').replace('_'+assigned_index+'-', '_t-')
      dist_name = dist_name.replace('_'+assigned_index+'*', '_t*').replace('/'+assigned_index+'-', '_t/')
      dist_name = dist_name.replace('_'+assigned_index+'**', '_t**')
      text += '\n{}{}'.format(str_tab2, dist_name)
    text += '\n{}return ({},t_carry+1),({})'.format(str_tab2, carry_return_names, node_return_names1)
    text += '\n\n{}carry_start=({},{})'.format(str_tab1, carry_start_names, start_scan_index)
    transition_values_list = []  
    for y_name in transition_var_names:
      transition_values_list.append('{}_values[{}:]'.format(y_name, start_scan_index))
    for input_name in list(self.array_inputs_dict.keys()):
      transition_values_list.append('{}_values[{}:]'.format(input_name, start_scan_index))
    str_transition_values = ','.join(transition_values_list)
    if len(transition_values_list) > 1:
      str_transition_values = '['+str_transition_values+']'
    text += '\n{}_,({})= scan(transition,carry_start,{})'.format(str_tab1, node_return_names2, str_transition_values) 
    return text
 
  def get_duration(self):
    return self.str_duration

  def get_model_code(self):
      return self.model_code

  def post_process(self, samples):
    # output samples with shape = (...,num_sample)
    post_processed_samples = {}
    samples_array = []
    multi_index_params = []
    for k,v in samples.items():
      if (len(v.shape) > 1):
        post_processed_samples[k] = np.transpose(v)
      else:
        pos = k.find('[')
        if pos >= 0:
          short_name = k[:pos]
          if k.find(',') >= 0:
            if np.char.count(k,',') < 2:
              if short_name not in multi_index_params:
                multi_index_params.append(short_name)
            else:
              post_processed_samples[k] = v
          else:
            samples_array.append(v)
            post_processed_samples[short_name] = np.array(samples_array)     
        else:
          post_processed_samples[k] = v
    if len(multi_index_params) > 0:
      for short_name in multi_index_params:
        assigned_index = self.array_variables_dict[short_name]['assigned_index']
        assigned_index_list = assigned_index.split(',')
        if len(assigned_index_list) == 2:
          len_i = self.indices_dict[assigned_index_list[0]]['max_index']+1
          len_j = self.indices_dict[assigned_index_list[1]]['max_index']+1
          matrix_values = np.zeros((len_i, len_j, self.num_samples))
          for i in range(len_i):
            for j in range(len_j):
              str_name = short_name+'['+str(i)+','+str(j)+']'
              matrix_values[i,j] = samples[str_name]
          post_processed_samples[short_name] = matrix_values
    if len(post_processed_samples) == 0:
      post_processed_samples = samples.copy() 
    return post_processed_samples

  def find_df(self, variable_name, b_display=True):
    found_df = None
    for df_i in self.data:
      if variable_name in df_i:
        found_df = df_i
        break
    if found_df is None:
      if b_display:
        print('no df was found for', variable_name)
    return found_df
