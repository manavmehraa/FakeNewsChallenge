import pandas as pd 

def read_data(data_dir):
  train_stances = pd.read_csv(data_dir+'train_stances.csv')
  train_bodies = pd.read_csv(data_dir+'train_bodies.csv')
  train_df = train_stances.join(train_bodies.set_index('Body ID'), on = 'Body ID')

  train_h, train_b = train_df['Headline'].to_list(), train_df['articleBody'].to_list()
  
  test_stances = pd.read_csv(data_dir+'competition_test_stances.csv')
  test_bodies = pd.read_csv(data_dir+'competition_test_bodies.csv')
  test_df = test_stances.join(test_bodies.set_index('Body ID'), on = 'Body ID')
  test_h, test_b = test_df['Headline'].to_list(), test_df['articleBody'].to_list()
  
  return train_df, test_df, train_h, train_b, test_h, test_b