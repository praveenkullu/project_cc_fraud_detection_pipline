import pandas as pd
import numpy as np

DATA = '../data/'

test_txn  = pd.read_csv(DATA + 'train_transaction.csv')
test_id   = pd.read_csv(DATA + 'train_identity.csv')

test  = test_txn.merge(test_id,   on='TransactionID', how='left')

test.to_csv(DATA + "merge_test.csv")
