### The modified pace data is in file find_offset/all_offset.py

### Whenever you format your code, make sure these five lines are at the top
```
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
```

### The steps from generating data to testing the model in the environment are as follows:
1. generate data
   - test data(plot data)
2. pretrain model Net
   - test model Net  
3. test model in environment
   
#### Training a fully connected network, run the following command:

1. `python collect/collect_o_a.py`  # collect o,a data
2. `python pretrain/pretrain_fc_deep.py`  # pretrain model Net
3. `python test_model/test_fc_deep.py`  # test model in environment

#### Training a attraction ring network, run the following command:
refactor 
1. `python collect/collect_oma_data.py`  # collect oma data
   - `python collect_test/test_oma.py`  # test data(plot data)
2. `python pretrain/pretrain_oma_data_Net.py`  # pretrain model Net
   - `python test_model/test_oma_Net.py`  # test model Net
3. `python test_model/test_oma_model.py`  # test model in environment


