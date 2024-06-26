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
```
1. collect_data/collect_o_a.py  # collect o,a data
2. pretrain/pretrain_fc_deep.py  # pretrain model Net
3. test_model/test_fc_deep.py  # test model in environment
```
#### Training a attraction ring network, run the following command:
```
refactor # 新版本
1. collect_data/collect_oma_data_from_pma_12dim.py  # 采集数据
2. pretrain/pretrain_save_data_V1.py  # 预训练
3. test_model/test_model.py  # 模型测试
```

```
old # 旧版本 
collect_data/save_data_V1.py    # 采集数据

pretrain/pretrain_save_data_V1.py # 预训练

test_model/sava_data_model_test_Net.py # 模型Net测试

test_model/save_data_model_right_offset.py   # 模型测试
```