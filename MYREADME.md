# Project Directory Structure
```
MOTION-IMITATION/
├── collect/         # 收集数据
├── collect_dataset/ # 存储增广数据集
├── collect_test/    # 画图查看数据集
├── find_offset/     # 找到所有偏移量
│ ├── all_offset.py  # 修改原始pace数据在这里
├── function_test/   # 测试不用model 的代码
├── model_test/      # 测试model在env中的表现
├── pretrain/        # 训练模型
├── pretrain_model/  # 存储训练的模型
├── pretrain_Net_test/ # 画图查看网络的输出
├── result/          # 存储偏移量结果、loss结果
├──MYREADME.md
```
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

1. `python collect/collect_o_a.py`                 # collect o,a data
2. `python pretrain/pretrain_fc_deep.py`           # pretrain model Net

#### Training a attraction ring network, run the following command:
refactor 
1. `python collect/collect_oma_data.py`            # collect oma data
   - `python collect_test/test_oma.py`             # test data(plot data)
2. `python pretrain/pretrain_oma_data_Net.py`      # pretrain model Net
   - `python pretrain_Net_test/test_oma_Net.py`    # test model Net
3. `python model_test/oma_model_test.py`           # test model in environment


