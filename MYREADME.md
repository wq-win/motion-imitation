
tensorboard --logdir=output/PPO1_144    
E:\VScode\motion-imitation\.venv\Scripts\python.exe test/step3_pretrain.py --pretrain_flag True
### History commit
+ disable fake actions: 
    + motion_imitation/envs/env_wrappers/imitation_task.py
        +  204 lines self._sync_sim_model(perturb_state) -> &#35; self._sync_sim_model(perturb_state)


+ disable action fileter:
    + motion_imitation/envs/locomotion_gym_config.py
        +  42 lines enable_action_filter = attr.ib(type=bool, default=True) -> &#35; enable_action_filter = attr.ib(type=bool, default=True) &#35; original config, affect the response time
        +  43 lines enable_action_filter = attr.ib(type=bool, default=False)
    + motion_imitation/robots/minitaur.py
    + 1403 lines # Trick
+ 4 conditions:
    + motion_imitation\envs\env_wrappers\imitation_terminal_conditions.py
        + 83-86 lines

### Run  
mpiexec -n 4 E:\VScode\motion-imitation\.venv\Scripts\python.exe motion_imitation/our_run_2.py
mpiexec -n 8 E:\VScode\motion-imitation\.venv\Scripts\python.exe motion_imitation/our_run_2.py --visualize
mpiexec -n 8 E:\VScode\motion-imitation\.venv\Scripts\python.exe motion_imitation/our_run_2.py  --pretrain_flag True --total_timesteps 40000000

E:\VScode\motion-imitation\.venv\Scripts\python.exe motion_imitation/our_run_2.py --pretrain_flag True 
--load_model_flag True
E:\VScode\motion-imitation\.venv\Scripts\python.exe motion_imitation/our_run_2.py --load_model_flag True --visualize


E:\VScode\motion-imitation\.venv\Scripts\python.exe motion_imitation/our_run.py --pretrain_flag True --load_model_flag True
E:\VScode\motion-imitation\.venv\Scripts\python.exe motion_imitation/our_run.py --load_model_flag True --visualize

mpiexec -n 8 python motion_imitation/our_run.py
mpiexec -n 8 python motion_imitation/our_run.py --total_timesteps 40000000
mpiexec -n 8 python motion_imitation/our_run.py --pretrain_flag True --load_model_flag True
mpiexec -n 8 E:\VScode\motion-imitation\.venv\Scripts\python.exe motion_imitation/our_run.py  --load_model_flag True

python motion_imitation/run.py --mode test --motion_file motion_imitation/data/motions/dog_trot.txt --model_file motion_imitation/data/policies/dog_trot.zip --visualize

### numpy version problem
+ E:\VScode\motion-imitation\motion_imitation\envs\env_wrappers\imitation_task.py 
  + 422 lines     vel_reward = np.exp(-self._velocity_err_scale * vel_err) ->     if -self._velocity_err_scale * vel_err <= -700:
      vel_reward = 0
    else: 
      vel_reward = np.exp(-self._velocity_err_scale * vel_err)
  + 503


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