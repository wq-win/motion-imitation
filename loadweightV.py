import pickle  
import numpy as np
  
# 使用pickle从文件加载数组  
with open('weight_dataV.pkl', 'rb') as f:  
    allresult = pickle.load(f)  

PN_V = allresult['PN_V']
PNtoKCweight = allresult['PNtoKCweight']
activate_KC_dims = allresult['activate_KC_dims']
KCtoMBONweight = allresult['KCtoMBONweight']

KC = PN_V @ PNtoKCweight
sorted_indices = np.argsort(KC, axis=1)[:, ::-1]  
inactivate_indices = sorted_indices[:, activate_KC_dims:]  
KC[np.arange(KC.shape[0])[:, None], inactivate_indices] = 0
MBON = KC @ KCtoMBONweight
print(MBON)

