import pandas as pd
import numpy as np
import tensorflow as tf

# 输入文件路径
# V2_HazardModel//hazardprediction_hds_model.h5 危险性评估模型
# V2_HazardEnv//traindata."+str(tem_renumber)+".npy 孕灾样本
# 'V2_RainfallInputs//Inputs_Wenchuan.csv' 流域编号与样本对应关系和降雨数据
# 'V2_Outputs//Outputs_Wenchuan.csv' 危险性输出位置

# 样本数据
# 初始化样本张量
#######################################
x_yunzai = np.zeros(shape=[1, 801, 801, 8], dtype='float32')
# 初始化降雨事件的降雨历时与诱发平均雨强
x_rainfall = np.zeros(shape=[2], dtype='float32')

# 危险性评估模型加载
#######################################
# 重新创建完全相同的模型，包括其权重和优化程序
hazard_model = tf.keras.models.load_model('V2_HazardModel//hazardprediction_hds_model.h5')
# 显示网络结构
hazard_model.summary()

# 为变量赋值
#######################################
rainfall_inputs_wenchuan = pd.read_csv('V2_RainfallInputs//Inputs_Wenchuan.csv')
# 新增一列
rainfall_inputs_wenchuan["Hazard"] = "NaN"



# 加载全流域二维特征张量
for index, row in rainfall_inputs_wenchuan.iterrows():
    # 读取孕灾样本数据，将用于危险性计算
    tem_renumber = int(rainfall_inputs_wenchuan.loc[index, 'renumber'])
    a = np.load("V2_HazardEnv//traindata."+str(tem_renumber)+".npy")
    # x_yunzai = a
    x_yunzai[0, :, :, :] = a

    print("第" + str(index) + "个孕灾样本已加载! " + str(tem_renumber))

    # 读取降雨历时（D）和平均降雨强度(I)
    x_rainfall = np.array([rainfall_inputs_wenchuan.loc[index, 'D'], rainfall_inputs_wenchuan.loc[index, 'I']], dtype='float32')
    x_rainfall = x_rainfall.reshape(-1, 2)
    # 进度
    print("第" + str(index) + "个降雨样本已加载!")

    # 调用危险性评估模型，评估危险性结果
    result_hazard = hazard_model.predict([x_yunzai, x_rainfall])
    rainfall_inputs_wenchuan.loc[index, 'Hazard'] = float(result_hazard)

# 结果保存

#将结果保存到第二个sheet里面
rainfall_inputs_wenchuan.to_csv('V2_Outputs//Outputs_Wenchuan.csv',  index=False, encoding="gb18030")
# 进度
print("危险性评估结束")
# 评估