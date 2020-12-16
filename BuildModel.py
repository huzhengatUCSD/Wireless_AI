# 导入库函数
import keras
from keras import layers

# 工具列表
########################################################
# 1. MiniDeepST
# 2. ResUnit
# 3. MiniSTResNet
# 4. ConvLSTMCell
# 5. ConvLSTM
########################################################

def MiniDeepST(closeness, period, trend, filters, kernel_size, activation, use_bias):    
    closeness = layers.Conv2D(filters, kernel_size, padding='same', activation=activation, use_bias=use_bias)(closeness)
    period    = layers.Conv2D(filters, kernel_size, padding='same', activation=activation, use_bias=use_bias)(period)
    trend     = layers.Conv2D(filters, kernel_size, padding='same', activation=activation, use_bias=use_bias)(trend)
    
    closeness = layers.Conv2D(filters, kernel_size, padding='same', activation=activation, use_bias=use_bias)(closeness)
    period    = layers.Conv2D(filters, kernel_size, padding='same', activation=activation, use_bias=use_bias)(period)
    trend     = layers.Conv2D(filters, kernel_size, padding='same', activation=activation, use_bias=use_bias)(trend)
    
    closeness = layers.Conv2D(filters, (1,1), padding='same', activation=activation, use_bias=use_bias)(closeness)
    period    = layers.Conv2D(filters, (1,1), padding='same', activation=activation, use_bias=use_bias)(period)
    trend     = layers.Conv2D(filters, (1,1), padding='same', activation=activation, use_bias=use_bias)(trend)
     
    fusion    = layers.Add()([closeness, period, trend])
    fusion    = layers.Conv2D(filters, kernel_size, padding='same', activation=activation, use_bias=use_bias)(fusion)
    fusion    = layers.Conv2D(filters, kernel_size, padding='same', activation=activation, use_bias=use_bias)(fusion)
    fusion    = layers.Conv2D(filters, kernel_size, padding='same', activation=activation, use_bias=use_bias)(fusion)
    fusion    = layers.Conv2D(1, kernel_size, padding='same', activation=activation, use_bias=use_bias)(fusion)
    res       = layers.Flatten(name='output')(fusion)
    return res

def ResUnit(data, filters, kernel_size):
    res = layers.ReLU()(data)
    res = layers.Conv2D(filters, kernel_size, padding='same', activation='relu', use_bias=True)(res)
    res = layers.Conv2D(filters, kernel_size, padding='same', activation=None, use_bias=True)(res)
    res = layers.Add()([res,data])
    return res

def MiniSTResNet(closeness, period, trend, filters, kernel_size, activation, use_bias):    
    closeness = layers.Conv2D(filters, kernel_size, padding='same', activation=activation, use_bias=use_bias)(closeness)
    period    = layers.Conv2D(filters, kernel_size, padding='same', activation=activation, use_bias=use_bias)(period)
    trend     = layers.Conv2D(filters, kernel_size, padding='same', activation=activation, use_bias=use_bias)(trend)
    
    closeness = ResUnit(closeness, filters, kernel_size)
    period    = ResUnit(period, filters, kernel_size)
    trend     = ResUnit(trend, filters, kernel_size)
    
    closeness = ResUnit(closeness, filters, kernel_size)
    period    = ResUnit(period, filters, kernel_size)
    trend     = ResUnit(trend, filters, kernel_size)
    
    closeness = layers.Conv2D(filters, (1,1), padding='same', activation=activation, use_bias=use_bias)(closeness)
    period    = layers.Conv2D(filters, (1,1), padding='same', activation=activation, use_bias=use_bias)(period)
    trend     = layers.Conv2D(filters, (1,1), padding='same', activation=activation, use_bias=use_bias)(trend)
    
    fusion = layers.Add()([closeness,period,trend])
    fusion = layers.Conv2D(1, (1,1), padding='same', activation=activation, use_bias=use_bias)(fusion)
    fusion = layers.Flatten(name='output')(fusion)
    return fusion

def ConvLSTMCell(data):
    conv = layers.Conv2D(64, (3,3), padding='same', activation='relu', use_bias=True)(data)
    conv = ResUnit(conv, 64, (3,3))
    conv = ResUnit(conv, 64, (3,3))
    
    lstm_data = layers.Reshape((data.shape[-1], 20, 20, 1))(data)
    lstm = layers.ConvLSTM2D(64, (3,3), strides=(1, 1), padding='same', activation='relu', use_bias=True)(lstm_data)
    
    lstm = layers.Conv2D(64, (1,1), padding='same', activation='relu', use_bias=True)(lstm)
    conv = layers.Conv2D(64, (1,1), padding='same', activation='relu', use_bias=True)(conv)
    res  = layers.Add()([lstm, conv])
    return res

def ConvLSTM(closeness, period, trend):
    closeness = ConvLSTMCell(closeness)
    period    = ConvLSTMCell(period)
    trend     = ConvLSTMCell(trend)
    closeness = layers.Conv2D(64, (1,1), padding='same', activation='relu', use_bias=True)(closeness)
    period    = layers.Conv2D(64, (1,1), padding='same', activation='relu', use_bias=True)(period)
    trend     = layers.Conv2D(64, (1,1), padding='same', activation='relu', use_bias=True)(trend)
    res = layers.Add()([closeness,period,trend])
    res = layers.Conv2D(1, (1,1), padding='same', activation='relu', use_bias=True)(res)
    res = layers.Flatten(name='output')(res)
    return res