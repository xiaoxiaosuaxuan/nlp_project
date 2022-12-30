1. without random teacher-forcing : Dev acc: 71.51  Dev fscore(p/r/f): (81.80/73.10/77.20) 
   [['inform-终点名称-朝阳县农机销售有限公司导航'], ['inform-操作-取消', 'inform-对象-导航'], ['inform-操作-导航']]
   model_1.bin
2. teacher-forcing rate = 0.5  :    Dev acc: 70.73  Dev fscore(p/r/f): (84.88/72.58/78.25)
   [['inform-终点名称-朝阳县农机销售有限公司导航'], ['inform-操作-取消', 'inform-对象-导航'], ['inform-操作-导航']]
   model_2.bin 
3. teacher-forcing rate = 0.25  :   Dev acc: 71.51  Dev fscore(p/r/f): (81.32/74.45/77.74)
   [['inform-终点名称-朝阳县农机销售有限', 'inform-操作-公司', 'inform-操作-导航'], ['inform-操作-取消', 'inform-对象-导航'], ['inform-操作-导航']]
   model_3.bin