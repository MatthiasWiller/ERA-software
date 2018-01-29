#!/usr/bin/env python
# -*- coding: utf-8 -*-
# script to demonstrate usage of snowmodule 'TIndex'
# Alexander von Ramm 
# 07.12.2016

import matplotlib.pyplot as plt
import pickle
from snowModules import snowModule

data = pickle.load(open('03_aBUS/python/code_betz/ZugspitzBlatt1415.pkl', 'rb'), encoding='latin1')
tempThreshM = 4
tempThreshA = 4
tempTrans = 2
c_0 = 0.2

data = snowModule(data, [tempThreshM, tempThreshA, tempTrans, c_0], 'TIndex')
plt.figure()
plt.plot(data['swe_sim'])
plt.plot(data['swe'])
plt.show()
