# -*- coding: utf-8 -*-
"""
Created on Tue May 15 17:01:17 2018

@author: lab
"""


import numpy as np


def osa_analys1(amp, wav):

    chan_list =  [1546.04, 1548.02, 1548.82, 1550.00, 
                  1550.42, 1550.84, 1551.75, 1552.04, 1552.46,
                  1552.84, 1553.64, 1554.04, 1554.42, 
                  1554.86, 1555.66, 1556.46, 1557.28,
                  1558.08, 1558.50, 1560.12, 1560.52]
    
    
    amp_ref = np.sort(amp)[len(amp)/10] + 15.0
    amp_chan = [None]*len(chan_list)
    
    for ii, chan in enumerate(chan_list):
        ind = np.argmin(abs(wav - chan))
        amp_chan[ii] = amp[ind]
    
    sig_chan = amp_chan > amp_ref
    return sig_chan, amp_chan 

#while(1):
#    a.sweep_and_get_the_trace()
#    # obtain spectrum occupancy
#    a.process_the_data(stp,endp,ptr,-60,0)
#    time.sleep(1)
    
#def test_lambda_occupy(amp1,ptr,threshold):
#    trace = a.sweep_and_get_the_trace()
#    channel = np.zeros(39)
#    for i in range(0,38):
#        if np.mean(amp1[51*i:51*i+15]>threshold):
#            channel[i] = 1;
#    return channel