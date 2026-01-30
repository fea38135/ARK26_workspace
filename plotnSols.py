from Robot import *
import matplotlib.pyplot as plt

robot_select = input("select robot: \n 0 for Robot T (Transpressor, Thor)-like or \n 1 for Robot F (Fanuc CRX)\n")

if robot_select == '0':
    nSols_tp = np.load('data/nSols_robot_t.npy')
    a2 = 540
    a5 = 150
    d1 = 0
    d4 = 540
    bot = Transpressor(a2,a5,d1,d4, nSols=nSols_tp)

elif robot_select == '1':
    nSols_fanuc = np.load('data/nSols_fanuc.npy')
    bot = FanucCR(nSols=nSols_fanuc)


plot_select = input("select plot: \n 0 for plotting positional and orientational slices or \n 1 for also showing the robot configurations \n")

if plot_select == '0':
    bot.plotnSols()

elif plot_select == '1':
    bot.plotIKnSols(circ_res=500)

