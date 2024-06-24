"""
Author: Eosandra Grund, 24.6.24
Changed this code from Source: https://matplotlib.org/stable/gallery/widgets/slider_demo.html
It depicts the behaviour of the lazy length cost pressure for the accuracy development and different message lengths
"""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.widgets import Button, Slider


# The parametrized function to be plotted
def separate(t, threshold, cost,length,l_threshold):
    """ returns the pressure value for all points"""
    pressure = t ** threshold * cost * length **l_threshold
    return pressure

def added(t, threshold, cost,length,l_threshold):
    """ returns the pressure value for all points added to the loss"""
    pressure = np.array(acc_data).T ** threshold * cost * length **l_threshold
    return pressure + loss_data

def something(t, threshold, cost,length,l_threshold):
    """ try harsh pressure, evtl need different hyperparameter"""
    #pressure = cost * (length-1) **l_threshold * (0.95) ** threshold
    pressure = np.where(t>0.95, cost * length ** l_threshold * (t - 0.95) ** threshold, 0)
    return pressure + loss_data # add loss here to stack pressure in picture on loss
# TODO What exactly is the influence of step loss and step loss pressure. 

#Change this area
#********************

# function to use
# added / separate / something
f = added

# Define initial parameters
init_threshold = 45
init_cost = 0.1
game = 3,4
length = 4
init_l_threshold = 1
lines = []

# End of Change Area
# ******************

# create arrays with loss points
data = [{"loss": 0.680751621723175, "acc": 0.5485671758651733, "length": 3.314653158187866, "mode": "train", "epoch": 1},{"loss": 0.5394455194473267, "acc": 0.7408324480056763, "length": 3.881948232650757, "mode": "train", "epoch": 2},{"loss": 0.4554688334465027, "acc": 0.7962578535079956, "length": 3.950700521469116, "mode": "train", "epoch": 3},{"loss": 0.39433369040489197, "acc": 0.8282887935638428, "length": 3.976868152618408, "mode": "train", "epoch": 4},{"loss": 0.35493752360343933, "acc": 0.845758855342865, "length": 3.988323211669922, "mode": "train", "epoch": 5},{"loss": 0.32317787408828735, "acc": 0.8611850142478943, "length": 3.991058826446533, "mode": "train", "epoch": 6},{"loss": 0.29906702041625977, "acc": 0.8723857402801514, "length": 3.993720293045044, "mode": "train", "epoch": 7},{"loss": 0.27281439304351807, "acc": 0.8841753602027893, "length": 3.995303153991699, "mode": "train", "epoch": 8},{"loss": 0.2625734806060791, "acc": 0.8944979310035706, "length": 3.99660062789917, "mode": "train", "epoch": 9},{"loss": 0.25509732961654663, "acc": 0.896294355392456, "length": 3.9971606731414795, "mode": "train", "epoch": 10},{"loss": 0.225555881857872, "acc": 0.9126116633415222, "length": 3.9977731704711914, "mode": "train", "epoch": 11},{"loss": 0.22699281573295593, "acc": 0.9137312173843384, "length": 3.9980454444885254, "mode": "train", "epoch": 12},{"loss": 0.20637620985507965, "acc": 0.9222429394721985, "length": 3.9982824325561523, "mode": "train", "epoch": 13},{"loss": 0.1966066062450409, "acc": 0.9270371198654175, "length": 3.9983103275299072, "mode": "train", "epoch": 14},{"loss": 0.18380340933799744, "acc": 0.9311637282371521, "length": 3.998699188232422, "mode": "train", "epoch": 15},{"loss": 0.17586366832256317, "acc": 0.9344054460525513, "length": 3.9986705780029297, "mode": "train", "epoch": 16},{"loss": 0.17076697945594788, "acc": 0.9364814758300781, "length": 3.998940944671631, "mode": "train", "epoch": 17},{"loss": 0.16127051413059235, "acc": 0.942894697189331, "length": 3.9989583492279053, "mode": "train", "epoch": 18},{"loss": 0.17612670361995697, "acc": 0.9370918869972229, "length": 3.999274969100952, "mode": "train", "epoch": 19},{"loss": 0.15072885155677795, "acc": 0.9467055797576904, "length": 3.99920916557312, "mode": "train", "epoch": 20},{"loss": 0.13978013396263123, "acc": 0.952268123626709, "length": 3.999204635620117, "mode": "train", "epoch": 21},{"loss": 0.16168124973773956, "acc": 0.9421738386154175, "length": 3.9993438720703125, "mode": "train", "epoch": 22},{"loss": 0.13579431176185608, "acc": 0.9514784812927246, "length": 3.9992611408233643, "mode": "train", "epoch": 23},{"loss": 0.12622514367103577, "acc": 0.9581165909767151, "length": 3.999229669570923, "mode": "train", "epoch": 24},{"loss": 0.11777788400650024, "acc": 0.9601405262947083, "length": 3.9995133876800537, "mode": "train", "epoch": 25},{"loss": 0.11635170131921768, "acc": 0.960307240486145, "length": 3.9994044303894043, "mode": "train", "epoch": 26},{"loss": 0.11189106851816177, "acc": 0.961249053478241, "length": 3.999553680419922, "mode": "train", "epoch": 27},{"loss": 0.10879890620708466, "acc": 0.9638320207595825, "length": 3.999575138092041, "mode": "train", "epoch": 28},{"loss": 0.10518777370452881, "acc": 0.9652239084243774, "length": 3.9996204376220703, "mode": "train", "epoch": 29},{"loss": 0.10173965990543365, "acc": 0.9650519490242004, "length": 3.999579906463623, "mode": "train", "epoch": 30},{"loss": 0.09666021913290024, "acc": 0.9681497812271118, "length": 3.999704599380493, "mode": "train", "epoch": 31},{"loss": 0.09081341326236725, "acc": 0.9706748127937317, "length": 3.9996910095214844, "mode": "train", "epoch": 32},{"loss": 0.08791577070951462, "acc": 0.9722950458526611, "length": 3.9997682571411133, "mode": "train", "epoch": 33},{"loss": 0.09235034137964249, "acc": 0.9698848128318787, "length": 3.9998178482055664, "mode": "train", "epoch": 34},{"loss": 0.08470503985881805, "acc": 0.9718125462532043, "length": 3.9997406005859375, "mode": "train", "epoch": 35},{"loss": 0.07714273780584335, "acc": 0.9763227701187134, "length": 3.99973464012146, "mode": "train", "epoch": 36},{"loss": 0.07679979503154755, "acc": 0.9766126871109009, "length": 3.9998788833618164, "mode": "train", "epoch": 37},{"loss": 0.07488251477479935, "acc": 0.9773197174072266, "length": 3.9998109340667725, "mode": "train", "epoch": 38},{"loss": 0.07447246462106705, "acc": 0.978116512298584, "length": 3.9998817443847656, "mode": "train", "epoch": 39},{"loss": 0.06763245165348053, "acc": 0.9791375994682312, "length": 3.999878168106079, "mode": "train", "epoch": 40},{"loss": 0.061947282403707504, "acc": 0.9812654852867126, "length": 3.999877691268921, "mode": "train", "epoch": 41},{"loss": 0.06648470461368561, "acc": 0.9806992411613464, "length": 3.9998958110809326, "mode": "train", "epoch": 42},{"loss": 0.06796997785568237, "acc": 0.979309618473053, "length": 3.9999096393585205, "mode": "train", "epoch": 43},{"loss": 0.06492703408002853, "acc": 0.979733943939209, "length": 3.999891996383667, "mode": "train", "epoch": 44},{"loss": 0.0772060826420784, "acc": 0.9755328893661499, "length": 3.999847412109375, "mode": "train", "epoch": 45},{"loss": 0.06648018211126328, "acc": 0.9807554483413696, "length": 3.9998934268951416, "mode": "train", "epoch": 46},{"loss": 0.05981585010886192, "acc": 0.9828566312789917, "length": 3.999922513961792, "mode": "train", "epoch": 47},{"loss": 0.06585778295993805, "acc": 0.9810407161712646, "length": 3.9999241828918457, "mode": "train", "epoch": 48},{"loss": 0.08030054718255997, "acc": 0.9753378629684448, "length": 3.9999489784240723, "mode": "train", "epoch": 49},{"loss": 0.06841021031141281, "acc": 0.9796232581138611, "length": 3.9999523162841797, "mode": "train", "epoch": 50}]
data_2 = [{"acc": 0.9862177968025208, "loss": 0.043772049248218536, "pressure": 0.12852364778518677, "length": 2.439239978790283, "mode": "train", "epoch": 185},{"acc": 0.9869731068611145, "loss": 0.04566303640604019, "pressure": 0.13741059601306915, "length": 2.407921314239502, "mode": "train", "epoch": 288}]
data.extend(data_2)
loss_data = []
acc_data = []
for dic in data:
    loss_data.append(dic["loss"])
    acc_data.append(dic["acc"])

t = np.linspace(0.0, 1, 1000) if f != added else np.array(acc_data)

# Create the figure and the line that we will manipulate
fig, ax = plt.subplots()
for l in range(length):
    lines.append(ax.plot(t, f(t, init_threshold, init_cost,l+1,init_l_threshold), lw=2,label=f"{l+1}"))
lines.append(ax.plot(t,f(t,init_threshold,init_cost,1.75,init_l_threshold),lw=2,label=f"{1.75}"))
ax.set_xlabel('accuracy')
ax.set_xlim([0.6,1.0])
ax.set_ylim([0.0,1.0])
#ax.scatter(acc_data,loss_data,alpha=0.5,label="Loss",s=2,c="purple") # scatter loss
ax.plot(acc_data,loss_data,label="Loss") # plot loss
ax.legend(title="Message Length",loc="best")

# adjust the main plot to make room for the sliders
fig.subplots_adjust(left=0.25, bottom=0.25, right=0.75)

# Make a horizontal slider to control the cost.
axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
freq_slider = Slider(
    ax=axfreq,
    label='cost',
    valmin=0.001,
    valmax=1,
    valinit=init_cost,
)

# Make a vertically oriented slider to control the threshold
axamp = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
amp_slider = Slider(
    ax=axamp,
    label="threshold",
    valmin=0,
    valmax=100,
    valinit=init_threshold,
    orientation="vertical"
)

# Make a vertically oriented slider to control the threshold
axl = fig.add_axes([0.9, 0.25, 0.0225, 0.63])
l_slider = Slider(
    ax=axl,
    label="length_threshold",
    valmin=1,
    valmax=5,
    valinit=init_l_threshold,
    orientation="vertical"
)


# The function to be called anytime a slider's value changes
def update(val):
    for l in range(length):
        lines[l][0].set_ydata(f(t, amp_slider.val, freq_slider.val,l+1,l_slider.val))
    lines[-1][0].set_ydata(f(t, amp_slider.val, freq_slider.val,1.75,l_slider.val))
    fig.canvas.draw_idle()


# register the update function with each slider
freq_slider.on_changed(update)
amp_slider.on_changed(update)
l_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')


def reset(event):
    freq_slider.reset()
    amp_slider.reset()
button.on_clicked(reset)

plt.show()