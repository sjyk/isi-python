import isipy
import matplotlib.pyplot as plt

KFILENAME =  '/Users/sanjayk/Dropbox/Data sharing with UC Berkeley/expert/' + '00074_015_000609_SI.mat'

#KFILENAME =  '/Users/sanjayk/Dropbox/Data sharing with UC Berkeley/novice1/' + '00023_009_000191_SI.mat'

VFILENAME =  '/Users/sanjayk/Dropbox/Data sharing with UC Berkeley/expert/' + '00074_009_000606_EndoscopeImageMemory_0.mat'
VDIRECTORY = '/Users/sanjayk/Dropbox/Data sharing with UC Berkeley/expert/00074_009_000606/'

#dataset = isipy.ISIDataset(KFILENAME, VFILENAME, VDIRECTORY)
dataset = isipy.ISIKinematicDataset(KFILENAME)

plt.figure(figsize=(10,80))

for i in range(8):
    plt.subplot(8, 1, i+1)
    
    plt.yticks([])
    if i < 7:
        plt.xticks([])

    plt.plot([d[i] for d in dataset.JointAngles_ECM])

plt.show()




