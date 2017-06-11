import isipy

KFILENAME =  '/Users/sanjayk/Dropbox/Data sharing with UC Berkeley/expert/' + '00074_009_000606_SI.mat'
VFILENAME =  '/Users/sanjayk/Dropbox/Data sharing with UC Berkeley/expert/' + '00074_009_000606_EndoscopeImageMemory_0.mat'
VDIRECTORY = '/Users/sanjayk/Dropbox/Data sharing with UC Berkeley/expert/00074_009_000606/'

dataset = isipy.ISIDataset(KFILENAME, VFILENAME, VDIRECTORY)




