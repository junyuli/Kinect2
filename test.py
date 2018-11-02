import sys
sys.path.append('/data/home/jli819/Kinect2/')
from Modules.VideoProcessor import VideoProcessor

videofile = '/data/home/jli819/noise_analysis/MC9_1/Videos/6/0006_vid.mp4'
outDirectory = '/data/home/jli819/noise_analysis/MC9_1/'
remVideoDirectory = 'dropb:BioSci-McGrath/Apps/CichlidPiData/MC9_1/Videos/' 

obj = VideoProcessor(videofile, outDirectory, remVideoDirectory)
obj.filterHMM(plot=True)
