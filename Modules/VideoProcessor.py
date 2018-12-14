import pims, math, psutil, shutil, os, datetime, subprocess, sys
import numpy as np
import scipy.ndimage
from hmmlearn import hmm
import matplotlib.pyplot as plt
from multiprocessing.dummy import Pool as ThreadPool
from Modules.HMM_data import HMMdata
from PIL import Image
import pickle
from sklearn.cluster import DBSCAN
from sklearn.neighbors import radius_neighbors_graph
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import glob 
#from pathos.multiprocessing import ProcessingPool as ThreadPool
np.warnings.filterwarnings('ignore')
import pdb

class VideoProcessor:
    # This class takes in an mp4 videofile and an output directory and performs the following analysis on it:
    # 1. Performs HMM analysis on all pixel files
    # 3. Clusters HMM data to identify potential spits and scoops
    # 4. Uses DeepLabCut to annotate fish

    #Parameters - blocksize, 
    def __init__(self, videofile, outDirectory, remVideoDirectory, rewrite = False):

        # Store arguments
        self.videofile = videofile
        self.baseName = self.videofile.split('/')[-1].split('.')[0]
        self.outDirectory = outDirectory if outDirectory[-1] == '/' else outDirectory + '/'
        self.remVideoDirectory = remVideoDirectory if remVideoDirectory[-1] == '/' else remVideoDirectory + '/'
        
        self.rewrite = rewrite
        
        # Set paramaters
        self.cores = psutil.cpu_count() # Number of cores that should be used to analyze the video

        # Set directories and make sure they exist
        os.makedirs(self.outDirectory) if not os.path.exists(self.outDirectory) else None

        self.hmmFile = self.outDirectory + self.baseName + '.hmm.npy'
        
        self.clusterDirectory = self.outDirectory + 'ClusterData/'
        os.makedirs(self.clusterDirectory) if not os.path.exists(self.clusterDirectory) else None
        self.annotationDirectory = self.outDirectory + 'AnnotationData/'
        os.makedirs(self.annotationDirectory) if not os.path.exists(self.annotationDirectory) else None
        self.exampleDirectory = self.outDirectory + 'ExampleData/'
        os.makedirs(self.exampleDirectory) if not os.path.exists(self.exampleDirectory) else None

        self.tempDirectory = self.outDirectory + 'Temp/'     
        shutil.rmtree(self.tempDirectory) if os.path.exists(self.tempDirectory) else None
        os.makedirs(self.tempDirectory)
      
        self.window = 120
        self.hmm_time = 60*60
        print('VideoProcessor: Analyzing ' + self.videofile, file = sys.stderr)

        # For redirecting stderr to null
        self.fnull = open(os.devnull, 'w')

        

    def downloadVideo(self):
        videoName = self.videofile.split('/')[-1]
        localDirectory = self.videofile.replace(videoName,'')
        if os.path.isfile(self.videofile):
            return
        self._print(self.videofile + ' not present in local path. Trying to find it remotely')
        subprocess.call(['rclone', 'copy', self.remVideoDirectory + videoName, localDirectory], stderr = self.fnull)                
        if not os.path.isfile(self.videofile):
            self._print(self.videofile + ' not present in remote path. Trying to find h264 file and convert it to mp4')
            if not os.path.isfile(self.videofile.replace('.mp4', '.h264')):
                subprocess.call(['rclone', 'copy', self.remVideoDirectory + videoName.replace('.mp4', '.h264'), localDirectory], stderr = self.fnull)
            if not os.path.isfile(self.videofile.replace('.mp4', '.h264')):
                self._print('Unable to find ' + self.remVideoDirectory + videoName.replace('.mp4', '.h264'))
                raise Exception
            
            subprocess.call(['ffmpeg', '-i', self.videofile.replace('.mp4', '.h264'), '-c:v', 'copy', self.videofile])
                
            if os.stat(self.videofile).st_size >= os.stat(self.videofile.replace('.mp4', '.h264')).st_size:
                try:
                    vid = pims.Video(self.videofile)
                    vid.close()
                    os.remove(self.videofile.replace('.mp4', '.h264'))
                except Exception as e:
                    self._print(e)
                    self._print('Unable to convert ' + self.videofile)
                    raise Exception
                subprocess.call(['rclone', 'copy', self.videofile, self.remVideoDirectory], stderr = self.fnull)
            self._print(self.videofile + ' converted and uploaded to ' + self.remVideoDirectory)

        #Grab info on video
        cap = pims.Video(self.videofile)
        self.height = int(cap.frame_shape[0])
        self.width = int(cap.frame_shape[1])
        self.frame_rate = int(cap.frame_rate)
        try:
            self.frames = min(int(cap.get_metadata()['duration']*cap.frame_rate), 12*60*60*self.frame_rate)
        except AttributeError:
            self.frames = min(int(cap.duration*cap.frame_rate), 12*60*60*self.frame_rate)
        cap.close()
 
    @profile
    def plotBrightnessOverTime(self, window=60):
       
        video = cv2.VideoCapture(videofile)
        os.makedirs(self.outDirectory + 'meanBright/', exist_ok=True)
        window = window * self.frame_rate
        frameN = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        windowBright = []

        while video.get(cv2.CAP_PROP_POS_FRAMES) < frameN:
            startFrame = video.get(cv2.CAP_PROP_POS_FRAMES)
            frameBright = np.zeros(window)
            for fi in range(0,window):
                if video.get(cv2.CAP_PROP_POS_FRAMES) >= frameN:
                    break
                ret, frame = video.read()
                frameBright[fi] = np.mean([frame.flatten()])

            np.save(self.outDirectory + 'MeanBright/startFrame_' + str(startFrame) + '_mean_brightness.npy',frameBright)
            print('saved '+ self.outDirectory + 'MeanBright/startFrame_' +str(startFrame) + '_mean_brightness.npy')

            windowBright.append(np.mean(frameBright))

        video.release()
        np.save(self.outDirectory + 'MeanBrightPer' + str(window) + 's.npy',windowBright)
        plt.figure()
        plt.plot(windowBright)
        plt.savefig(self.outDirectory + 'MeanBrightPer' + str(window) + 's.pdf', bbox_inches='tight')

    @profile
    def calculateHMM(self, blocksize = 5*60, delete = True):
        """
        This functon decompresses video into smaller chunks of data formated in the numpy array format.
        Each numpy array contains one row of data for the entire video.
        This function then smoothes the raw data
        Finally, an HMM is fit to the data and an HMMobject is created
        """
        print(self.hmmFile)
        print(os.path.exists(self.hmmFile))
        if os.path.exists(self.hmmFile) and not self.rewrite:
            print('Hmmfile already exists. Will not recalculate it unless rewrite flag is True')
            return

        self.downloadVideo()
        
        self.blocksize = blocksize
        total_blocks = math.ceil(self.frames/(blocksize*self.frame_rate)) #Number of blocks that need to be analyzed for the full video

        # Step 1: Convert mp4 to npy files for each row
        pool = ThreadPool(self.cores) #Create pool of threads for parallel analysis of data
        start = datetime.datetime.now()
        print('calculateHMM: Converting video into HMM data', file = sys.stderr)
        print('TotalBlocks: ' + str(total_blocks), file = sys.stderr)
        print('TotalThreads: ' + str(self.cores), file = sys.stderr)
        print('Video processed: ' + str(self.blocksize/60) + ' min per block, ' + str(self.blocksize/60*self.cores) + ' min per cycle', file = sys.stderr)
        print('Converting mp4 data to npy arrays at 1 fps', file = sys.stderr)
        print('StartTime: ' + str(start), file = sys.stderr)
        
        for i in range(0, math.ceil(total_blocks/self.cores)):
            blocks = list(range(i*self.cores, min(i*self.cores + self.cores, total_blocks)))
            print('Seconds since start: ' + str((datetime.datetime.now() - start).seconds) + ', Processing blocks: ' + str(blocks[0]) + ' to ' +  str(blocks[-1]))
            results = pool.map(self._readBlock, blocks)
            print('Data read: ' + str((datetime.datetime.now() - start).seconds) + ' seconds')
            for row in range(self.height):
                row_file = self._row_fn(row)
                out_data = np.concatenate([results[x][row] for x in range(len(results))], axis = 1)
                if os.path.isfile(row_file):
                    out_data = np.concatenate([np.load(row_file),out_data], axis = 1)
                np.save(row_file, out_data)
            print('Data wrote: ' + str((datetime.datetime.now() - start).seconds) + ' seconds', file = sys.stderr)
        pool.close() 
        pool.join() 
        print('TotalTime: ' + str((datetime.datetime.now() - start).seconds/60) + ' minutes', file = sys.stderr)

        # Step 2: Smooth data to remove outliers
        pool = ThreadPool(self.cores)
        start = datetime.datetime.now()
        print('Smoothing data to filter out outliers', file = sys.stderr)
        print('StartTime: ' + str(start), file = sys.stderr)
        for i in range(0, self.height, self.cores):
            rows = list(range(i, min(i + self.cores, self.height)))
            print('Seconds since start: ' + str((datetime.datetime.now() - start).seconds) + ' seconds, Processing rows: ' + str(rows[0]) + ' to ' +  str(rows[-1]), file = sys.stderr)
            results = pool.map(self._smoothRow, rows)
        print('TotalTime: Took ' + str((datetime.datetime.now() - start).seconds/60) + ' minutes to smooth ' + str(self.height) + ' rows')
        pool.close() 
        pool.join()

        # Step 3: Calculate HMM values for each row
        pool = ThreadPool(self.cores)
        start = datetime.datetime.now()
        print('Calculating HMMs for all data', file = sys.stderr)
        print('StartTime: ' + str(start), file = sys.stderr)
        for i in range(0, self.height, self.cores):
            rows = list(range(i, min(i + self.cores, self.height)))
            print('Seconds since start: ' + str((datetime.datetime.now() - start).seconds) + ' seconds, Processing rows: ' + str(rows[0]) + ' to ' +  str(rows[-1]), file = sys.stderr)
            results = pool.map(self._hmmRow, rows)
        print('TotalTime: Took ' + str((datetime.datetime.now() - start).seconds/60) + ' minutes to calculate HMMs for ' + str(self.height) + ' rows', file = sys.stderr)
        pool.close() 
        pool.join()

        # Step 4: Create HMM object and delete temporary data if necessary
        start = datetime.datetime.now()
        if delete:
            print('Converting HMMs to internal data structure and deleting temporary data', file = sys.stderr)
        else:
            print('Converting HMMs to internal data structure and keeping temporary data', file = sys.stderr)

        print('StartTime: ' + str(start), file = sys.stderr)
        
        self.obj = HMMdata(self.width, self.height, self.frames, self.frame_rate)
        self.obj.add_data(self.tempDirectory, self.hmmFile)
        # Copy example data to directory containing videofile
        subprocess.call(['cp', self._row_fn(int(self.height/2)), self._row_fn(int(self.height/2)).replace('.npy', '.smoothed.npy'), self._row_fn(int(self.height/2)).replace('.npy', '.hmm.npy'), self.exampleDirectory])

        if delete:
            shutil.rmtree(self.tempDirectory)
        print('Took ' + str((datetime.datetime.now() - start).seconds/60) + ' convert HMMs', file = sys.stderr)

    def filterHMM(self, minMagnitude = 10, nlargest = 1, timepoint = None, threshold = None, mask = None, plot = True, write = True):
        # minMagnitude: integer. HMM changes with a smaller magnitude will be removed. 
        # nlargest: integer. Timepoints will be sorted by the number of HMM changes, HMM changes at the largest n timepoints will be removed.  
        # timepoint: a list of integer. HMM changes at the n-th second will be removed.
        # threshold: integer. If at anytimpoint, the number of HMM changes exceeds n, then all HMM change at this timepoint will be removed. 
        # mask: path to a tank mask image. HMM changes within the 'True' region of the mask will be removed
        # 
        # Plot: generate a pdf file of HMM statistics
        
        if os.path.exists(self.clusterDirectory + 'FilteredCoords.npy') and write and not self.rewrite:
            print('already exists filteredCoord, not rewriting')
            self.coords = np.load(self.clusterDirectory + 'FilteredCoords.npy')
            return
        if os.path.isfile(self.clusterDirectory + 'RawCoords.npy'):
            self.coords = np.load(self.clusterDirectory + 'RawCoords.npy')
        else:
            try:
                self.obj
            except AttributeError:
                self.obj = HMMdata(filename = self.hmmFile)
            
            self.coords = self.obj.retDBScanMatrix()
            np.save(self.clusterDirectory + 'RawCoords.npy', self.coords)

        rawCount = self.coords.shape[0]
        print('HMM change count: '+ str(rawCount))

        if minMagnitude:           
            rowsDelete = np.where(self.coords[:,3] < minMagnitude)[0]
            print('Filtered on minMagnitude. Total removed:' + str(rowsDelete.shape[0]))

        if nlargest:
            count = Counter(self.coords[:,0])
            timeDelete = count.most_common(nlargest)
            rowsDelete = np.concatenate((rowsDelete, np.where(self.coords[:,0] == timeDelete)[0]))
            print('Filtered on nlargest. Total removed:' + str(rowsDelete.shape[0]))

        if timepoint:
            for t in timepoint:
                rowsDelete = np.concatenate((rowsDelete, np.where(self.coords[:,0] == t)[0]))
            print('Filtered on timepoint. Total removed:' + str(rowsDelete.shape[0]))

        if threshold:
            rowsDelete = np.concatenate((rowsDelete, np.where(self.coords[:,0] >= threshold)[0]))
            print('Filtered on threshold. Total removed:' + str(rowsDelete.shape[0]))
       
        filteredCoords = np.delete(self.coords,np.unique(rowsDelete),0)

        # tank is the non-zero(white) part of mask
        if mask:
            maskImg = np.array(Image.open(mask))
            img = np.zeros([972,1296])
            img[filteredCoords[:,1],filteredCoords[:,2]] = True
            img = np.logical_and(img,~maskImg)
            filteredCoords = np.where(img==True)[0]
            print('After masking, remain count: '+str(filteredCoords.shape[0]))

        if plot:
            tempCoordsFile = self.clusterDirectory + 'tempFilteredCoords'+ str(datetime.datetime.now().timestamp())+'.npy'
            np.save(tempCoordsFile, filteredCoords)
            self.hmmStat(tempCoordsFile)

        if write:
            self.coords = filteredCoords
            np.save(self.clusterDirectory + 'FilteredCoords.npy', filteredCoords)

    def hmmStat(self,coordsFile):

        coords = np.load(coordsFile) 
        fig, axs = plt.subplots(2, 2, figsize=(10, 8), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace = 0.2, wspace=.001)
        axs = axs.ravel()
        hmmTimeHr = self._int(np.floor(coords[:,0]/3600))
        
        axs[0].hist(hmmTimeHr, bins = np.arange(np.min(hmmTimeHr),np.max(hmmTimeHr)+1,0.5))
        axs[0].set_title('timepoint of hmm changes')
        
        axs[1].hist(coords[:,3], bins = np.unique(coords[:,3]))
        axs[1].set_title('magnitude of hmm changes')

        axs[2].hist(coords[:,2], bins = np.unique(coords[:,1]))
        axs[2].set_title('x coordinate of hmm changes')    
        
        axs[3].hist(coords[:,1], bins = np.unique(coords[:,2]))
        axs[3].set_title('y coordinate of hmm changes') 

        fig.savefig(coordsFile.replace('.npy', '_HMMStat.pdf'), bbox_inches='tight')

    @profile
    def clusterHMM(self, treeR = 22, leafNum = 190, neighborR = 22, timeScale = 10, eps = 18, minPts = 170):

        if os.path.exists(self.clusterDirectory + 'Labels.npy') and not self.rewrite:
            print('Cluster label file already exists. Will not recalculate it unless rewrite flag is True')
            
        else:
            try:
                self.obj
            except AttributeError:
                self.obj = HMMdata(filename = self.hmmFile)

            print('Identifying raw coordinate positions for cluster analysis', file = sys.stderr)
            if os.path.isfile(self.clusterDirectory + 'FilteredCoords.npy'):
                self.coords = np.load(self.clusterDirectory + 'FilteredCoords.npy')
                print('self.coords size: '+str(sys.getsizeof(self.coords)))
                print('self.coords count: '+str(self.coords.shape[0]))
            else:
                #self.coords = self.obj.retDBScanMatrix(minMagnitude)
                #np.save(self.clusterDirectory + 'FilteredCoords.npy', self.coords)
                print('filter HMM first')
                return 
            
            print('Calculating nearest neighbors and pairwise distances between clusters', file = sys.stderr)

            if os.path.isfile(self.clusterDirectory + 'PairwiseDistances.npz'):
                dist = np.load(self.clusterDirectory + 'PairwiseDistances.npz')
            else:
                self.coords[:,0] = self.coords[:,0].astype(np.float64) * timeScale 
                if os.path.isfile(self.clusterDirectory + 'NearestNeighborTree'):
                    X = pickle.load(open(self.clusterDirectory + 'NearestNeighborTree.pkl', 'rb'))
                else:
                    X = NearestNeighbors(radius=treeR, metric='minkowski', p=2, algorithm='kd_tree',leaf_size=leafNum,n_jobs=24).fit(self.coords)
                    pickle.dump(X, open(self.clusterDirectory + 'NearestNeighborTree.pkl', 'wb'))
                
                dist = X.radius_neighbors_graph(self.coords, neighborR, 'distance')
                print('dist size: '+ str(dist.data.nbytes))
                scipy.sparse.save_npz(self.clusterDirectory + 'PairwiseDistances.npz', dist, compressed=False)
           
            self.labels = DBSCAN(eps=eps, min_samples=minPts, metric='precomputed', n_jobs=1).fit_predict(dist)

            np.save(self.clusterDirectory + 'Labels.npy', self.labels)

        #if os.path.exists(self.clusterDirectory + 'ClusterCenters.npy') and not self.rewrite:
        #    print('Cluster centers file already exists. Will not recalculate it unless rewrite flag is True')
        #    return

        self.coords = np.load(self.clusterDirectory + 'FilteredCoords.npy')
        self.labels = np.load(self.clusterDirectory + 'Labels.npy')

        # calculate center z, y, x, number points for each cluster
        clusterDataFile = self.clusterDirectory + 'ClusterCenters.npy'
        if os.path.isfile(clusterDataFile):
            self.clusterData = np.load(clusterDataFile)
        else:
            uniqueLabel = set(self.labels[self.labels!=-1])
            self.clusterData = np.zeros([len(uniqueLabel),7])  # clusterData z,y,x, max z,y,x, min z,y,x
            for l in uniqueLabel:
                currCluster = self.coords[self.labels==l,:]
                self.clusterData[l,0:3] = np.mean(currCluster[:,0:3],axis=0)  # center coordinate of cluster
                self.clusterData[l,3] = currCluster.shape[0]  # sand change count
                self.clusterData[l,4] = np.max(currCluster[:,0]) - np.min(currCluster[:,0])
                self.clusterData[l,5] = np.max(currCluster[:,1]) - np.min(currCluster[:,1])  # span of y coordinates
                self.clusterData[l,6] = np.max(currCluster[:,2]) - np.min(currCluster[:,2]) 

            np.save(clusterDataFile, self.clusterData)


        if os.path.exists(self.clusterDirectory + 'ClusterStat.pdf') and not self.rewrite:
            print('Cluster stats exists.')
        else:
            self.clusterStat()

    def clusterStat(self, interval = 2):
        try:
            self.clusterData
        except AttributeError:
            self.clusterData = np.load(self.clusterDirectory + 'ClusterCenters.npy')
        # e.g. change at 45 min counted as change in the 0st hr
        clusterTimeHr = self._int(np.floor(self.clusterData[:,0]/3600))
        # One histogram and a number of progression plots depending on timespan
        numPlot = (np.max(clusterTimeHr) - np.min(clusterTimeHr)) / float(interval) +1
        rowNum = int(np.ceil(numPlot/2))
        fig, axs = plt.subplots(rowNum, 2, figsize=(15, 30), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace = 0.2, wspace=.001)
        axs = axs.ravel()
       
        # histogram of timepoint of cluster centers
        axs[0].hist(clusterTimeHr, bins=np.arange(np.min(clusterTimeHr),np.max(clusterTimeHr)+1,0.5))
        axs[0].set_title('timepoint of cluster centers')       

        i = 1

        for startTime in range(np.min(clusterTimeHr),np.max(clusterTimeHr),interval):
         
            endTime = min(startTime + interval, np.max(clusterTimeHr)+1)
            points = self.clusterData[np.logical_and(clusterTimeHr >= startTime, clusterTimeHr < endTime),:]
            
            axs[i].scatter(points[:,2],points[:,1],s=10,alpha=0.7)

            axs[i].set_ylim([972,0])
            axs[i].set_xlim([0,1296])

            axs[i].set_title(str(startTime)+ 'hr to ' + str(endTime) + 'hr')

            i+=1

        fig.savefig(self.clusterDirectory +'ClusterStat.pdf', bbox_inches='tight')

    def createClusterClipsToAnnotate(self, n = 100, length = 6, size = 400):
        # n: number of clips
        # size: height of window
       
        # check if cluster results exist, and construct object from clusterHMM
    
        if glob.glob(self.clusterDirectory + 'ClusterClipsToAnnotate/*') and not self.rewrite:
            print('Clips already exist. Will not redo unless rewrite flag is True')
        if not os.path.exists(self.clusterDirectory + 'Labels.npy') or not os.path.exists(self.clusterDirectory + 'ClusterCenters.npy'):
            print('run clusterHMM first')
        
        self.coords = np.load(self.clusterDirectory + 'FilteredCoords.npy')
        self.labels = np.load(self.clusterDirectory + 'Labels.npy')
        self.clusterData = np.load(self.clusterDirectory + 'ClusterCenters.npy')

        rgbVideo = cv2.VideoCapture(self.videofile)
        os.makedirs(self.clusterDirectory + 'ClusterClipsToAnnotate/', exist_ok=True)

        for l in range(0,n):
            z = self.clusterData[i,0]
            out = cv2.VideoWriter(self.clusterDirectory + 'ClusterClipsToAnnotate/' + str(l) + '.mp4', 0x00000021, 25, (size*2,size))

            hmmStartZ = z - self._int(length/2)
            hmmEndZ = z + self._int(length/2)
            rgbVideo.set(cv2.CAP_PROP_POS_FRAMES,hmmStartZ * self.frame_rate)

            for hmmZ in range(hmmStartZ,hmmEndZ+1):
                hmmFrame = _create_hmm_frame(hmmZ,i,self.coords,self.labels)
                hmmFrame = _mark_rectangle(hmmFrame,self.clusterData[l],l,(0,175,255))
                # make a marker when time is the center time of cluster
                hmmFrame = _mark_center(hmmFrame,hmmZ,self.clusterData[l])

                hmmFrame,centerCoord = _fill_frame_edge(hmmFrame,self.clusterData[l],size)
                subhmmFrame = _cut_frame(hmmFrame, centerCoord[1],centerCoord[2],size,size)

                # go to the rgb video timpepoint
                for t in range(0,hmmFrameblock):
                    ret, rgbFrame = rgbVideo.read()
                    rgbFrame = _mark_rectangle(rgbFrame,clusterCenter[l],l,(0,175,255))
                    rgbFrame = _mark_center(rgbFrame,hmmZ,clusterCenter[l])

                    # mirror replicate the edges to prevent out of boundaries
                    rgbFrame,centerCoord = _fill_frame_edge(rgbFrame,clusterCenter[l],size)
                    subrgbFrame = _cut_frame(rgbFrame, centerCoord[1],centerCoord[2],size,size)
                    outFrame = np.concatenate((subhmmFrame,subrgbFrame),axis=1)
                    out.write(outFrame)

            out.release()

        
    def createFramesToAnnotate(self, n = 300):
        rerun = False
        for i in range(n):
            if not os.path.isfile(self.annotationDirectory + 'AnnotateImage' + str(i).zfill(4) + '.jpg'):
                rerun = True
                break
        if not rerun:
            self._print('AnnotationFrames already created... skipping')
            return
        self.downloadVideo()
        cap = pims.Video(self.videofile)
        counter = 0
        for i in [int(x) for x in np.linspace(1.25*3600*self.frame_rate, self.frames - 1.25*3600*self.frame_rate, n)]:
            frame = cap[i]
            t_image = Image.fromarray(frame)
            t_image.save(self.annotationDirectory + 'AnnotateImage' + str(counter).zfill(4) + '.jpg')
            counter += 1
        
    def summarize_data(self):
        try:
            self.obj
        except AttributeError:
            self.obj = HMMdata(self.width, self.height, self.frames)
            self.obj.read_data(self.outdir)

        t_hours = int(self.frames/(self.frame_rate*60*60))
        rel_diff = np.zeros(shape = (t_hours, self.height, self.width), dtype = 'uint8')
        
        for i in range(1,t_hours+1):
            rel_diff[i-1] = self.obj.ret_difference((i-1)*60*60*25,i*60*60*25 - 1)

        return rel_diff
    
    def _readBlock(self, block):
        min_t = block*self.blocksize
        max_t = min((block+1)*self.blocksize, int(self.frames/self.frame_rate))
        ad = np.empty(shape = (self.height, self.width, max_t - min_t), dtype = 'uint8')
        cap = pims.Video(self.videofile)
        counter = 0
        for i in range(min_t, max_t):
            current_frame = i*self.frame_rate
            frame = cap[current_frame]
            ad[:,:,counter] =  0.2125 * frame[:,:,0] + 0.7154 * frame[:,:,1] + 0.0721 * frame[:,:,2]
            counter += 1
        return ad

    def _smoothRow(self, row, seconds_to_change = 60*30, non_transition_bins = 2, std = 100):

        ad = np.load(self._row_fn(row))
        original_shape = ad.shape

        ad[ad == 0] = 1 # 0 used for bad data to save space and use uint8 for storing data (np.nan must be a float)

        # Calculate means
        lrm = scipy.ndimage.filters.uniform_filter(ad, size = (1,self.window), mode = 'reflect', origin = -1*int(self.window/2)).astype('uint8')
        rrm = np.roll(lrm, int(self.window), axis = 1).astype('uint8')
        rrm[:,0:self.window] = lrm[:,0:1]

        # Identify data that falls outside of mean
        ad[(((ad > lrm + 7.5) & (ad > rrm + 7.5)) | ((ad < lrm - 7.5) & (ad < rrm - 7.5)))] = 0
        del lrm, rrm

        # Interpolation missing data for HMM
        ad = ad.ravel(order = 'C') #np.interp requires flattend data
        nans, x = ad==0, lambda z: z.nonzero()[0]
        ad[nans]= np.interp(x(nans), x(~nans), ad[~nans])
        del nans, x

        # Reshape array to save it
        ad = np.reshape(ad, newshape = original_shape, order = 'C').astype('uint8')
        np.save(self._row_fn(row).replace('.npy', '.smoothed.npy'), ad)
        
        return True

    def _hmmRow(self, row, seconds_to_change = 60*30, non_transition_bins = 2, std = 100, hmm_window = 60):

        data = np.load(self._row_fn(row).replace('.npy', '.smoothed.npy'))
        zs = np.zeros(shape = data.shape, dtype = 'uint8')
        for i, column in enumerate(data):

            means = scipy.ndimage.filters.uniform_filter(column, size = hmm_window, mode = 'reflect').astype('uint8')
            freq, bins = np.histogram(means, bins = range(0,257,2))
            states = bins[0:-1][freq > hmm_window]
            comp = len(states)
            if comp == 0:
                print('For row ' + str(row) + ' and column ' + str(i) + ', states = ' + str(states))
                states = [125]
            model = hmm.GaussianHMM(n_components=comp, covariance_type="spherical")
            model.startprob_ = np.array(comp*[1/comp])
            change = 1/(seconds_to_change)
            trans = np.zeros(shape = (len(states),len(states)))
            for k,state in enumerate(states):
                s_trans = np.zeros(shape = states.shape)
                n_trans_states = np.count_nonzero((states > state + non_transition_bins) | (states < state - non_transition_bins))
                if n_trans_states != 0:
                    s_trans[(states > state + non_transition_bins) | (states < state - non_transition_bins)] = change/n_trans_states
                    s_trans[states == state] = 1 - change
                else:
                    s_trans[states == state] = 1
                trans[k] = s_trans
                   
            model.transmat_ = np.array(trans)
            model.means_ = states.reshape(-1,1)
            model.covars_ = np.array(comp*[std])
            
            z = [model.means_[x][0] for x in model.predict(column.reshape(-1,1))]
            zs[i,:] = np.array(z).astype('uint8')
        np.save(self._row_fn(row).replace('.npy', '.hmm.npy'), zs)

        return True

    def _row_fn(self, row):
        return self.tempDirectory + str(row) + '.npy'

    def _print(self, outtext):
        #       now = datetime.datetime.now()
        #        print(str(now) + ': ' + outtext, file = self.anLF)
        print(outtext, file = sys.stderr)

    def _int(self,n):
        return np.round(n).astype(int)

    def _fill_frame_edge(self,frame,coord,padSize):
        padFrame = np.stack([np.pad(frame[:,:,c], (padSize,), mode='constant',constant_values= 0) for c in range(3)], axis=2)
        # adjust coord after edge padding 
        ajustCoord = coord.copy()
        ajustCoord[1:3] = coord[1:3] + padSize

        return padFrame,ajustCoord

    def _mark_rectangle(self,frame,box,label,markColor):
        cv2.rectangle(frame,(box[8],box[7]),(box[5],box[4]),markColor,2)
        changeCount = box[9]
        # put change count on the box
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (box[8],box[7])
        fontScale              = 1
        fontColor              = markColor
        lineType               = 2

        #cv2.putText(frame, str(changeCount), bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
        cv2.putText(frame, str(label), bottomLeftCornerOfText, font, fontScale,fontColor,lineType)

        return frame

    def _mark_center(self, frame, t, centerCoord):
        if t == centerCoord[0]:
            cv2.drawMarker(frame,(centerCoord[2],centerCoord[1]),(114,0,225),cv2.MARKER_TRIANGLE_UP,8,2,8)
        return frame     
    
    def _cut_frame(self,frame,centerY,centerX,height,width):
        return frame[self._int(centerY)-self._int(height/2):self._int(centerY)+self._int(height/2),self._int(centerX)-self._int(width/2):self,_int(centerX)+self._int(width/2),:]

    def _create_hmm_frame(self,z,currLabel):
        # only one cluster is colored, others are white
        hmmFrame = np.uint8(np.zeros([self.height,self.width,3]))
        currClusterCoord = self.coords[self.labels == currLabel,:]
        # all other hmm points on this frame
        if np.any(self.coords[:,0]==z):
            [y,x] = self.coords[self.coords[:,0]==z,1:3].T
            hmmFrame[y,x,:] = [255,255,255]
        # current cluster on this frame
        if np.any(currClusterCoord[:,0]==z):
            [y,x] = currClusterCoord[currClusterCoord[:,0]==z,1:3].T
            hmmFrame[y,x,:] = [0,175,255]

        return hmmFrame



print(" ")
