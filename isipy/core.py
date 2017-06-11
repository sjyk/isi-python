import scipy.io
import numpy as np
import cv2

"""
Python Interface to the ISI Da Vinci API
"""


class ISIRecord(object):

    """
    Constructs a structured ISI record from the
    raw matlab output
    """
    def __init__(self, raw):
        
        #seperates the header from the data
        self.header, self.data = raw

        #internal store of the data
        self.processed_data = {}


        #handles alternate format
        if 'JointAngles' in self.data.dtype.names:
            return self.__init_alternate__(raw)


        #iterate through all of the fields
        for index, attr in  enumerate(self.data.dtype.names):
            
            #bunch of special cases for processing
            if 'Pose_Base' in attr:
                
                self.processed_data[attr] = self._3toPose(self.data[0][0][index])

            elif 'Pose_ECM' in attr or 'Pose_Workplace' in attr:

                self.processed_data[attr] = self._12toPose(self.data[0][0][index])

            elif 'Pose_PSM' in attr:

                for arm in range(3):

                    self.processed_data[attr+str((arm+1))] = self._12toPose(self.data[0][0][index][arm*12:(arm+1)*12])

            elif 'Pose_RCM' in attr or 'Pose_Mount' in attr:

                for arm in range(4):

                    if arm > 0:
                        self.processed_data[attr+"_PSM"+str((arm))] = self._12toPose(self.data[0][0][index][arm*12:(arm+1)*12])
                    else:
                        self.processed_data[attr+"_ECM"] = self._12toPose(self.data[0][0][index][arm*12:(arm+1)*12])


            else:
                self.processed_data[attr] = np.squeeze(self.data[0][0][index])


    #some helper methods
    def _3toPose(self, array):
        pose_matrix = np.zeros((4,4))
        pose_matrix[3,3] = 1
        pose_matrix[0:3,3] = array.reshape((3,))
        return pose_matrix

    #some helper methods
    def _12toPose(self, array):
        pose_matrix = np.zeros((4,4))
        pose_matrix[3,3] = 1
        pose_matrix[0:3,0:4] = array.reshape((3,4))
        return pose_matrix


        """
        populate the same data structure with the alternate format
        """
    def __init_alternate__(self, raw):
        #seperates the header from the data
        self.header, self.data = raw

        #internal store of the data
        self.processed_data = {}

        pose_headers = ['Base', 'ECM', 'PSM', 'RCM', 'Mount', 'Workplace']
        joint_headers = ['ECM', 'PSM1', 'PSM2', 'PSM3']

        #iterate through all of the fields
        for index, attr in  enumerate(self.data.dtype.names):

            if attr == 'JointAngles':

                for i, header in enumerate(joint_headers):
                    self.processed_data['JointAngles_'+header] = np.squeeze(self.data[0][0][index][0][0][i])

            elif attr == 'Pose':

                for i, header in enumerate(pose_headers):

                    if 'Base' in header:
                        self.processed_data['Pose_'+header] = self._3toPose(self.data[0][0][index][0][0][i])
                    elif 'ECM' in header or 'Workplace' in header:
                        self.processed_data['Pose_'+header] = self._12toPose(self.data[0][0][index][0][0][i])
                    elif 'PSM' in header:
                        for arm in range(3):
                            #print(self.data[0][0][index][0][0][i])
                            self.processed_data['Pose_'+header+str(arm+1)] = self._12toPose(self.data[0][0][index][0][0][i][arm])
                    elif 'RCM' in header:
                        for arm in range(1,4):

                            self.processed_data['Pose_'+header+"_PSM"+str(arm)] = self._12toPose(self.data[0][0][index][0][0][i][arm])

                        self.processed_data['Pose_'+header+"_ECM"] = self._12toPose(self.data[0][0][index][0][0][i][0])

            else:
                self.processed_data[attr] = np.squeeze(self.data[0][0][index])


        #print(self.processed_data)

    """
    Syntactic sugar to make the interface a little cleaner
    """
    def __getattr__(self, attr):
        return self.processed_data[attr]


    """
    String operation
    """
    def __str__(self):
        return str(self.processed_data)





"""
ISI Kinematic dataset is a collection of timestamped records
"""
class ISIKinematicDataset(object):

    def __init__(self, inputFile):

        self.data = scipy.io.loadmat(inputFile)

        _, self.N = self.data['dOUT'].shape

        self.array = []

        for i in range(self.N):
            obs = self.data['dOUT'][0,i]
            self.array.append(ISIRecord(obs))

    """
    Syntactic sugar for easy access
    """
    def __getattr__(self, attr):
        return [obj.__getattr__(attr) for obj in self.array]

    def __getitem__(self, index):
        try:
            return self.array[index]
        except:
            return None




class ISIVideoDataset(object):

    def __init__(self, endoscopeIndex, imageDirectory):
        self.endoscopeIndex = endoscopeIndex
        self.imageDirectory = imageDirectory

        
        self.data = scipy.io.loadmat(endoscopeIndex)
        _ , self.N = self.data['dOUT'].shape

        self.indexedFrames = {}

        for i in range(self.N):
            header, record = self.data['dOUT'][0,i]
            frameIndex = np.squeeze(record[0][0][6])
            imageFileName = self.imageDirectory+'/frame'+str(i)+'.jpg'
            self.indexedFrames[int(frameIndex)] = cv2.imread(imageFileName,0)


    def __getitem__(self, index):
        try:
            return self.indexedFrames[index]
        except:
            return None


class ISIDataset(object):

    def __init__(self, kinematics, endoscopeIndex, imageDirectory):
        self.kinematics = ISIKinematicDataset(kinematics)
        self.video = ISIVideoDataset(endoscopeIndex, imageDirectory)

        self.combined = {}

        for i in range(self.kinematics.N):
            videoFrame = self.video[i]
            kinematicsFrame = self.kinematics[i]

            #test if properly formed np array
            try:
                videoFrame[0]
            except:
                continue

            self.combined[i] = (videoFrame, kinematicsFrame)


    def __getitem__(self, index):
        try:
            return {'kinematics':self.combined[index][1], 'video': self.combined[index][0]}
        except:
            return None

    def keys(self):
        return sorted(self.combined.keys())






