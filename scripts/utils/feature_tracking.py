import cv2
from collections import deque

class FeatureTrackerDrawer:

    lineColor = (200, 0, 200)
    pointColor = (0, 0, 255)
    circleRadius = 2
    maxTrackedFeaturesPathLength = 30
    # for how many frames the feature is tracked
    trackedFeaturesPathLength = 10

    trackedIDs = None
    trackedFeaturesPath = None

    def onTrackBar(self, val):
        FeatureTrackerDrawer.trackedFeaturesPathLength = val
        pass

    def trackFeaturePath(self, features, disps):
        """
        disps: list of disparity maps
        """

        newTrackedIDs = set()
        for currentFeature in features:
            currentID = currentFeature.id
            newTrackedIDs.add(currentID)

            if currentID not in self.trackedFeaturesPath:
                self.trackedFeaturesPath[currentID] = deque()

            path = self.trackedFeaturesPath[currentID]

            x, y = round(currentFeature.position.x), round(currentFeature.position.y)
            featureVector = [x, y]
            for disp in disps:
                featureVector.append(disp[y, x])
            # path.append(currentFeature.position)
            path.append(featureVector)
            while(len(path) > max(1, FeatureTrackerDrawer.trackedFeaturesPathLength)):
                path.popleft()

            self.trackedFeaturesPath[currentID] = path

        featuresToRemove = set()
        for oldId in self.trackedIDs:
            if oldId not in newTrackedIDs:
                featuresToRemove.add(oldId)

        for id in featuresToRemove:
            self.trackedFeaturesPath.pop(id)

        self.trackedIDs = newTrackedIDs

    def drawFeatures(self, img):

        cv2.setTrackbarPos(self.trackbarName, self.windowName, FeatureTrackerDrawer.trackedFeaturesPathLength)

        for featurePath in self.trackedFeaturesPath.values():
            path = featurePath

            for j in range(len(path) - 1):
                src = (int(path[j][0]), int(path[j][1]))
                dst = (int(path[j + 1][0]), int(path[j + 1][1]))
                cv2.line(img, src, dst, self.lineColor, 1, cv2.LINE_AA, 0)
            j = len(path) - 1
            cv2.circle(img, (int(path[j][0]), int(path[j][1])), self.circleRadius, self.pointColor, -1, cv2.LINE_AA, 0)

    def trackDisparities(self):
        delta_disps = {}
        for path in self.trackedFeaturesPath.values():
            if len(path) >= 2:
                i = len(path)-2
                j = len(path)-1
                for k in range(2, len(path[i])):
                    delta_disp = abs(path[j][k]-path[i][k])
                    if k not in delta_disps.keys():
                        delta_disps[k-2] = []
                    delta_disps[k-2].append(delta_disp)
        return delta_disps

    def __init__(self, trackbarName, windowName):
        self.trackbarName = trackbarName
        self.windowName = windowName
        cv2.namedWindow(windowName)
        cv2.createTrackbar(trackbarName, windowName, FeatureTrackerDrawer.trackedFeaturesPathLength, FeatureTrackerDrawer.maxTrackedFeaturesPathLength, self.onTrackBar)
        self.trackedIDs = set()
        self.trackedFeaturesPath = dict()
