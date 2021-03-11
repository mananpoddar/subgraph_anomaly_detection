
# import cv2
import numpy as np 

class Threshold(object):
    def __init__(self, distribution, method, percentage):
        self.distribution = distribution
        self.method = method
        self.percentage = percentage

    def meanDistribution(self):
        distribution = self.distribution
        sum = 0
        for element in distribution :
            sum = sum + element
        return (sum/len(distribution))

    def topOfDistribution(self, percentage) :
        sorted_errors = sorted(self.distribution)
        index = int((1-(percentage/100.0))*len(self.distribution))
        threshold = sorted_errors[index] 
        return threshold

    def optimumThreshold(self):
        distribution = self.distribution
        if self.method == "mean":
            mean_of_distribution = self.meanDistribution()
            return mean_of_distribution*(1+self.percentage/100)
        elif self.method == "top":
            threshold = self.topOfDistribution(self.percentage)
            return threshold

        otsu_distribution = [[1,2,3],[2,3,4]]

        otsu_distribution = np.array(otsu_distribution)
        otsu_distribution = cv2.cvtColor(otsu_distribution, cv2.COLOR_BGR2GRAY)



        # for element in distribution :
        #     if element>= mean_of_distribution and element <= (mean_of_distribution + (self.aboveMean*mean_of_distribution)/100 ):
        #         otsu_distribution.append(element)

        # Otsu's thresholding
        ret2,th2 = cv2.threshold(otsu_distribution,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        print(ret2 + " " + th2)
