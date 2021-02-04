from scipy.stats import ttest_ind
import numpy as np
import sys

normal_avg = np.genfromtxt("stats_processing/020221_4_64/012821-4_64_normal.csv",  delimiter=",")
softmax_avg = np.genfromtxt("stats_processing/020221_4_64/020121-4_64_relu.csv",  delimiter=",")
# print(normal_avg[1:, 1:])

normal_avg = normal_avg[1:, 1:30]
softmax_avg = softmax_avg[1:, 1:30]

normal_avg_mean = np.mean(normal_avg)
softmax_avg_mean = np.mean(softmax_avg)

print("normal_avg mean value:",normal_avg_mean)
print("softmax_avg mean value:",softmax_avg_mean)

normal_avg_std = np.std(normal_avg)
softmax_avg_std = np.std(softmax_avg)

print("normal_avg std value:",normal_avg_std)
print("softmax_avg std value:",softmax_avg_std)

ttest,pval = ttest_ind(normal_avg,softmax_avg, axis=1)
print("ttest", ttest)
print("p-value",pval)
print((pval / 2 < 0.05))
print("average 2-tail p-value", np.average(pval))

# print(np.sum(pval > 0))

# if pval <0.05:
#   print("we reject null hypothesis")
# else:
#   print("we accept null hypothesis")