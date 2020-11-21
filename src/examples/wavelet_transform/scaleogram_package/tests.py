import matplotlib.pyplot as plt
import scaleogram as scg
import seaborn as sns
import sys

print("TEST notebook for scaleogram version="+scg.__version__+
      " python version=" + sys.version)

plt.rcParams['figure.figsize'] = [14,6]
plt.rcParams['font.size'] = 14.0
scg.test_cws()