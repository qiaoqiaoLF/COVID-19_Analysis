#TODO visualize the data
import matplotlib.pyplot as plt
def plot(x,y):
    plt.figure()
    plt.plot(x,y)
    for a,b in zip(x,y):
        plt.text(a,b,(a,b),ha='center',va='bottom',fontsize=15)
    plt.show()
#x = [1,2,3,4,5]
#y = [2,5,7,3,4]
#plot(x,y)
