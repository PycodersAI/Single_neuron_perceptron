import matplotlib.pyplot as plt


plt.plot([-1,-3],[2,3],'r+',label='+')
plt.plot([3,3],[-2,-4],'bo',label='-')
plt.plot([5,-5],[8/4.9,-6/4.9],'g')
plt.plot([5,-5],[5.16,-4.89],'g')
plt.plot([5,-5],[7.17,-7.8],'g')
plt.plot([5,-5],[1.472,-3.906],'g')
plt.xlabel('x')
plt.ylabel('y')
plt.title('y vs x')
plt.grid()
#plt.legend()
plt.show()
