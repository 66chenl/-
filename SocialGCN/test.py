import matplotlib.pyplot as plt
x = [1,2,3,4]
y = [2,3,4,5]
z = [1,2,3,4]
plt.plot(x,y,label='x')
plt.savefig('figure_y.png')

plt.clf()
plt.plot(x,z,label= 'z')
plt.savefig('figure_z.png')
