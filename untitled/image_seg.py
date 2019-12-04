from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#pic = plt.imread('interlocking_ub.png')/255  # dividing by 255 to bring the pixel values between 0 and 1
img = mpimg.imread('..\grass-tiger.png')
#print(img)
#print(img.shape)
pic_n = img.reshape(img.shape[0]*img.shape[1], img.shape[2])



kmeans = KMeans(n_clusters=8, random_state=0).fit(pic_n)
pic2show = kmeans.cluster_centers_[kmeans.labels_]

cluster_pic = pic2show.reshape(img.shape[0], img.shape[1], img.shape[2])
print("Done")
plt.imshow(cluster_pic)
plt.show()