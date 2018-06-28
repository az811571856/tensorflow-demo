import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image

# a = [[1, 2, 3, 4], [6, 7, 4], [1, 2]]
# print(a[0, 1])

a = np.array([[1, 2, 3, 4], [6, 7, 4, 7], [10, 9, 44, 55]])
print(a[2, 1])
print(a)
print(a)
print(a + a)
print(a * a)
b = np.array([[1, 2], [6, 7]])

print(b * b)
print(np.dot(b, b))
print(np.dot(b, b) == np.array([[13, 16], [48, 61]]))
print([1, 2] == [1, 2])
print('-----')
print(np.sum(b))
print(np.sum(b, axis=1))
print(np.sum(b, axis=0))

b = np.array([[[1], [2]], [[3], [4]], [[45], [21]],  [[31], [41]]])
print(b[1, 0, 0])

print(type(3.56111111111111111111))
print(1.23e-9)
print(0.00000123)
print(type(not True))

a = 'ABC'
b = a
a = 'XTZ'
print(b)

nums = [2, 3, 4, 5]
print(nums[:-1])
print(nums[0:1] + nums[2:5])
print([1, 2] + [1])

for n in nums[0:1]:
    print(n)


for idx, n in enumerate(nums[0:1]):
    print(idx, n)


ns = nums[0:1]
for idx, n in enumerate(ns):
    ns[idx] = 3
    print(idx, n)

print(ns)
print(not (3 in ns))

sumN = 0
for n in range(101):
    # print(n)
    sumN += n
print(sumN)

dic = {'a': 'aaa', 'b': 'bb'}
del dic['a']
print(dic)


ar = np.array([[1, 2, 3, 4], [6, 7, 8, 9]])
print(ar[:2, 2:4])
# ar1 = [[1, 2, 3, 4], [6, 7, 8, 9]]
# print(ar1[1, :])

x = np.arange(0, 8, 0.4)
print(x)
y = np.sin(x)
plt.plot(x, y)
# plt.show()

z = range(0, 8, 3)
print(z)

img = Image.open('./image/my.jpg')
img_array = np.array(img)
print(img_array.shape)
img.transpose(Image.FLIP_LEFT_RIGHT)
img

