x = [1, 3, 4]
print(x, x[2])
x.append(3)
print(x)
x.remove(3)
print(x)
print(x.count(3))
x.extend([6, 7])
print(x)
r = reversed(x)
for n in r:
    print(n)

print('\n----')

for n in x:
    print(n)

d = {'c2', 'c5', 'c2'}
print(d)

di = {'cat': 'cat cat', 'dog': 'dog dg'}
print(di)

print(di.get('cat'))

di['chicken'] = 'c k e n'
print(di)

di.setdefault('888', 'sss')
print(di)

# di.clear()
# print(di)

m = di.items()
print(m)
t = m[0]

print(m[0])
print(t[0]+'----\n')


def f(p1):
    if p1 == 'x':
        return 'xx'
    elif p1 == 'z':
        return 'zz'


print(f('y'))


class Cla(object):
    def __init__(self, name, mobile):
        self.name = name
        self.mobile = mobile

    def prt(self):
        print(self.name, self.mobile)


cla = Cla('xx', '199')
print('{}:{}'.format(cla.name, cla.mobile))
