# -*- coding: utf-8 -*-

# name=input('please enter your name:') # 返回str
# print('hello',name)

# print(r'''hello,\n world''')

# print(u'中文测试')

# print('Hello,%s'%('Jennie'))
# print('%2d-%02d'%(3,1))
# print('%.2f'%3.14)
# print(list(range(5)))

# classmates = ['Michael', 'Bob', 'Tracy']
# tuple1=(1,2,3)
# set1=([1,2,3])
# integers={(1,2,3):1}
# print(integers)
# integers2={(1,2,[3]):1} # error!
# print(integers2)
import pdb
def cal(*numbers):
    sum=0
    for n in numbers:
        sum = sum+n*n
    return sum

def person(name,age,**kw):
    print('name:',name,'age:',age,'other:',kw)
    return

def f1(a,b,c=0,*args,**kw):
    print('a=',a,'b=',b,'c=',c,'args=',args,'kw=',kw)
    return

class Student(object):
    name='Student'
    def __init__(self,name,score):
        self.__name=name
        self.__score=score
    def print_score(self):
        print('%s:%s'%(self.__name,self.__score))
    def get_grade(self):
        if self.__score>=90:
            return 'A'
        elif self.__score>=60:
            return 'B'
        else:
            return 'C'
    def get_name(self):
        return self.__name
    def get_score(self):
        return self.__score
    def set_score(self,score):
        if 0<=score<=100:
            self.__score=score
        else:
            raise ValueError('bad score')

class Animal(object):
    def run(self):
        print('Animal is running')
class Dog(Animal):
    def run(self):
        print('Dog is running')
    def __len__(self):
        return 100
class Cat(Animal):
    def run(self):
        print('Cat is running')
class Timer(object):
    def run(self):
        print('Starting')

def run_twice(animal):
    animal.run()
    animal.run()

import logging
logging.basicConfig(level=logging.INFO)
class FooError(ValueError):
    pass

def foo(s):
    # return 10/int(s)
    n=int(s)
    logging.info('n=%d'%n)
    if n==0:
        raise FooError('invalid value:%s'%n)
    return 10/n

def bar(s):
    # return foo(s)*2
    try:
        foo(s)
    except FooError as e:
        print('FooError!')
        raise

def main():
    try:bar('0')
    except Exception as e:
        # print('Error:',e)
        logging.exception(e)
    finally:
        print('finally...')

import unittest

class Dict(dict):
    '''
    Simple dict but also support access as x.y style.

    >>> d1 = Dict()
    >>> d1['x'] = 100
    >>> d1.x
    100
    >>> d1.y = 200
    >>> d1['y']
    200
    >>> d2 = Dict(a=1, b=2, c='3')
    >>> d2.c
    '3'
    >>> d2['empty']
    Traceback (most recent call last):
        ...
    KeyError: 'empty'
    >>> d2.empty
    Traceback (most recent call last):
        ...
    AttributeError: 'Dict' object has no attribute 'empty'
    '''
    def __init__(self,**kw):
        super().__init__(**kw)
    def __getattr__(self,key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(r"'Dict' object has no attribute '%s'" % key)
    def __setattr__(self,key,value):
        self[key]=value

class TestDict(unittest.TestCase):
    def setUp(self):
        print('setUp...')
    def test_init(self):
        d=Dict(a=1,b='test')
        self.assertEqual(d.a,1)
        self.assertEqual(d.b,'test')
        self.assertTrue(isinstance(d,dict))
    def test_key(self):
        d=Dict()
        d['key']='value'
        self.assertEqual(d.key,'value')
    def test_attr(self):
        d=Dict()
        d.key='value'
        self.assertTrue('key' in d)
        self.assertEqual(d['key'],'value')

    def test_keyerror(self):
        d = Dict()
        with self.assertRaises(KeyError):
            value = d['empty']

    def test_attrerror(self):
        d = Dict()
        with self.assertRaises(AttributeError):
            value = d.empty

    def tearDown(self):
        print('tearDown...')

if __name__=='__main__':
    import doctest
    doctest.testmod()

# if __name__ =="__main__":
#     numbers=[1,2]
#     print(cal(*numbers))
#     print(person('Adam',45,gender='M',job='Engineer'))
#     extra={'city':'Hangzhou','school':'ZJU'}
#     person('Jack',24,**extra)
#     args=(1,2,3,4)
#     kw={'d':90,'x':'#'}
#     f1(*args,**kw)
#     bart=Student('Bart Simpson',59)
#     lisa=Student('Lisa Simpson',87)
#     bart.age=8
#     print(bart.age)
#     # print(lisa.age)
#     run_twice(Animal())
#     run_twice(Dog())
#     run_twice(Timer())
#     print(len(Dog()))
#     pdb.set_trace()
#     main()

