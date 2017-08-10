# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 11:01:46 2017

@author: Aian Fund
"""
import os
os.chdir('D:/Neal/Py4Invst')

from Analyzers import Analyzer

x = Analyzer("D:/Neal/Quant/PythonProject/ValuesFile/values1.csv")

from Analyzers import Draw_Down

x = Draw_Down("D:/Neal/Quant/PythonProject/ValuesFile/values1.csv")

class Animail(object):
    def run(self):
        print("Animal is running")
        
class Dog(Animail):
    def run(self):
        print("Dog is running")
    
    def eat(self):
        print("eating")

class Cat(Animail):
    def run(self):
        print("cat is running")
        
dog = Dog()
cat = Cat()

dog.run()
cat.run()

class MyObject(object):
    def __init__(self):
        self.x = 9
    
    def power(self):
        return self.x * self.x

obj = MyObject()


class Screen(object):
    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = value

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        self._height = value

    @property
    def resolution(self):
        return self._width * self._height

s = Screen()
s.width = 1024
s.height = 768
print(s.resolution)
assert s.resolution == 786432, '1024 * 768 = %d ?' % s.resolution

import re
a = "D:/Neal/Quant/PythonProject/ValuesFile/values1.csv"
