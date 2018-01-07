import pandas as pd
import numpy as np
import random
import string

numberOfData = 100000

def generate_symbol_type10():
    number1 = random.randint(0,8)
    schaltplan = '{}'.format(number1)
    layoutData = '{}'.format(number1+1)
    return (0, schaltplan, layoutData, layoutData)

def generate_symbol_type20():
    number1 = random.randint(0,9)
    schaltplan = '{}'.format(number1)
    layoutData = '{}'.format(number1)
    return (1, schaltplan, schaltplan, 3)

def generate_symbol_type1():
    letter = random.choice(string.ascii_letters)
    number1 = random.randint(0,9)
    number2 = random.randint(0,9)
    schaltplan = 'KF__{}.{}'.format(letter,letter,number1,number2)
    layoutData = 'KF{}{}{}.{}'.format(letter,letter,number1,number2)
    return (1, schaltplan, layoutData, layoutData)


def generate_symbol_type2():
    letter = random.choice(string.ascii_letters)
    number = random.randint(0,9999)
    layoutData = 'M{}{}'.format(letter, number)
    return (2, layoutData, layoutData, layoutData)


generators = [
    generate_symbol_type10,
    generate_symbol_type20
]

data = []

for i in range(numberOfData):
    rowFunc = random.choice(generators)
    row = rowFunc()
    data.append(row)

columns = ['SymbolType', 'CircuitDiagram', 'Layout', 'Result']
df = pd.DataFrame(data, columns=columns)
df.to_csv('data.csv', index=False)