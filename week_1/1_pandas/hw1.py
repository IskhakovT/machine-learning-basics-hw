# Copyright (c) Timur Iskhakov.


import pandas
from collections import defaultdict
from nameparser import HumanName


def static_var(varname, value):
    def decorate(func):
        setattr(func, varname, value)
        return func
    return decorate


@static_var('test_num', 1)
def print_ans(*args, sep=' '):
    with open('{}.txt'.format(print_ans.test_num), 'w') as fout:
        fout.write(sep.join(list(map(str, args))))
    print_ans.test_num += 1


data = pandas.read_csv('titanic.csv', index_col='PassengerId')

print_ans(data['Sex'].value_counts().get('male'), data['Sex'].value_counts().get('female'), sep=',')
print_ans(round(data['Survived'].value_counts().get(1) / data['Survived'].value_counts().sum() * 100, 2))
print_ans(round(data['Pclass'].value_counts().get(1) / data['Pclass'].value_counts().sum() * 100, 2))
print_ans(round(data['Age'].mean(), 2), round(data['Age'].median(), 2), sep='')
print_ans(round(data['SibSp'].corr(data['Parch']), 2))

first_name_dict = defaultdict(int)

for index, row in data.iterrows():
    if row['Sex'] == 'female':
        name = row['Name']
        if '(' in name:
            name = name[name.find('(') + 1: name.find(')')]

        first = HumanName(name).first
        if not first:
            continue

        first_name_dict[first] += 1

names = []
for name in first_name_dict:
    if first_name_dict[name] > 3:
        names.append((first_name_dict[name], name))

names.sort()
names.reverse()
print_ans(names[0][1])
