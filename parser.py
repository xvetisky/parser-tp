# _*_ coding: utf-8 _*_
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

power = re.compile('\$\{\^\-?[0-9]\}\$')
power_numerical = re.compile('[\-]?[0-9]')
comma = re.compile('^[0-9,-]*')
f = open("data.txt", 'r')
allowed = "1234567890, -^${}·"
numerical = "1234567890."
model = LinearRegression()


def remove_junk(file):
    result = []
    for line in file:
        current = []
        for c in line:
            if c in allowed:
                current.append(c)
        result.append(''.join(current))
    return result


def strdata_to_float(str):
    #print(str)
    result = ""
    power_test = power.search(str)
    comma_test = comma.match(str)
    result += comma_test.group(0)
    if ',' in result:
        result = right_comma(result)
    number = float(result)
    #print(power_test)
    if power_test:
        pwrz = power_numerical.search(power_test.group(0))
        pwr = pwrz.group(0)
        int_pwr = int(pwr)
        number *= 10**int_pwr
    return number


def right_comma(str):
    result = ""
    for c in str:
        if c == ',':
            result += '.'
            continue
        result += c
    return result

def final_data(l):
    result = []
    for subl in l:
        sub_result = []
        for e in subl:
            sub_result += [strdata_to_float(e)]
        result.append(sub_result)
    return result


def clean(l):
    result = []
    for subl in l:
        sub_result = []
        current = ""
        for char in subl:
            if char == " ":
                sub_result.append(current) if len(current) else None
                current = ""
                continue
            current += char
        result.append(sub_result)
    return result


if __name__ == "__main__":
    clean_lines = remove_junk(f)
    clean_data = clean(clean_lines)
    final = final_data(clean_data)
    n = len(final)
    t = np.array([final[i][3] for i in range(n)])
    t1 = t.reshape((-1, 1))
    #y = np.array([final[i][5] for i in range(n)])
    y2 = np.array([final[i][4] for i in range(n)])
    y = np.log(y2)

    new_model = model.fit(t1, y)
    print(f'r^2 = {model.score(t1, y)}')
    a, b = new_model.coef_, new_model.intercept_
    print(f'a = -k = {a}', f'b = {b}')
    Y = a*t + b

    plt.plot(t1, y2, 'b-')
    plt.plot(t1, Y, 'r-')
    plt.show()
