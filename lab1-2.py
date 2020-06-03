import math
import random
import matplotlib.pyplot as plt
import time
import numpy as np

N = 1024
n = 6
W = 2100

minimal_range = 0
maximum_range = 1


# Функція генератора стаціонарного випадкового сигналу
def stat_random_signal(n, W):
    A = [random.random() for _ in range(n)]
    phi = [random.random() for _ in range(n)]

    def f(t):
        x = 0
        for i in range(n):
            x += A[i]*math.sin(W / n * t * i + phi[i])
        return x
    return f


# Функція повертає значення мат. очікування
def get_m(x):
    return sum(x)/len(x)


# Функція повертає значення Дисперсії
def get_D(x, m=None):
    if m is None:
        m = get_m(x)
    return sum([(i - m) ** 2 for i in x]) / (len(x) - 1)

# Функція повертає Кореляційну функцію
def get_K(x, m):
    autocorrelation = []
    c=len(x)
    for t in range(c):
        num = 0
        den = 0
        for i in range(c):
            xim = x[i]-m
            num += xim*(x[(i+t) % c]-m)
            den += xim*xim
        autocorrelation.append(num/(c-1))
    return autocorrelation


def get_Kxy(x, y, m):
    correlation = []
    c=len(x)
    for t in range(c):
        num = 0
        den = 0
        for i in range(c):
            xim = x[i]-m
            num += xim*(y[(i+t) % c]-m)
            den += xim*xim
        correlation.append(num/(c-1))
    return correlation


s_gen = stat_random_signal(n, W)
s_gen2 = stat_random_signal(n, W)
s = [s_gen(i) for i in range(N)]
s2 = [s_gen2(i) for i in range(N)]

m_start_time = time.time()
m = get_m(s)
m_end_time = time.time()
m_time = m_end_time - m_start_time


D_start = time.time()
D = get_D(s, m)
D_end = time.time()
D_time = D_end - D_start


K_start = time.time()
K = get_K(s, m)
K_end = time.time()
K_time = K_end - K_start

Kxy_start = time.time()
K3 = get_Kxy(s, s2, m)
Kxy_end = time.time()
Kxy_time = Kxy_end - Kxy_start


print("Час розрахунку мат. очікування: ", m_time)
print("Час розрахунку дисперсії: ", D_time)
print("Час розрахунку автокореляції: ", K_time)
print("Час розрахунку взаємокореляції: ", Kxy_time)
if K_time > Kxy_time:
    print("Час розрахунку автокореляції на ", K_time-Kxy_time, "с довше")
else:
    print("Час розрахунку взаємокореляції на ", Kxy_time-K_time, "с довше")


print("m =", m)
print("D =", D)
# print("K = ", K)

plt.subplot(2, 1, 1)
plt.plot(range(N), s)
plt.plot(range(N), K)
plt.subplot(2, 1, 2)
plt.plot(range(N), s2)
plt.plot(range(N), s)
plt.plot(range(N), K3)

plt.show()
