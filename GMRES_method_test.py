from solve import *

def get_random_universal_matrix(n, min_val=-1000, max_val=1000, b_min_val=100000, b_max_val=500000):
    A, b = [[uniform(min_val, max_val) for _ in range(n)] for _ in range(n)], [uniform(b_min_val, b_max_val) for _ in range(n)]
    return A, b


def generate_dd_system(n, min_val=-1000, max_val=1000, dominance_factor=2.0, b_min_val=100000, b_max_val=500000):
    A = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        sum_off_diag = 0.0
        for j in range(n):
            if i != j:
                A[i][j] = uniform(min_val, max_val)
                sum_off_diag += abs(A[i][j])
        A[i][i] = sum_off_diag 
        if uniform(0,1) < 0.3:  
            A[i][i] = -A[i][i]
    b = [uniform(b_min_val, b_max_val) for _ in range(n)]
    return A, b

def generate_gilbert_matrix(n):
    A = [[1/(i + j + 1) for j in range(n)] for i in range(n)]
    b = [1 / (i + 1) for i in range(n)]
    return A, b

print("--------------Тестирование метода GMRES с вращениями Гивенса--------------------")
print("Выберите тип матрицы:")
print('''0.Матрица без диагонального преобладания и симметричности
1.Заданная матрица
2.Случайная матрица
3.Матрица Гильберта 
4.Вырожденная матрица 
5.Единичная матрица''')
t = int(input("Введите число: "))
while t <= -1 or t > 5:
    t = int(input("Введите число: "))

if t == 0:
    n = int(input("Введите размерность матрицы без диагонального преобладания и симметричности: "))
    A, b = get_random_universal_matrix(n)
elif t == 1:
    A, b = [
        [20, -2, 1, 0, 4, -1, 2, 5, -3, 0],
        [-1, 12, 0, 2, -3, 1, 4, 0, 1, -2],
        [2, 0, 14, -1, 0, 3, -2, 1, 4, 1],
        [0, 3, -2, 14, 1, 0, 5, -1, 2, 3],
        [4, 1, -3, 0, 15, 2, -1, 3, 0, -4],
        [-2, 4, 1, 3, 0, 11, 0, -2, 1, 5],
        [1, -1, 5, -2, 2, 0, 17, 4, -3, 0],
        [3, 0, 2, 1, -4, 5, 0, 16, 1, -2],
        [0, 2, -1, 5, 3, -2, 1, 0, 18, 4],
        [5, -3, 0, -4, 1, 2, 3, -5, 0, 19]
    ], [31, 12, 74, 76, 35, 105, 68, 112, 151, 43]
elif t == 2:
    n = int(input("Введите размерность случайной матрицы: "))
    A, b = generate_dd_system(n = n)
elif t == 3:
    n = int(input("Введите размерность матрицы Гильберта: "))
    A, b = generate_gilbert_matrix(n = n)
elif t == 4:
    A, b = [[1,2,3,4,5], [6,7,8,9,10], [11,12,13,14,15],[16,17,18,19,20], [21,22,23,24,25]], [15,25,35,45,55]
elif t == 5:
    n = int(input("Введите размерность единичной матрицы"))
    A, b = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)], [i + 1 for i in range(n)]

linalg_solver = LinalgSolver(A, b, 10**(-5))
A, b = Matrix(A), Vector(b)


try:
    x = linalg_solver.get_GMRES_givens()
    print("Решение методом GMRES с вращениями Гивенса = ", x.vect)
    print("Погрешность = ", (b - A * x).get_norm())
except Exception as e:
    print(str(e))