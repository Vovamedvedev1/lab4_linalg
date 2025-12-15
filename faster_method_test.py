from solve import *

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

A, b = generate_dd_system(n = 500)
linalg_solver = LinalgSolver(A, b, 10**(-7))
A, b = Matrix(A), Vector(b)

try:
    x = linalg_solver.get_faster_spusk()
    print("Решение методом Скорейшего спуска = ", x.vect)
    print("Погрешность решения = ", (b - A * x).get_norm())
except Exception as e:
    print(str(e))