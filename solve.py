from copy import deepcopy
from math import sqrt
from random import uniform
import numpy as np

def ilu_preconditioner_simple(A, drop_tol=1e-100):
    A = np.array(A, dtype=float)
    n = A.shape[0]
    L = np.eye(n, dtype=float)
    U = np.zeros((n, n), dtype=float)    
    for i in range(n):
        for j in range(i, n):
            U[i, j] = A[i, j]        
        for k in range(i):
            if abs(L[i, k]) > drop_tol and abs(U[k, k]) > 1e-100:
                for j in range(i, n):
                    U[i, j] -= L[i, k] * U[k, j]        
        if abs(U[i, i]) < 1e-100:
            U[i, i] = 1.0        
        pivot = U[i, i]
        for k in range(i+1, n):
            if abs(A[k, i]) > drop_tol:
                L[k, i] = A[k, i] / pivot                
                for j in range(i):
                    if abs(L[k, j]) > drop_tol:
                        L[k, i] -= L[k, j] * U[j, i] / pivot    
    def solve_L_fast(r):
        y = r.copy()
        for i in range(n):
            if i > 0:
                y[i] -= np.dot(L[i, :i], y[:i])
        return y
    
    def solve_U_fast(y):
        x = y.copy()
        for i in range(n-1, -1, -1):
            if i < n-1:
                x[i] -= np.dot(U[i, i+1:], x[i+1:])
            if abs(U[i, i]) > 1e-12:
                x[i] /= U[i, i]
            else:
                x[i] = 0.0
        return x
    
    def solve_M(r):
        return solve_U_fast(solve_L_fast(r)) 
    return solve_M, L, U

 
class Vector:
    def __init__(self, b):
        self.vect = deepcopy(b)
        self.n = len(self.vect)
 
    def __mul__(self, other):
        if not isinstance(other, Vector):
            raise ArithmeticError("Скалярно перемножаться могут только два вектора!!!")
        if self.n != other.n:
            raise ArithmeticError("Некорректные размеры векторов при поиске скалярного произведения!!!")
        return sum([self.vect[k] * other.vect[k] for k in range(self.n)])
    
    def __rmul__(self, scalar):
        return Vector([scalar * self.vect[k] for k in range(self.n)])
 
    def __add__(self, other):
        if not isinstance(other, Vector):
            raise ArithmeticError("Складываться могут только два вектора!!!")
        if self.n != other.n:
            raise ArithmeticError("Некорректные размеры векторов при сложении!!!")
        return Vector([self.vect[k] + other.vect[k] for k in range(self.n)])
    
    def __sub__(self, other):
        if not isinstance(other, Vector):
            raise ArithmeticError("Вычитаться могут только два вектора!!!")
        if self.n != other.n:
            raise ArithmeticError("Некорректные размеры векторов при вычитании!!!")
        return Vector([self.vect[k] - other.vect[k] for k in range(self.n)])
    
    def get_norm(self):
        return sqrt(sum(self.vect[k] ** 2 for k in range(self.n)))
 
class Matrix:
    def __init__(self, A):
        self.matr = deepcopy(A)
        self.n = len(self.matr)
        self.m = len(self.matr[0])
 
    def T(self):
        matrix_transpone = [[0 for _ in range(self.n)] for _ in range(self.m)]
        for i in range(self.m):
            for j in range(self.n):
                matrix_transpone[i][j] = self.matr[j][i]
        return Matrix(matrix_transpone)
 
    def __mul__(self, other):
        result = None
        if isinstance(other, Matrix):
            if self.m != other.n:
                raise ArithmeticError("Некорректные размеры матриц для умножения!!!")
            result = [[0 for _ in range(other.m)] for _ in range(self.n)]
            for i in range(self.n):
                for j in range(other.m):
                    result[i][j] = sum([self.matr[i][k] * other.matr[k][j] for k in range(self.m)])
            result = Matrix(result)
        elif isinstance(other, Vector):
            if self.m != other.n:
                raise("Некорректные размеры матрицы и вектора для перемножения!!!")
            result = [0] * self.n
            for i in range(self.n):
                result[i] = sum([self.matr[i][k] * other.vect[k] for k in range(self.m)])
            result = Vector(result)
        return result
    
    def __rmul__(self, scalar):
        return Matrix([[scalar * self.matr[i][j] for j in range(self.m)] for i in range(self.n)])
    
    def __sub__(self, other):
        if not isinstance(other, Matrix):
            raise ArithmeticError("Вычитаться могут только две матрицы")
        if self.n != other.n or other.m != self.m:
            raise ArithmeticError("Некорректные размеры матриц при вычитании!!!")
        return Matrix([[self.matr[i][j] - other.matr[i][j] for j in range(self.m)] for i in range(self.n)])
    
    def get_norm(self):
        sum_p = 0
        for i in range(self.n):
            for j in range(self.m):
                sum_p += self.matr[i][j] ** 2
        return sqrt(sum_p)
 
def scalar_proizv(vector_1, vector_2):
    n_1, n_2 = len(vector_1), len(vector_2)
    if n_1 != n_2:
        raise ArithmeticError("Некорректные размеры векторов при поиске скалярного произведения!!!")
    return sum([vector_1[k] * vector_2[k] for k in range(n_1)])
 
def vector_plus_minus(vector_1, vector_2, operand):
    n_1, n_2 = len(vector_1), len(vector_2)
    if n_1 != n_2:
        raise ArithmeticError("Некорректные размеры векторов при поиске скалярного произведения!!!")
    return [vector_1[k] + vector_2[k] if operand == '+' else vector_1[k] - vector_2[k] for k in range(n_1)]
 
    
class LinalgSolver:
    def __init__(self, A, b, eps):
        self.matrix_A = deepcopy(A)
        self.vector_b = deepcopy(b)
        self.eps = eps
        self.n = len(A[0])
 
    def get_gauss_method(self):
        A, b = deepcopy(self.matrix_A), deepcopy(self.vector_b)
        E = [[1 if i==j else 0 for j in range(self.n)] for i in range(self.n)]
        obr = [[0 for j in range(self.n)] for i in range(self.n)]
        x = [0.0] * self.n
        count_permutation = 0 
        for i in range(self.n):
            leader_index = i
            max_value = abs(A[i][i])
            for k in range(i + 1, self.n):
                if abs(A[k][i]) > max_value:
                    max_value = abs(A[k][i])
                    leader_index = k
            if abs(A[leader_index][i]) < 1e-200:
                raise ArithmeticError("Матрица вырождена")
            if leader_index != i:
                A[i], A[leader_index] = A[leader_index], A[i]
                b[i], b[leader_index] = b[leader_index], b[i]
                E[i], E[leader_index] = E[leader_index], E[i]
                count_permutation += 1
            for k in range(i + 1, self.n):
                coef = A[k][i] / A[i][i] 
                for j in range(i, self.n):
                    A[k][j] -= coef * A[i][j]
                b[k] -= coef * b[i]
                for f in range(self.n):
                    E[k][f] -= coef * E[i][f]      
        product = 1.0
        for i in range(self.n):
            product *= A[i][i]
        determinant = ((-1) ** count_permutation) * product
        for i in range(self.n - 1, -1, -1):
            s = sum(A[i][j] * x[j] for j in range(i + 1, self.n))
            x[i] = (b[i] - s) / A[i][i]   
            for f in range(self.n):
                so = sum(A[i][j] * obr[j][f] for j in range(i + 1, self.n))
                obr[i][f] = (E[i][f] - so) / A[i][i]    
        return Vector(x), Matrix(obr), determinant
    
        
    def lup_decomposition(self):
        A = deepcopy(self.matrix_A)
        P = [i for i in range(self.n)]
        count_permutation = 0
        for i in range(self.n):
            leader_index = i
            max_value = abs(A[i][i])
            for k in range(i+1, self.n):
                if abs(A[k][i]) > max_value:
                    max_value = abs(A[k][i])
                    leader_index = k
            if abs(A[leader_index][i]) < 10**(-20):
                raise ArithmeticError("Матрица вырожденная")
            if leader_index != i:
                A[i], A[leader_index] = A[leader_index], A[i]
                P[i], P[leader_index] = P[leader_index], P[i]
                count_permutation += 1
            for j in range(i + 1, self.n):
                factor = A[j][i] / A[i][i]
                A[j][i] = factor
                for k in range(i + 1, self.n):
                    A[j][k] -= factor * A[i][k]
        L = [[0.0] * self.n for _ in range(self.n)]
        U = [[0.0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                if i > j:
                    L[i][j] = A[i][j]
                elif i == j:
                    L[i][j] = 1.0
                    U[i][j] = A[i][j]
                else:
                    U[i][j] = A[i][j]
        return (L,U,P,count_permutation)
    
    
    def criter(self, x_new):
        return (Vector(self.vector_b) - Matrix(self.matrix_A) * Vector(x_new)).get_norm()


    def get_canonic_matrix(self, A, b): 
        A, b = deepcopy(A), deepcopy(b)
        n = len(A)
        for k in range(n):
            max_val = abs(A[k][k])
            i_max = k
            for i in range(k + 1, n):
                if abs(A[i][k]) > max_val:
                    max_val = abs(A[i][k])
                    i_max = i            
            if max_val < 1e-7:
                raise ValueError(f"Матрица вырождена или близка к вырожденной на шаге {k}")            
            if i_max != k:
                A[k], A[i_max] = A[i_max], A[k]
                b[k], b[i_max] = b[i_max], b[k]
        return A, b
        '''for i in range(n):
            leader_index = i
            max_value = abs(A[i][i])
            for k in range(i + 1, self.n):
                if abs(A[k][i]) > max_value:
                    max_value = abs(A[k][i])
                    leader_index = k
            if abs(A[leader_index][i]) < 1e-15:
                raise ArithmeticError("Матрица вырождена")
            if leader_index != i:
                A[i], A[leader_index] = A[leader_index], A[i]
                b[i], b[leader_index] = b[leader_index], b[i]
            for k in range(i + 1, self.n):
                coef = A[k][i] / A[i][i] 
                for j in range(i, self.n):
                    A[k][j] -= coef * A[i][j]
                b[k] -= coef * b[i]'''
        return A, b

    
    def apply_givens_rotation(self, h, cs, sn, k):
        for i in range(k):
            temp = cs[i] * h[i] + sn[i] * h[i+1]
            h[i+1] = -sn[i] * h[i] + cs[i] * h[i+1]
            h[i] = temp        
        if abs(h[k+1]) > 1e-15:
            beta = sqrt(h[k]**2 + h[k+1]**2)
            cs_k = h[k] / beta
            sn_k = h[k+1] / beta
            h[k] = beta
            h[k+1] = 0.0
        else:
            cs_k = 1.0 if h[k] >= 0 else -1.0
            sn_k = 0.0
            h[k] = abs(h[k])
            h[k+1] = 0.0
        cs.append(cs_k)
        sn.append(sn_k)
        return h, cs, sn 
    
    def get_GMRES_givens(self, max_iterations=10**6, m=50):
        A, b = np.array(self.matrix_A, dtype=float), np.array(self.vector_b, dtype=float)
        n = len(b)        
        if n < m:
            print(f"Предупреждение: размерность системы ({n}) меньше размера подпространства Крылова ({m})")
            m = n        
        solve_M, _, _ = ilu_preconditioner_simple(A)    
        x = b.copy()        
        r = b - A @ x
        r_norm_initial = np.linalg.norm(r)        
        residuals = [r_norm_initial]
        converged = False
        total_iterations = 0 
        for iteration in range(max_iterations):
            if converged:
                break            
            r0 = solve_M(r)
            beta = np.linalg.norm(r0)            
            H = np.zeros((m + 1, m))
            V = np.zeros((m + 1, n))              
            g = np.zeros(m + 1)
            g[0] = beta            
            V[0] = r0 / beta if beta > 1e-15 else r0            
            cs_list = []  
            sn_list = []              
            for j in range(m):
                z = A @ V[j]                
                w = solve_M(z)                
                for i in range(j + 1):
                    H[i, j] = np.dot(w, V[i])
                    w = w - H[i, j] * V[i]
                H[j+1, j] = np.linalg.norm(w)                
                if H[j+1, j] < 1e-15:
                    m_actual = j + 1
                    H = H[:m_actual+1, :m_actual]
                    V = V[:m_actual+1]
                    g = g[:m_actual+1]
                    break
                else:
                    m_actual = m                
                V[j+1] = w / H[j+1, j]                
                h_col = H[:j+2, j].copy()
                h_col, cs_list, sn_list = self.apply_givens_rotation(h_col, cs_list, sn_list, j)
                H[:j+2, j] = h_col
                for i in range(j):
                    temp = cs_list[i] * g[i] + sn_list[i] * g[i+1]
                    g[i+1] = -sn_list[i] * g[i] + cs_list[i] * g[i+1]
                    g[i] = temp                
                temp = cs_list[j] * g[j] + sn_list[j] * g[j+1]
                g[j+1] = -sn_list[j] * g[j] + cs_list[j] * g[j+1]
                g[j] = temp                
                residual_norm = abs(g[j+1])
                residuals.append(residual_norm)
                total_iterations += 1                
                if residual_norm < self.eps * r_norm_initial:
                    m_actual = j + 1
                    H = H[:m_actual+1, :m_actual]
                    V = V[:m_actual+1]
                    g = g[:m_actual+1]
                    break
            H_small = H[:m_actual, :m_actual]
            g_small = g[:m_actual]            
            y = np.linalg.solve(H_small, g_small)            
            x_update = np.zeros(n)
            for i in range(m_actual):
                x_update += y[i] * V[i]
            x += x_update            
            r = b - A @ x
            R = self.criter(x.tolist())
            print(f"Итерация: {iteration}; Норма вектора невязки = {R}")
            if R < self.eps:
                converged = True
                break
        return Vector(x.tolist())

    def jacobi_method(self, max_iterations=10**6):
        A, b = self.get_canonic_matrix(self.matrix_A, self.vector_b)
        for i in range(self.n):
            diag = A[i][i]
            scale = 1.0 / diag
            for j in range(self.n):
                A[i][j] *= scale
            b[i] *= scale
        x_old, x_new = deepcopy(b), deepcopy(b)
        for iteration in range(max_iterations):
            for i in range(self.n):
                x_new[i] = (b[i] - sum([A[i][j] * x_old[j] for j in range(i)]) - sum([A[i][j] * x_old[j] for j in range(i + 1, self.n)])) / A[i][i]
            R = self.criter(x_new)
            print(f"Итерация номер {iteration}. Норма вектора невязки = {R}")
            if R < self.eps:
                break
            x_old = deepcopy(x_new)      
        return Vector(x_new)
    

    def zeidel_method(self, max_iterations=10**6, w = 1):
        A, b = self.get_canonic_matrix(self.matrix_A, self.vector_b)
        for i in range(self.n):
            diag = A[i][i]
            scale = 1.0 / diag
            for j in range(self.n):
                A[i][j] *= scale
            b[i] *= scale
        x_old, x_new = deepcopy(b), deepcopy(b)
        for iteration in range(max_iterations):
            for i in range(self.n):
                tt = (b[i] - sum([A[i][j] * x_new[j] for j in range(i)]) - sum([A[i][j] * x_old[j] for j in range(i + 1, self.n)])) / A[i][i]
                x_new[i] = (1 - w) * x_old[i] + w * tt
            R = self.criter(x_new)
            print(f"Итерация: {iteration}. R = {R}")
            if R < self.eps:
                break
            x_old = deepcopy(x_new)
        return Vector(x_new)
 
    def get_faster_spusk(self, max_iterations=10**6):
        A, b = Matrix(self.matrix_A), Vector(self.vector_b)
        A_T = A.T()
        A, b = A_T * A, A_T * b
        x_old, x_new = Vector(b.vect), Vector([0] * self.n)
        for iteration in range(max_iterations):
            r = A * x_old - b
            A_r = A * r
            den = A_r * r
            if abs(den) < 10**(-100):
                break
            tau = (r * r) / den
            x_new = x_old - tau * r 
            R = self.criter(x_new.vect)
            print(f"Итерация: {iteration}. R = {R}")
            if R < self.eps:
                break
            x_old = Vector(x_new.vect)
        return x_new
    
    def get_minimal_nevazka(self, max_iterations=10**6):
        A, b = Matrix(self.matrix_A), Vector(self.vector_b)
        A_T = A.T()
        A, b = A_T * A, A_T * b
        x_old, x_new = Vector(b.vect), Vector(b.vect)
        for iteration in range(max_iterations):
            r = A * x_old - b
            A_r = A * r
            den = A_r * A_r
            if abs(den) < 10**(-100):
                break
            tau = (r * A_r) / den
            x_new = x_old - tau * r 
            R = self.criter(x_new.vect)
            print(f"Итерация: {iteration}. R = {R}")
            if R < self.eps:
                break
            x_old = Vector(x_new.vect)
        return x_new
    
    def BSG_method(self, max_iterations=10**6):
        A, b = Matrix(self.matrix_A), Vector(self.vector_b)
        D = [1.0] * self.n    
        for i in range(self.n):
            max_val = max(abs(A.matr[i][j]) for j in range(self.n))
            if max_val > 1e-100: 
                D[i] = max_val
                scale = 1.0 / max_val                
                for j in range(self.n):
                    A.matr[i][j] *= scale                
                b.vect[i] *= scale
            else:
                D[i] = 1.0
        print(f"Масштабирование завершено. Коэффициенты: {D[:5]}...")
        x = Vector(b.vect)
        r = b - A * x 
        r_pred = Vector(r.vect)
        r_tilda = Vector(b.vect)
        r_tilda_pred = Vector(r_tilda.vect)
        p = Vector(r.vect)
        p_tilda = Vector(r_tilda.vect)
        for j in range(max_iterations):
            A_p = A * p
            A_p_tilda = A_p * p_tilda
            if abs(A_p_tilda) < 10**(-200):
                print("Деление на 0_1")
                break 
            alpha = (r * r_tilda) / A_p_tilda
            x = x + alpha * p 
            r_pred = Vector(r.vect)
            r = r - alpha * A_p
            r_tilda_pred = Vector(r_tilda.vect) 
            r_tilda = r_tilda - alpha * A.T() * p_tilda
            r_pred_r_tilda_pred = r_pred * r_tilda_pred
            if abs(r_pred_r_tilda_pred) < 10**(-200):
                print("Деление на 0_2")
                break
            betta = (r * r_tilda) / r_pred_r_tilda_pred
            R = self.criter(x.vect)
            print(f"Итерация = {j}. Норма вектора невязки = {R}")
            if R < self.eps:
                break
            if np.abs(betta) < 10**(-50):
                raise ValueError("Метод не сходится")
            p = r + betta * p 
            p_tilda = r_tilda + betta * p_tilda
        return x