import strassen_algorithms as s_a
import numpy as nmp  #per l'implementazione delle matrici
import random        #per generare i numeri casuali
import time          #per tracciare i tempi

#############################################################################
################## FUNZIONI PER IL TESTING DEGLI ALGORITMI ##################
#############################################################################

def generate_2_random_matrix(a,b,c):
    A = nmp.random.randint(low=-100, high=100, size=(a,b), dtype='l') 
    B = nmp.random.randint(low=-100, high=100, size=(b,c), dtype='l')
    return A, B

def different_q(dim_min: int, dim_max: int, n_test:int, file_name:str, qmin, qmax):

    random.seed(1234)
                
    with open(file_name, "w") as f:
        
        f.write(f"ID,tipo,Q,a,b,c,riga_per_colonna,dynamic_time,static_time,peeling_time\n")

        for i in range(0,n_test):

            a = random.randint(dim_min,dim_max)
            b = random.randint(dim_min,dim_max)
            c = random.randint(dim_min,dim_max)

            A, B = generate_2_random_matrix(a,b,c)

            print("\n-----------Matrice numero", i+1, end="-----------\n")

            print("Prodotto riga per colonna...")
            start = time.process_time()
            nmp.matmul(A,B)
            end = time.process_time()
            rbc_time = max(end - start, 1e-6)

            for q in range(qmin,qmax):
                Q = 2**q

                print(f"\n---Q={Q}---")
                f.write(f"{i},rettangolare,{Q},{a},{b},{c},{rbc_time},")
                Strassens_speed_test(A=A,B=B,f=f, Q=(Q))

        f.close()


def generate_sparse_matrix(rows, cols):

    total_elements = rows * cols
    num_nonzero_elements = int(total_elements * 0.5)

    random_values = nmp.random.randint(low=-100, high=100, size=num_nonzero_elements, dtype='l') 
    
    ris = nmp.zeros((rows, cols))
    indices = nmp.random.choice(range(total_elements), size=num_nonzero_elements, replace=False)
    ris.flat[indices] = random_values

    return ris


def sparse_dense_matrix(dim_min: int, dim_max: int, Q:int, n_test:int):
    
    random.seed(1234)
                
    with open("dataset\sparse_vs_dense.txt", "w") as f:
        
        f.write(f"ID,tipo,Q,a,b,c,riga_per_colonna,dynamic_time,static_time,peeling_time\n")

        print("Matrici dense")
        for i in range(0,n_test):

            a = random.randint(dim_min, dim_max)
            b = random.randint(dim_min, dim_max)
            c = random.randint(dim_min, dim_max)

            A, B = generate_2_random_matrix(a,b,c)

            print("\n-----------Matrice numero", i+1, end="-----------\n")

            print("Prodotto riga per colonna...")
            start = time.process_time()
            nmp.matmul(A,B)
            end = time.process_time()
            rbc_time = max(end - start, 1e-6)

            f.write(f"{i},densa,{Q},{a},{b},{c},{rbc_time},")
                
            Strassens_speed_test(A=A,B=B,f=f, Q=(Q))

        print("Matrici sparse")
        for i in range(0,n_test):

            a = random.randint(dim_min, dim_max)
            b = random.randint(dim_min, dim_max)
            c = random.randint(dim_min, dim_max)

            A = generate_sparse_matrix(a,b) 
            B = generate_sparse_matrix(b,c)

            print("\n-----------Matrice numero", i+1, end="-----------\n")

            print("Prodotto riga per colonna...")
            start = time.process_time()
            nmp.matmul(A,B)
            end = time.process_time()
            rbc_time = max(end - start, 1e-6)

            f.write(f"{n_test+i},sparsa,{Q},{a},{b},{c},{rbc_time},")
                
            Strassens_speed_test(A=A,B=B,f=f, Q=(Q))

        f.close()


def lunghe_vs_larghe(dim_min: int, dim_max: int, Q:int, n_test:int):

    random.seed(1234)

    with open("dataset\lunghe_vs_larghe.txt", "w") as f:
        
        f.write(f"ID,tipo,Q,a,b,c,riga_per_colonna,dynamic_time,static_time,peeling_time\n")

        print("########## a piccolo ##########")
        for i in range(0,n_test):

            a = random.randint(dim_min//10, dim_max//10)
            b = random.randint(dim_min, dim_max)
            c = random.randint(dim_min, dim_max)

            A, B = generate_2_random_matrix(a,b,c)

            print("\n-----------Matrice numero", i+1, end="-----------\n")

            print("Prodotto riga per colonna...")
            start = time.process_time()
            nmp.matmul(A,B)
            end = time.process_time()
            rbc_time = max(end - start, 1e-6)

            f.write(f"{i},a_piccolo,{Q},{a},{b},{c},{rbc_time},")
                
            Strassens_speed_test(A=A,B=B,f=f, Q=(Q))


        print("\n########## b piccolo ##########")
        for i in range(0,n_test):

            a = random.randint(dim_min, dim_max)
            b = random.randint(dim_min//10, dim_max//10)
            c = random.randint(dim_min, dim_max)

            A, B = generate_2_random_matrix(a,b,c)

            print("\n-----------Matrice numero", i+1, end="-----------\n")

            print("Prodotto riga per colonna...")
            start = time.process_time()
            nmp.matmul(A,B)
            end = time.process_time()
            rbc_time = max(end - start, 1e-6)

            f.write(f"{i},b_piccolo,{Q},{a},{b},{c},{rbc_time},")
                
            Strassens_speed_test(A=A,B=B,f=f, Q=(Q))
        


        print("\n########## c piccolo ##########")
        for i in range(0,n_test):

            a = random.randint(dim_min, dim_max)
            b = random.randint(dim_min, dim_max)
            c = random.randint(dim_min//10, dim_max//10)

            A, B = generate_2_random_matrix(a,b,c)

            print("\n-----------Matrice numero", i+1, end="-----------\n")

            print("Prodotto riga per colonna...")
            start = time.process_time()
            nmp.matmul(A,B)
            end = time.process_time()
            rbc_time = max(end - start, 1e-6)

            f.write(f"{i},c_piccolo,{Q},{a},{b},{c},{rbc_time},")
                
            Strassens_speed_test(A=A,B=B,f=f, Q=(Q))

        f.close()



def increasing_dimension(dim_min0: int, dim_max0: int, Q:int, n_test:int):

    random.seed(1234)

    with open("dataset\dimensioni.txt", "w") as f:
        
        f.write(f"ID,tipo,Q,a,b,c,riga_per_colonna,dynamic_time,static_time,peeling_time\n")

        for i in range(0,n_test):

            dim_min0 += 10
            dim_max0 += 10

            a = random.randint(dim_min0, dim_max0)
            b = random.randint(dim_min0, dim_max0)
            c = random.randint(dim_min0, dim_max0)

            A, B = generate_2_random_matrix(a,b,c)

            print("\n-----------Matrice numero", i+1, end="-----------\n")

            print("Prodotto riga per colonna...")
            start = time.process_time()
            nmp.matmul(A,B)
            end = time.process_time()
            rbc_time = max(end - start, 1e-6)

            f.write(f"{i},dimensioni_crescenti,{Q},{a},{b},{c},{rbc_time},")
                
            Strassens_speed_test(A=A,B=B,f=f, Q=(Q))


def floating_time(dim_min0: int, dim_max0: int, Q:int, n_test:int):

    random.seed(1234)

    with open("dataset\on_floating.txt", "w") as f:
        
        f.write(f"ID,tipo,Q,a,b,c,riga_per_colonna,dynamic_time,static_time,peeling_time\n")
        for i in range(0,n_test):

            a = random.randint(dim_min0, dim_max0)
            b = random.randint(dim_min0, dim_max0)
            c = random.randint(dim_min0, dim_max0)

            A = nmp.random.uniform(low=-100, high=100, size=(a,b))
            B = nmp.random.uniform(low=-100, high=100, size=(b,c))

            print("\n-----------Matrice numero", i+1, end="-----------\n")

            print("Prodotto riga per colonna...")
            start = time.process_time()
            nmp.matmul(A,B)
            end = time.process_time()
            rbc_time = max(end - start, 1e-6)

            f.write(f"{i},floating,{Q},{a},{b},{c},{rbc_time},")
                
            Strassens_speed_test(A=A,B=B,f=f, Q=(Q))


def floating_error(dim_min0: int, dim_max0: int, Q:int, n_test:int, floating_accuracy):

    random.seed(1234)

    with open("dataset\error_floating.txt", "w") as f:
        
        f.write(f"ID,tipo,Q,a,b,c,riga_per_colonna,dynamic_time,static_time,peeling_time\n")

        for i in range(0,n_test):

            a = random.randint(dim_min0, dim_max0)
            b = random.randint(dim_min0, dim_max0)
            c = random.randint(dim_min0, dim_max0)

            A = nmp.random.uniform(low=-100, high=100, size=(a,b))
            B = nmp.random.uniform(low=-100, high=100, size=(b,c))

            print("\n-----------Matrice numero", i+1, end="-----------\n")
            print("Prodotto riga per colonna...")
            start = time.process_time()
            nmp.matmul(A,B)
            end = time.process_time()
            rbc_time = max(end - start, 1e-6)

            f.write(f"{i},floating,{Q},{a},{b},{c},{rbc_time},")
                
            Strassens_speed_test(A=A,B=B,f=f, Q=(Q))



def Strassens_speed_test(A,B,f,Q):

    print("Strassen padding dinamico...")

    start = time.process_time()
    s_a.Strassen_dynamic_padding(A,B,Q)
    end = time.process_time()
    dynamic_time = max(end - start, 1e-6)

    print("Strassen padding statico...")

    start = time.process_time()
    s_a.Strassen_static_padding(A,B,Q)
    end = time.process_time()
    static_time = max(end - start, 1e-6)

    print("Strassen peeling dinamico...")

    start = time.process_time()
    s_a.Strassen_dynamic_peeling(A,B,Q)
    end = time.process_time()
    peeling_time = max(end - start, 1e-6)

    f.write(f"{dynamic_time},{static_time},{peeling_time}\n")


def Strassens_results(A,B,Q):

    C_RBC = nmp.matmul(A,B)
    C_DPA = s_a.Strassen_dynamic_padding(A,B,Q)
    C_SPA = s_a.Strassen_static_padding(A,B,Q)
    C_DPE = s_a.Strassen_dynamic_peeling(A,B,Q)
    
    return C_RBC, C_DPA, C_SPA, C_DPE


#############################################################################



if __name__ == "__main__":

    #different_q(600, 1000, n_test=30, file_name="dataset\q_test.txt", qmin=4, qmax=10)
    #sparse_dense_matrix(600, 1000, 128, 40)
    lunghe_vs_larghe(800, 1200, 64, 60)
    #increasing_dimension(200,400,128,75)
    #floating_time(600, 1000, 128, 40)
    #floating_error(600, 1000, 128, 40, 8)