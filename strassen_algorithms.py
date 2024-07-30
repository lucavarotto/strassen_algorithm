import numpy as nmp  #per l'implementazione delle matrici
import math          #per i logaritmi e arrotondare i numeri

#############################################################################
###################### ALGORITMO DI STRASSEN ORIGINALE ######################
#############################################################################

# Funzione per dividere una matrice in 4 blocchi di uguali dimensioni
## Argomenti:
## @M, la matrice
### Return:
### @M11, @M12, @M21, @M22, ossia i 4 blocchi
def split_matrix(M):
    mid_r = M.shape[0] // 2
    mid_c = M.shape[1] // 2
    return M[:mid_r, :mid_c], M[:mid_r, mid_c:], M[mid_r:, :mid_c], M[mid_r:, mid_c:]

# Funzione per unire una matrice divisa in 4 blocchi
## Argomenti:
## @M11, @M12, @M21, @M22, ossia i 4 blocchi
### Return:
### @M, la matrice
def merge_matrix(M11, M12, M21, M22):
    return nmp.vstack((nmp.hstack((M11, M12)), nmp.hstack((M21, M22))))

# Funzione che riceve i 7 prodotti indicati da strassen e li somma per
# trovare i blocchi della matrice C
## Argomenti:
## @M1, @M2, @M3, @M4, @M5, @M6, @M7, le matrici ottenute con i prodotti
### Return:
### @C11, @C12, @C21, @C22, ossia i 4 blocchi della matrice dei risultati
def strassen_sums(M1, M2, M3, M4, M5, M6, M7):
    return (M1 + M4 - M5 + M7), (M3 + M5), (M2 + M4), (M1 - M2 + M3 + M6)

# Funzione ricorsiva che esegue la versione
# di base dell'algoritmo di Strassen.
# Accetta quindi 2 matrici di pari dimensioni
# con numero di righe e colonne potenze di 2
## Argomenti:
## @A: la prima matrice della moltiplicazione
## @B: la seconda matrice della moltiplicazione
## @Q: livello a cui fermare la ricorsione
### Return:
### @C: la matrice risultante
def Strassen_squareM(A,B,Q=2):

    #se si supera il crossover point si passa al prodotto riga per colonna
    if A.size[0] <= Q:
        return nmp.matmul(A,B)

    #se una dimensione è dispari non funziona
    if A.size[0]%2==1:
        raise ValueError("Funziona solo con matrici di dimensioni pari")

    A11, A12, A21, A22 = split_matrix(B)
    B11, B12, B21, B22 = split_matrix(B)
    
    M1 = Strassen_squareM(A11+A22, B11+B22)
    M2 = Strassen_squareM(A21+A22, B11)
    M3 = Strassen_squareM(A11, B12-B22)
    M4 = Strassen_squareM(A22, B21-B11)
    M5 = Strassen_squareM(A11+A12, B22)
    M6 = Strassen_squareM(A21-A11, B11+B12)
    M7 = Strassen_squareM(A12-A22, B21+B22)    

    C11, C12, C21, C22 = strassen_sums(M1, M2, M3, M4, M5, M6, M7)

    C = merge_matrix(C11, C12, C21, C22)
    
    return C

#############################################################################
############## ALGORITMO DI STRASSEN PER MATRICI RETTANGOLARI ###############
#############################################################################

# Funzione ricorsiva che esegue la versione
# di base dell'algoritmo di Strassen su matrici rettangolari.
# Accetta quindi 2 matrici di dimensioni pari e che rimangono
# pari ad ogni divisione fino al raggiungimento del crossover point
## Argomenti:
## @A: la prima matrice della moltiplicazione
## @B: la seconda matrice della moltiplicazione
## @Q: livello a cui fermare la ricorsione
## Return:
## @C: la matrice risultante
def Strassen_rectM(A,B,Q):
    
    a, b = A.shape #dimensioni delle matrici quadrate
    c = B.shape[1]
    
    # se viene raggiunto il cross-over point si passa al riga per colonna
    if a <= Q or b <= Q or c <= Q:
        return nmp.matmul(A,B)

    #se una dimensione è dispari non funziona
    if a%2==1 or b%2==1 or c%2==1:
        raise ValueError("Funziona solo con matrici di dimensioni pari")
    
    # partizione di A e B in 4 blocchi di uguali dimensioni
    A11, A12, A21, A22 = split_matrix(A)
    B11, B12, B21, B22 = split_matrix(B)

    #calcolo dei prodotti indicati da strassen
    M1 = Strassen_rectM(A11+A22, B11+B22,Q)
    M2 = Strassen_rectM(A21+A22, B11,Q)
    M3 = Strassen_rectM(A11, B12-B22,Q)
    M4 = Strassen_rectM(A22, B21-B11,Q)
    M5 = Strassen_rectM(A11+A12, B22,Q)
    M6 = Strassen_rectM(A21-A11, B11+B12,Q)
    M7 = Strassen_rectM(A12-A22, B21+B22,Q)    

    #calcolo delle somme indicate da strassen
    C11, C12, C21, C22 = strassen_sums(M1, M2, M3, M4, M5, M6, M7)

    #unione dei risultati
    C = merge_matrix(C11, C12, C21, C22)
    
    return C

#############################################################################
######################## ALGORITMO "DYNAMIC PADDING" ########################
#############################################################################

# Funzione ricorsiva che esegue l'algoritmo di Strassen con l'approccio
# dynamic padding.
# Accetta matrici di qualsiasi dimensione poichè ad ogni chiamata
# aggiunge una riga/colonna di zeri nel caso in cui una delle
# dimensioni sia dispari.
## Argomenti:
## @A: la prima matrice della moltiplicazione
## @B: la seconda matrice della moltiplicazione
## @Q: il cross-over point
### Return:
### @C: la matrice risultante
def Strassen_dynamic_padding(A,B,Q):

    RA, CA = A.shape #variabile che contiene le dimensioni della prima matrice
    RB, CB = B.shape #variabile che contiene le dimensioni della seconda matrice

    # Se le matrici sono più piccole del "crossover point" (Q)
    if (RA*RB*CB)**(1/3) <= Q or min(RA,CA,CB) <= 2:
        # eseguo il classico algoritmo riga per colonna
        return nmp.matmul(A,B)
    
    # Se le matrici NON sono più piccole del "crossover point" (Q)

    drop_c = False #booleano che diventa True se aggiungo almeno una colonna
    drop_r = False #booleano che diventa True se aggiungo almeno una riga

    if RA%2: #se il numero di righe è dispari
        A = nmp.pad(A, ((0,1), (0,0)) )
        drop_r = True #booleano che indica che ho aggiunto una riga
        RA += 1 #variabile che contiene il numero di righe della prima matrice

    if CA%2: #se il numero di colonne è dispari
        
        A = nmp.pad(A, ((0,0), (0,1)) )
        CA += 1 #aggiorno la variabile con le dimensioni di A
        B = nmp.pad(B, ((0,1), (0,0)) )
        RB += 1

    if CB%2: #se il numero di colonne è dispari
        B = nmp.pad(B, ((0,0), (0,1)) )
        drop_c = True
        CB += 1

    # In seguito verrà partizionata la matrice A in blocchi in maniera
    # analoga a quanto fatto da Strassen nel suo paper
    A11, A12, A21, A22 = split_matrix(A)
    B11, B12, B21, B22 = split_matrix(B)
    
    # eseguo i prodotti indicati da Strassen in maniera ricorsiva
    # usando questa stessa funzione ("Strassen_advanced")
    M1 = Strassen_dynamic_padding(A11+A22, B11+B22,Q)
    M2 = Strassen_dynamic_padding(A21+A22, B11,Q)
    M3 = Strassen_dynamic_padding(A11, B12-B22,Q)
    M4 = Strassen_dynamic_padding(A22, B21-B11,Q)
    M5 = Strassen_dynamic_padding(A11+A12, B22,Q)
    M6 = Strassen_dynamic_padding(A21-A11, B11+B12,Q)
    M7 = Strassen_dynamic_padding(A12-A22, B21+B22,Q)    

    # sommmo le matrici come indicato da Strassen
    # per trovare i blocchi della matrice finale
    C11, C12, C21, C22 = strassen_sums(M1, M2, M3, M4, M5, M6, M7)
    
    # unisco i blocchi della matrice finale per
    # trovare la matrice risultante
    C = merge_matrix(C11, C12, C21, C22)

    #controllo se è necessario rimuovere eventuali
    #righe/colonne di zeri
    if drop_r:
        C = nmp.delete(C,RA-1,0) #togli l'ultima riga
    if drop_c:
        C = nmp.delete(C,CB-1,1) #togli l'ultima colonna

    return C

#############################################################################
######################### ALGORITMO "STATIC PADDING" ########################
#############################################################################

# Funzione che restituisce "True" se il valore è una potenza di 2.
## Argomenti:
## @n il numero di interesse.
### Return:
### @booleano, "True" se è potenza di 2, no altrimenti.
def is_power_of_two(n:int) -> bool:
    return math.log(n,2)%1==0

# Funzione per verificare se una dimensione è pari fino al
# raggiungimento del crossover point.
## Argomenti:
## @n, il numero di righe/colonne
## @Q, il cross-over point
### Return:
### @booleano, "True" rimane sempre pari, "False" altrimenti.
def is_even_recursively(n:int, Q:int) -> bool:
    while n > Q:
        if n % 2 == 1:
            return False
        n = n / 2
    return True

# Funzione per calcolare il numero di chiamate ricorsive
# per raggiungere il crossover point.
## Argomenti:
## @n, il numero di righe/colonne
## @Q, il cross-over point
### Return:
### @int, il numero di chiamate ricorsive necessarie.
def n_recursion_needed(n:int,Q:int):
    x = math.log(n/Q,2)
    ris = math.ceil(x)
    return ris

# Funzione per determinare il numero minimo di righe/colonne
# che una matrice deve avere per poter rimanere a dimensioni pari
# fino al raggiungimento del crossover point
## Argomenti:
## @n, il numero di righe/colonne
## @n_rec, il numero di ricorsioni necessarie per superare il crossover point
### Return:
### @int, il numero di righe/colonne necessarie
def n_row_or_col_needed(RMorCM, n_rec):
    return math.ceil(RMorCM/2**n_rec)*2**n_rec

# Funzione che esegue l'algoritmo di Strassen
# chiamando Strassen_naive(A,B,Q) [vedi sopra].
# Accetta matrici di qualsiasi dimensione poichè
# alla prima chiamata le aggiunge abbastanza
# righe/colonne di 0 in modo tale che rimangano di
# dimensioni pari ad ogni chiamata ricorsiva
## Argomenti:
## @A: la prima matrice della moltiplicazione
## @B: la seconda matrice della moltiplicazione
## @Q: il cross-over point
## Return:
## @C: la matrice risultante
def Strassen_static_padding(A,B,Q:int):

    rows_added = 0 #intero che indica le colonne aggiunte
    cols_added = 0 #intero che indica le righe aggiunte
    
    RA, CA = A.shape #variabile che contiene le dimensioni della prima matrice
    RB, CB = B.shape #variabile che contiene le dimensioni della seconda matrice

    # calcolo del numero di ricorsioni necessarie fino al raggiungimento del cross-over point
    z = n_recursion_needed(min(RA, CA, RB, CB), Q)

    # aggiunge il numero di righe a A in base all'esegenza
    if not is_even_recursively(RA,Q):
        x = n_row_or_col_needed(RA, z)
        rows_added = x-RA
        A = nmp.pad(A, ((0,rows_added), (0,0)) )
        RA = x

    # aggiunge il numero di colonne ad A e di righe a B in base all'esigenza
    if not is_even_recursively(CA,Q):
        x = n_row_or_col_needed(CA, z)
        A = nmp.pad(A, ((0,0), (0,x-CA)) )
        B = nmp.pad(B, ((0,x-RB), (0,0)) )
        CA = RB = x

    # aggiunge il numero di colonne a B in base all'esegenza
    if not is_even_recursively(CB,Q):
        x = n_row_or_col_needed(CB, z)
        cols_added = x-CB
        B = nmp.pad(B, ((0,0), (0,cols_added)) )
        CB = x

    C = Strassen_rectM(A,B,Q) #chiamata a Strassen_naive che risolve ricorsivamente
                              #il problema

    #controllo se è necessario rimuovere eventuali righe di zeri
    if rows_added > 0:
        C = nmp.delete(C,list(range(RA-rows_added, RA,1)),0) #togle le righe extra

    #controllo se è necessario rimuovere eventuali colonne di zeri
    if cols_added > 0:
        C = nmp.delete(C,list(range(CB-cols_added, CB,1)),1) #togle le colonne extra

    return C

#############################################################################
######################## ALGORITMO "DYNAMIC PEELING" ########################
#############################################################################

# Funzione per dividere una matrice in due blocchi, di cui uno
# è l'ultima riga
## Argomenti:
## @M, la matrice
### Return:
### @M11, @m21 ossia i 2 blocchi
def split_last_r(M):
    RM, CM = M.shape
    M11 = M[0:(RM-1),:]
    m21 = M[(RM-1),:]
    m21.shape = (1,CM) #fissa le dimensioni per evitare problemi
                            #nei calcoli matriciali successivi
    return M11, m21

# Funzione per dividere una matrice in due blocchi, di cui uno
# è l'ultima colonna
## Argomenti:
## @M, la matrice
### Return:
### @M11, @m12 ossia i 2 blocchi
def split_last_c(M):
    RM, CM = M.shape
    M11 = M[:,0:(CM-1)]
    m12 = M[:,(CM-1)]
    m12.shape = (RM,1) #fissa le dimensioni per evitare problemi
                            #nei calcoli matriciali successivi
    return M11, m12

# Funzione per dividere una matrice M[j,k] in quattro blocchi, di cui uno
# di dimensioni [1,k-1], uno di dimensioni [j-1,1] e uno di dimensioni [1,1]
## Argomenti:
## @M, la matrice
### Return:
### @M11, @m12, @m21, @m22 ossia i 4 blocchi
def split_last_rc(M):
    RM, CM = M.shape
    M11 = M[0:(RM-1),0:(CM-1)]
    m12 = nmp.matrix(M[0:(RM-1),(CM-1)])
    m12.shape = (RM-1,1) #fissa le dimensioni per evitare problemi
                            #nei calcoli matriciali successivi
    m21 = nmp.matrix(M[RM-1,0:(CM-1)])
    m21.shape = (1,CM-1) #fissa le dimensioni per evitare problemi
                            #nei calcoli matriciali successivi
    m22 = nmp.matrix(M[(RM-1), (CM-1)])
    m22.shape = (1,1) #fissa le dimensioni per evitare problemi
                            #nei calcoli matriciali successivi
    return M11, m12, m21, m22

# Funzione che esegue l'algoritmo di Strassen nella versione
# "dynamic peeling". Accetta matrici di qualsiasi dimensione poichè
# ad ogni chiamata toglie una riga/colonna in caso una di queste dimensioni
# sia dispari. Il cross-over point è dato dalla media geometrica delle dimensioni.
## Argomenti:
## @A: la prima matrice della moltiplicazione
## @B: la seconda matrice della moltiplicazione
## @Q: il cross-over point
## Return:
## @C: la matrice risultante
def Strassen_dynamic_peeling(A,B,Q:int):
    
    RA, CA = A.shape #variabile che contiene le dimensioni della prima matrice
    RB, CB = B.shape #variabile che contiene le dimensioni della seconda matrice

    #se la media geometrica delle dimensioni è più piccola del cross-over point
    # o almeno una delle dimensioni è minore o uguali a 2
    if (RA*CA*CB)**(1/3) <= Q  or min(RA,CA,CB) <= 2:
        return nmp.matmul(A,B)

    a_odd = False #se il numero di righe di A è dispari
    b_odd = False #se il numero di colonne di A/numero di righe di B è dispari
    c_odd = False #se il numero di colonne di B è dispari

    if RA%2: #se il numero di righe di A è dispari
        a_odd = True

    if CA%2: #se il numero di colonne di A è dispari
    #il numero di righe di B è uguale al numero di colonne di A
        b_odd = True

    if CB%2: #se il numero di colonne di B è dispari
        c_odd = True

    # 8 combinazioni possibili:

    # 1)
    # solo prima dimensione dispari
    if a_odd and (not b_odd) and (not c_odd):
        
        #svolgimento delle operazioni come indicate in:
        #"Implementation of Strassen’s Algorithm for Matrix Multiplication"
        A11, a21 = split_last_r(A)
        C11 = Strassen_dynamic_peeling(A11,B,Q)

        C21 = nmp.matmul( a21, B)
        return nmp.vstack((C11,C21))

    # 2)
    # solo seconda dimensione dispari
    elif (not a_odd) and b_odd and (not c_odd):

        #svolgimento delle operazioni come indicate in:
        #"Implementation of Strassen’s Algorithm for Matrix Multiplication"
        A11, a12 = split_last_c(A)
        B11, b21 = split_last_r(B)

        C = Strassen_dynamic_peeling(A11,B11,Q)
        C += nmp.matmul(a12, b21)
        return C
    
    # 3)
    # solo terza dimensione dispari
    elif (not a_odd) and (not b_odd) and c_odd:

        #svolgimento delle operazioni come indicate in:
        #"Implementation of Strassen’s Algorithm for Matrix Multiplication"
        B11, b12 = split_last_c(B)

        C11 = Strassen_dynamic_peeling(A, B11, Q)
        C12 = nmp.matmul(A, b12)
        return nmp.hstack((C11,C12))

    # 4)
    #solo prima e seconda dimensione dispari
    elif a_odd and b_odd and (not c_odd):

        #svolgimento delle operazioni come indicate in:
        #"Implementation of Strassen’s Algorithm for Matrix Multiplication"
        A11, a12, a21, a22 = split_last_rc(A)
        B11, b21 = split_last_r(B)

        C11 = Strassen_dynamic_peeling( A11, B11, Q)
        C11 += nmp.matmul( a12, b21 )

        C21 = nmp.matmul( a21, B11 ) + nmp.matmul( a22, b21 )

        return nmp.vstack((C11,C21))
    
    # 5)
    #solo prima e terza dimensione dispari
    elif a_odd and (not b_odd) and c_odd:

        #svolgimento delle operazioni come indicate in:
        #"Implementation of Strassen’s Algorithm for Matrix Multiplication"
        A11, a21 = split_last_r(A)
        B11, b12 = split_last_c(B)

        C11 = Strassen_dynamic_peeling(A11, B11, Q)
        C12 = nmp.matmul(A11, b12)
        C21 = nmp.matmul(a21, B11)
        C22 = nmp.matmul(a21, b12 )

        return merge_matrix(C11,C12,C21,C22)
    
    # 6)
    #solo seconda e terza dimensione dispari
    elif (not a_odd) and b_odd and c_odd:

        #svolgimento delle operazioni come indicate in:
        #"Implementation of Strassen’s Algorithm for Matrix Multiplication"
        A11, a12 = split_last_c(A)
        B11, b12, b21, b22 = split_last_rc(B)

        C11 = Strassen_dynamic_peeling( A11, B11, Q)
        C11 += nmp.matmul( a12, b21 )

        C21 = nmp.matmul( A11, b12 ) + nmp.matmul( a12, b22 )

        return nmp.hstack((C11,C21))
    
    # 7)
    #tutte le dimensioni dispari
    elif a_odd and b_odd and c_odd:

        #svolgimento delle operazioni come indicate in:
        #"Implementation of Strassen’s Algorithm for Matrix Multiplication"
        A11, a12, a21, a22 = split_last_rc(A)
        B11, b12, b21, b22 = split_last_rc(B)

        C11 = Strassen_dynamic_peeling( A11, B11, Q)
        C11 += nmp.matmul( a12, b21 )

        C12 = nmp.matmul(A11, b12) + nmp.matmul(a12, b22)

        C21 = nmp.matmul( a21, B11 ) + nmp.matmul( a22, b21 )

        C22 = nmp.matmul(a21,b12) + nmp.matmul(a22, b22)

        return merge_matrix(C11,C12,C21,C22)

    # 8:
    # tutte le dimensioni sono pari
    elif (not a_odd) and (not b_odd) and (not c_odd):
        
        #svolgimento delle operazioni come indicate in:
        #"Implementation of Strassen’s Algorithm for Matrix Multiplication"
        A11, A12, A21, A22 = split_matrix(A)
        B11, B12, B21, B22 = split_matrix(B)

        M1 = Strassen_dynamic_peeling(A11+A22, B11+B22,Q)
        M2 = Strassen_dynamic_peeling(A21+A22, B11,Q)
        M3 = Strassen_dynamic_peeling(A11, B12-B22,Q)
        M4 = Strassen_dynamic_peeling(A22, B21-B11,Q)
        M5 = Strassen_dynamic_peeling(A11+A12, B22,Q)
        M6 = Strassen_dynamic_peeling(A21-A11, B11+B12,Q)
        M7 = Strassen_dynamic_peeling(A12-A22, B21+B22,Q)    

        C11, C12, C21, C22 = strassen_sums(M1, M2, M3, M4, M5, M6, M7)

        C = merge_matrix(C11, C12, C21, C22)
        
        return C
    
    #se non rientra in nessuno dei casi precedenti
    raise ValueError("Errore nel dynamic peeling")