In 1969, V. Strassen presented an algorithm to reduce the complexity of matrix multiplications from O(n^3) to O(n^2.807) using a recursive approach. This repository contains code for various implementations of the original algorithm, testing scripts, and a detailed report on the algorithm.

Files Included:

- **strassen_algorithm.py**:
    This file contains four implementations of the Strassen algorithm. The first is the original version, which only works with matrices where the number of rows and columns is even. The other three approaches handle matrices with odd dimensions:

    Dynamic Padding: Adds a row/column of zeros at each recursive call if the numbers of rows/columns are odd.

    Static Padding: Adds as many rows/columns of zeros as needed during the first call to ensure even dimensions in all recursive calls.
    
    Dynamic Peeling: Removes a row/column at each recursive call if the number of rows/columns is odd. These removed rows/columns are considered later.

- **testing.py**:
    This file is used to test the performance of the three different algorithms for matrices with odd dimensions. It evaluates their numerical error with floating-point numbers, performance across different matrix sizes, handling of matrices with irregular dimensions, performance with sparse and dense matrices, and the effect of varying the "crossover point."

- **Varotto_Luca_report.pdf**:
    This is a report I created about the Strassen algorithm for a bachelor's degree exam.

- **grafici.Rmd**:
    An R Markdown file used to create visualizations and plots for the report.
