import logging as logg
import os

import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange
from numba.typed import List
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
import random

import time
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import os
import json
import time

def readAlist(filename):
    """
    Read an ALIST file and reconstruct the binary parity-check matrix H.

    Parameters
    ----------
    filename : str
        Path to the ALIST file.

    Returns
    -------
    H : ndarray of shape (m, n)
        Reconstructed binary parity-check matrix.
    """
    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    n, m = map(int, lines[0].split())
    var_conn_lines = lines[4 : 4 + n]

    H = np.zeros((m, n), dtype=np.uint8)

    for j, line in enumerate(var_conn_lines):
        for entry in map(int, line.split()):
            if entry > 0:
                H[entry - 1, j] = 1

    return csr_matrix(H)

@njit


def writeAlist(H, filename):
    """
    Save a binary parity-check matrix H (numpy array) to ALIST format.

    Parameters
    ----------
    H : ndarray of shape (m, n)
        Binary parity-check matrix.
    filename : str
        Name of the ALIST file to be written.
    """
    # A função 'todense' gera a matriz 'completa' com os zeros
    if type(H) == csr_matrix:
        H = csr_matrix.todense(H).astype(np.uint8)
    elif type(H) == csc_matrix:
        H = csc_matrix.todense(H).astype(np.uint8)
    elif type(H) == coo_matrix:
        H = coo_matrix.todense(H).astype(np.uint8)
    else:
        H = H.astype(np.int8)

    m, n = H.shape

    # Variable and check node degrees
    varDegrees = [int(np.sum(H[:, j])) for j in range(n)]
    checkDegrees = [int(np.sum(H[i, :])) for i in range(m)]
    maxColDeg = max(varDegrees)
    maxRowDeg = max(checkDegrees)

    with open(filename, "w") as f:
        f.write(f"{n} {m}\n")
        f.write(f"{maxColDeg} {maxRowDeg}\n")

        f.write(" ".join(str(d) for d in varDegrees) + "\n")
        f.write(" ".join(str(d) for d in checkDegrees) + "\n")

        # Variable node connections (1-based indexing)
        for j in range(n):
            connections = np.where(H[:, j] == 1)[0] + 1
            padded = list(connections) + [0] * (maxColDeg - len(connections))
            f.write(" ".join(str(i) for i in padded) + "\n")

        # Check node connections (1-based indexing)
        for i in range(m):
            connections = np.where(H[i, :] == 1)[1] + 1
            padded = list(connections) + [0] * (maxRowDeg - len(connections))
            f.write(" ".join(str(j) for j in padded) + "\n")

    f.close()


@njit(parallel=True)


def sumProductAlgorithm(llrs, checkNodes, varNodes, syndrome_y, maxIter, prec=np.float32):
    """
    Performs belief propagation decoding using the sum-product algorithm (SPA) for multiple codewords.

    Parameters
    ----------
    llrs : ndarray of shape (n, numCodewords)
        Array of log-likelihood ratios (LLRs) for each bit of the received codeword.
    checkNodes : list of ndarray
        List of length :math:`m`, where each element is a 1D array containing the indices
        of variable nodes (bits) involved in the corresponding check node (parity-check equation).
    varNodes : list of ndarray
        List of length :math:`n`, where each element is a 1D array containing the indices
        of check nodes that the corresponding variable node participates in.
    maxIter : int
        Maximum number of belief propagation iterations.
    prec : data-type, optional
        Data type for the computations (default is np.float32).

    Returns
    -------
    finalLLR : ndarray of shape (n,)
        Updated log-likelihood ratios after message passing.
    numIter : int
        Number of iterations executed until decoding converged or reached `maxIter`.
    frameDecodingFail : ndarray of shape (numCodewords,)
        Array indicating whether decoding was successful (0) or failed (1) for each codeword.
        A value of 0 indicates successful decoding, while 1 indicates failure.

    References
    ----------
    [1] F. R. Kschischang, B. J. Frey and H. . -A. Loeliger, "Factor graphs and the sum-product algorithm," IEEE Transactions on Information Theory, vol. 47, no. 2, pp. 498-519, Feb 2001.

    [2] T. J. Richardson and R. L. Urbanke, "The capacity of low-density parity-check codes under message-passing decoding," IEEE Transactions on Information Theory, vol. 47, no. 2, pp. 599-618, Feb 2001.
    """
    m, n = len(checkNodes), len(varNodes)
    msgVtoC = np.zeros((m, n), dtype=prec)
    msgCtoV = np.zeros((m, n), dtype=prec)
    llrs = llrs.astype(prec)
    print('Shape llrs: ', np.shape(llrs))

    #numCodewords = llrs.shape[1]
    numCodewords = 1
    finalLLR = np.zeros(n, dtype=prec)
    #frameDecodingFail = np.ones((numCodewords,), dtype=np.int8)
    #lastIter = np.zeros((numCodewords,), dtype=np.uint32)

    #for indCw in range(numCodewords):
    decodedBits = np.zeros(n, dtype=np.uint8)
    llr = llrs.copy()
        # Initialize variable-to-check messages with input LLRs
    for var in prange(n):
        for check in varNodes[var]:
            msgVtoC[check, var] = llr[var]

    for indIter in range(maxIter):
        # Check-to-variable update
        for check in prange(m):
            for var_idx in range(len(checkNodes[check])):
                var = checkNodes[check][var_idx]
                product = 1.0
                for neighbor_idx in range(len(checkNodes[check])):
                    neighbor = checkNodes[check][neighbor_idx]
                    if neighbor != var:
                        product *= np.tanh(msgVtoC[check, neighbor] / 2)
                product *= (1-2*syndrome_y[check]) #Liveris et. al term
                product = min(0.999999, max(-0.999999, product))  # clip
                msgCtoV[check, var] = 2 * np.arctanh(product)

        # Variable-to-check update
        for var in prange(n):
            for check_idx in range(len(varNodes[var])):
                check = varNodes[var][check_idx]
                sumMsg = llr[var]
                for neighbor_idx in range(len(varNodes[var])):
                    neighbor = varNodes[var][neighbor_idx]
                    if neighbor != check:
                        sumMsg += msgCtoV[neighbor, var]
                msgVtoC[check, var] = sumMsg

        # Final LLR computation
        for var in prange(n):
            finalLLR[var] = llr[var]
            for check in varNodes[var]:
                finalLLR[var] += msgCtoV[check, var]
                decodedBits[var] = (-np.sign(finalLLR[var]) + 1) // 2

        # Check parity conditions (cálculo da síndrome)
        parity_checks = np.zeros(m, dtype=np.uint8)
        for indParity in prange(m):
            for check in checkNodes[indParity]:
                    parity_checks[indParity] ^= decodedBits[check]  # accumulate XORs
        #print('Soma parity: ', np.sum(parity_checks))

        #Verificação com a síndrome de Bob
        success = False
        if np.all(parity_checks == syndrome_y):
          print('Síndromes igualadas!')
          #frameDecodingFail = 0
          lastIter = indIter
          success = True
          break

            #if np.sum(parity_checks) == 0:
            #    frameDecodingFail[indCw] = 0
            #    lastIter[indCw] = indIter
            #    break

        if indIter == maxIter - 1:
            lastIter = indIter

    return finalLLR, lastIter, decodedBits, success
    #return finalLLR


@njit(parallel=True, fastmath=True)



def minSumAlgorithm(llrs, checkNodes, varNodes, maxIter, prec=np.float32):
    """
    Performs belief propagation decoding using the Min-Sum Algorithm (MSA) for multiple codewords.

    Parameters
    ----------
    llrs : ndarray of shape (n, numCodewords)
        Log-likelihood ratios (LLRs) of the received codeword bits.
    checkNodes : list of ndarray
        List of length :math:`m`, where each entry contains the indices of variable nodes
        connected to the corresponding check node.
    varNodes : list of ndarray
        List of length :math:`n`, where each entry contains the indices of check nodes
        connected to the corresponding variable node.
    maxIter : int
        Maximum number of iterations for belief propagation.
    prec : data-type, optional
        Numerical precision to use in computations (default is np.float32).

    Returns
    -------
    finalLLR : ndarray of shape (n,)
        Updated LLR values for the decoded codeword after the final iteration.
    numIter : int
        Number of iterations performed before successful decoding or reaching `maxIter`.
    frameDecodingFail : ndarray of shape (numCodewords,)
        Array indicating whether decoding was successful (0) or failed (1) for each codeword.
        A value of 0 indicates successful decoding, while 1 indicates failure.

    References
    ----------
    [1] M. P. C. Fossorier, M. Mihaljevic and H. Imai, "Reduced complexity iterative decoding of low-density parity check codes based on belief propagation," IEEE Transactions on Communications, vol. 47, no. 5, pp. 673-680, May 1999
    """
    m, n = len(checkNodes), len(varNodes)
    msgVtoC = np.zeros((m, n), dtype=prec)
    msgCtoV = np.zeros((m, n), dtype=prec)
    llrs = llrs.astype(prec)

    #numCodewords = llrs.shape[1]
    numCodewords = 1
    finalLLR = np.zeros((n, numCodewords), dtype=prec)
    frameDecodingFail = np.ones((numCodewords,), dtype=np.int8)
    lastIter = np.zeros((numCodewords,), dtype=np.uint32)

    for indCw in range(numCodewords):
        decodedBits = np.zeros(n, dtype=np.uint8)
        llr = llrs[:, indCw]

        # Initialize variable-to-check messages with input LLRs
        for var in prange(n):
            for check in varNodes[var]:
                msgVtoC[check, var] = llr[var]

        for indIter in range(maxIter):
            # Check-to-variable update (Min-Sum)
            for check in prange(m):
                for var in checkNodes[check]:
                    signProduct = 1
                    min_abs = np.inf
                    for neighbor in checkNodes[check]:
                        if neighbor != var:
                            val = msgVtoC[check, neighbor]
                            signProduct *= np.sign(val)
                            min_abs = min(min_abs, abs(val))
                    msgCtoV[check, var] = signProduct * min_abs

            # Variable-to-check update
            for var in prange(n):
                for check in varNodes[var]:
                    sumMsg = llr[var]
                    for neighbor in varNodes[var]:
                        if neighbor != check:
                            sumMsg += msgCtoV[neighbor, var]
                    msgVtoC[check, var] = sumMsg

            # Final LLR and decision
            for var in prange(n):
                finalLLR[var, indCw] = llr[var]
                for check in varNodes[var]:
                    finalLLR[var, indCw] += msgCtoV[check, var]
                decodedBits[var] = (-np.sign(finalLLR[var, indCw]) + 1) // 2

            # Check parity conditions
            parity_checks = np.zeros(m, dtype=np.uint8)
            for indParity in prange(m):
                for check in checkNodes[indParity]:
                    parity_checks[indParity] ^= decodedBits[check]  # accumulate XORs

            if np.sum(parity_checks) == 0:
                frameDecodingFail[indCw] = 0
                lastIter[indCw] = indIter
                break

            if indIter == maxIter - 1:
                lastIter[indCw] = indIter

    return finalLLR, lastIter, frameDecodingFail



def decodeLDPC(llrs, nIter, H_LDPC, syndrome_y):
    """
    Decode multiple LDPC codewords using the belief propagation algorithms.

    Parameters
    ----------
    llrs : ndarray of shape (n, numCodewords)
        Array of log-likelihood ratios (LLRs) for each bit of the received codewords.
        Codewords are assumed to be disposed in columns.
    param : object
        Object containing the following attributes:

        - H : ndarray of shape (m, n)
            Sparse binary parity-check matrix of the LDPC code.

        - maxIter : int
            Maximum number of iterations for belief propagation.

        - alg : str
            Decoding algorithm to use ('SPA' for sum-product or 'MSA' for min-sum).

        - prec : data-type
            Numerical precision to use in computations (default is np.float32).

    Returns
    -------
    decodedBits : ndarray of shape (n, numCodewords)
        Array of decoded bits for each codeword.
    outputLLRs : ndarray of shape (n, numCodewords)
        Array of updated log-likelihood ratios (LLRs) after decoding.
    """
    # check input parameters
    prgsBar = True
    H = H_LDPC
    maxIter = nIter
    alg = "SPA"
    prec = np.float32

    if H is None:
        logg.error("H is None. Please provide a valid parity-check matrix.")

    m, n = H.shape
    #numCodewords = llrs.shape[1]
    numCodewords = 1
    #n_ = llrs.shape[0]
    n_ = len(llrs)
    Hcsc = H.tocsc()  # convert to CSC format for efficient column access

    llrs = np.clip(llrs, -200, 200)
    outputLLRs = np.zeros_like(llrs, dtype=prec)

    # depuncturing LLRs if necessary
    if n_ < n:
        llrs = np.pad(llrs, ((0, n - n_), (0, 0)), mode="constant")

    # Build adjacency lists using fixed-size lists for Numba
    checkNodes = List([H[i].indices.astype(np.uint32) for i in range(m)])
    varNodes = List([Hcsc[:, j].indices.astype(np.uint32) for j in range(n)])

    logg.info(f"Decoding {numCodewords} LDPC codewords with {alg}")
    if alg == "SPA":
        outputLLRs, lastIter, decodedBitsSPA, success = sumProductAlgorithm(
        #outputLLRs = sumProductAlgorithm(
          llrs, checkNodes, varNodes, syndrome_y, maxIter, prec
        )
    elif alg == "MSA":
        outputLLRs, lastIter, frameErrors = minSumAlgorithm(
            llrs, checkNodes, varNodes, maxIter, prec
        )
    else:
        logg.error(f"Unsupported algorithm: {alg}. Supported algorithms are: SPA, MSA.")
        return None, None


    return decodedBitsSPA, lastIter, success

#Distributional Transform Expansion Functions

def ecdf(sample):

    # convert sample to a numpy array, if it isn't already
    sample = np.atleast_1d(sample)

    # find the unique values and their corresponding counts
    quantiles, counts = np.unique(sample, return_counts=True)

    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between 0 and 1
    cumprob = np.cumsum(counts).astype(np.double) / sample.size

    return quantiles, cumprob

def d2b(n):

    strtemp = ''
    while n != 0:
        resto = str(int(n%2))
        strtemp = resto + '' + strtemp
        n = np.floor(n/2)

    return strtemp

#teste = d2b(8)
#print (teste, type(teste))
# ----------------------------------------------------------------------------
#Convert real numbers not integers to binary sequences.

#n = 10.7560

def f_d2b(n, bits_de_representacao):

    #bits_de_representacao = 3
    strn = str(n)
    #strn = strn.strip()
    strn = strn.replace(' ', '')

    if strn.find('.') == (-1):
        return d2b(n)
    else:
        k = strn.find('.')

    i_part = strn[0:k]
    f_part = strn[k::]

    number_i_part = int(i_part)
    number_f_part = float(f_part)

    bin_i_part = d2b(number_i_part)


    strtemp = ''
    temp = number_f_part
    #t = '1'
    #s = '0'

    aux = 0
    inf = 0
    sup = 1
    media = (sup - inf) / 2

    while aux < bits_de_representacao:

        if temp >= media:

            strtemp = strtemp + '1'
            inf = media
            media = ((sup - inf) / 2) + inf

        else:

            strtemp = strtemp + '0'
            sup = media
            media = ((sup - inf) / 2) + inf

        aux += 1

    if(i_part == '0'):
        return strtemp
    else:
        return (bin_i_part + '.' + strtemp)

# ============================================================
# Parâmetros globais
# ============================================================

V_mod_tilde = 1
excess_noise = 0.02
tau = np.arange(0.02, 0.28, 0.02, dtype=np.float32)
D = -np.log10(tau) * (10 / 0.2)
V_mod = 4 * V_mod_tilde

SNR_list = (tau * (V_mod) / (1 + excess_noise)).astype(np.float32)
SNRdB_list = (10 * np.log10(SNR_list)).astype(np.float32)

mu0 = 0
sigma = 1

realizacoes = 30000
bits_de_representacao = 2

# Carregue sua H aqui (idealmente esparsa)
H = readAlist("LDPC_DVBS2_30000b_R002.txt")

# ============================================================
# Funções auxiliares
# ============================================================

def generate_first_bit_from_ecdf(samples):
    """
    Gera apenas o PRIMEIRO bit da expansão binária da probabilidade acumulada.
    Isso replica sua lógica essencial, mas evita armazenar a matriz completa.
    
    Observação:
    - Ainda usa ECDF/ranking empírico
    - Muito mais leve que montar Matriz_Alice/Matriz_Bob inteira
    """
    n = len(samples)

    # ranking empírico
    order = np.argsort(samples, axis=0).reshape(-1)
    ranks = np.empty(n, dtype=np.int32)
    ranks[order] = np.arange(n, dtype=np.int32)

    # prob acumulada aproximada
    p = (ranks + 1) / n
    p = p.astype(np.float32)

    # evitar 0 e 1
    p = np.clip(p, 1e-5, 0.99999)

    # Como bits_de_representacao = 2 e você só usa o primeiro bit,
    # o primeiro bit da expansão binária de p é equivalente a:
    # bit = 1 se p >= 0.5, senão 0
    #
    # Isso corresponde ao primeiro dígito binário fracionário.
    first_bit = (p >= 0.5).astype(np.int8)

    return first_bit


def simulate_one_seed(seed, realizacoes, bits_de_representacao, mu0, sigma, tau, H):
    """
    Executa 1 seed completa (todos os SNRs) e retorna os acumuladores.
    """
    rng = np.random.default_rng(seed)
    #random.seed(seed)
    #np.random.seed(seed)

    snrs = len(tau)

    Pe_sum = np.zeros(snrs, dtype=np.float64)
    frame_error = np.zeros(snrs, dtype=np.float64)
    sum_iterations = np.zeros(snrs, dtype=np.float64)
    syndromes_match = np.zeros(snrs, dtype=np.float64)   # success do decoder
    full_match = np.zeros(snrs, dtype=np.float64)        # decodedBits == y

    # ========================================================
    # Alice
    # ========================================================
    M = rng.normal(mu0, sigma, size=30000).astype(np.float32)
    #M = np.random.normal(mu0, sigma, (100000, 1)).astype(np.float32)
    #x = M
    x = generate_first_bit_from_ecdf(M)

    """kx = np.zeros(realizacoes, dtype = int)
    probabilidade_acumulada_x = np.zeros(realizacoes)

    Matriz_Alice = np.zeros((realizacoes, bits_de_representacao), dtype=np.int8)
    # compute the ECDF of the samples
    x1, px = ecdf(x)

    for i in range(0, realizacoes):

        indices = np.where(x1 == x[i])[0]  # Get the indexes where x1 == x[i]

        if indices.size > 0:  # Garante que há pelo menos um índice correspondente
            kx[i] = int(indices[0])  # Obtém o primeiro índice encontrado
            probabilidade_acumulada_x[i] = px[kx[i]]  # Atribui a probabilidade acumulada

        #foi gerado por Alice
        probabilidade_acumulada_x = np.clip(probabilidade_acumulada_x, 1e-5, 0.99999)
        
        if probabilidade_acumulada_x[i] == 1:
            probabilidade_acumulada_x[i] = 0.99999
        elif probabilidade_acumulada_x[i] == 0:
            probabilidade_acumulada_x[i] = 0.00001

        expansao_base2_Alice_str = f_d2b(probabilidade_acumulada_x[i], bits_de_representacao)

        ii = 0
        for a in expansao_base2_Alice_str:
            Matriz_Alice[i,ii] = a
            ii += 1

    #print(np.shape(Matriz_Alice))

    # Gerando x e s(x) ---------------------------------------------------------------------------------------
    x = np.copy(Matriz_Alice[:,0])"""

    # ========================================================
    # Loop em SNR
    # ========================================================
    for glob in range(snrs):
        sigmar = np.sqrt((1.02) / (tau[glob] * 4)).astype(np.float32)
        Mr = rng.normal(mu0, sigmar, size=30000).astype(np.float32)
        #Mr = np.random.normal(mu0, sigmar, (100000, 1)).astype(np.float32) #ruído

        y = M + Mr
        y = generate_first_bit_from_ecdf(y)
        """y1, py = ecdf(y)

        ky = np.zeros(realizacoes, dtype = int)
        probabilidade_acumulada_y = np.zeros(realizacoes)
        Matriz_Bob = np.zeros((realizacoes, bits_de_representacao), dtype = np.int8)

        #Geração da matriz binária de Bob

        for ii in range(0, realizacoes):

            indices_y = np.where(y1 == y[ii])[0]
            if indices_y.size > 0:  # Garante que há pelo menos um índice correspondente
                ky[ii] = int(indices_y[0])  # Obtém o primeiro índice encontrado
                probabilidade_acumulada_y[ii] = py[ky[ii]]  # Atribui a probabilidade acumulada

            #foi recebido por Bob

            probabilidade_acumulada_y = np.clip(probabilidade_acumulada_y, 1e-5, 0.99999)

            if probabilidade_acumulada_y[ii] == 1:
                probabilidade_acumulada_y[ii] = 0.99999
            elif probabilidade_acumulada_y[ii] == 0:
                probabilidade_acumulada_y[ii] = 0.00001

            expansao_base2_Bob_str = f_d2b(probabilidade_acumulada_y[ii], bits_de_representacao)
            #Bob = []
            iii = 0
            for b in expansao_base2_Bob_str:
                #print("b", b)
                Matriz_Bob[ii,iii] = b
                iii = iii + 1

        # Gerando y e s(y) --------------------------------------------------------------------
        y = np.copy(Matriz_Bob[:,0])"""

        # BER antes da decodificação
        d_hamming = np.bitwise_xor(x, y).astype(np.int8)
        dist = np.sum(d_hamming)
        Pe = dist / realizacoes

        # proteção numérica
        eps = 1e-12
        Pe_clip = np.clip(Pe, eps, 1 - eps)

        Pe_sum[glob] += Pe

        # síndrome
        syndromeBitsY = H.dot(y) % 2

        # LLRs
        llrs = (1 - 2 * x) * np.log((1 - Pe_clip) / Pe_clip)

        nIter = 50
        decodedBits, numberIter, success = decodeLDPC(llrs, nIter, H, syndromeBitsY)

        if success:
            syndromes_match[glob] += 1

        if np.array_equal(decodedBits, y):
            full_match[glob] += 1
        else:
            frame_error[glob] += 1

        sum_iterations[glob] += numberIter

    return {
        "Pe_sum": Pe_sum,
        "frame_error": frame_error,
        "sum_iterations": sum_iterations,
        "syndromes_match": syndromes_match,
        "full_match": full_match,
    }


def worker_block(seed_list, realizacoes, bits_de_representacao, mu0, sigma, tau, H):
    """
    Worker que processa um bloco de seeds.
    """
    snrs = len(tau)

    Pe_sum_total = np.zeros(snrs, dtype=np.float64)
    frame_error_total = np.zeros(snrs, dtype=np.float64)
    sum_iterations_total = np.zeros(snrs, dtype=np.float64)
    syndromes_match_total = np.zeros(snrs, dtype=np.float64)
    full_match_total = np.zeros(snrs, dtype=np.float64)

    for seed in seed_list:
        out = simulate_one_seed(
            seed=seed,
            realizacoes=realizacoes,
            bits_de_representacao=bits_de_representacao,
            mu0=mu0,
            sigma=sigma,
            tau=tau,
            H=H
        )

        Pe_sum_total += out["Pe_sum"]
        frame_error_total += out["frame_error"]
        sum_iterations_total += out["sum_iterations"]
        syndromes_match_total += out["syndromes_match"]
        full_match_total += out["full_match"]

    return {
        "Pe_sum": Pe_sum_total,
        "frame_error": frame_error_total,
        "sum_iterations": sum_iterations_total,
        "syndromes_match": syndromes_match_total,
        "full_match": full_match_total,
        "n_runs": len(seed_list)
    }


def split_seeds(seed_start, total_runs, n_blocks):
    seeds = list(range(seed_start, seed_start + total_runs))
    return np.array_split(seeds, n_blocks)


def run_parallel_simulation(
    realizacoes,
    bits_de_representacao,
    mu0,
    sigma,
    tau,
    H,
    seed_start=8759023,
    total_runs=4,
    n_workers=None,
    output_dir="results_parallel"
):
    t0 = time.time()

    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    os.makedirs(output_dir, exist_ok=True)

    seed_blocks = split_seeds(seed_start, total_runs, n_workers)

    snrs = len(tau)

    Pe_sum_global = np.zeros(snrs, dtype=np.float64)
    frame_error_global = np.zeros(snrs, dtype=np.float64)
    sum_iterations_global = np.zeros(snrs, dtype=np.float64)
    syndromes_match_global = np.zeros(snrs, dtype=np.float64)
    full_match_global = np.zeros(snrs, dtype=np.float64)

    completed_runs = 0

    print(f"Executando {total_runs} seeds com {n_workers} processos...")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(
                worker_block,
                list(block),
                realizacoes,
                bits_de_representacao,
                mu0,
                sigma,
                tau,
                H
            )
            for block in seed_blocks if len(block) > 0
        ]

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()

            Pe_sum_global += result["Pe_sum"]
            frame_error_global += result["frame_error"]
            sum_iterations_global += result["sum_iterations"]
            syndromes_match_global += result["syndromes_match"]
            full_match_global += result["full_match"]
            completed_runs += result["n_runs"]

            print(f"run_atual = {completed_runs}/{total_runs} | bloco {i}/{len(futures)} finalizado")

            print(f"[{i}/{len(futures)}] bloco concluído | seeds acumuladas = {completed_runs}")

    # ========================================================
    # Médias finais
    # ========================================================
    BER_media = Pe_sum_global / total_runs
    FER_media = frame_error_global / total_runs
    Iter_media = sum_iterations_global / total_runs
    Syndromes_media = syndromes_match_global / total_runs
    FullMatch_media = full_match_global / total_runs

    # salvar
    np.savetxt(os.path.join(output_dir, "BER_media.txt"), BER_media, fmt="%.10e")
    np.savetxt(os.path.join(output_dir, "FER_media.txt"), FER_media, fmt="%.10e")
    np.savetxt(os.path.join(output_dir, "Iter_media.txt"), Iter_media, fmt="%.10e")
    np.savetxt(os.path.join(output_dir, "Syndromes_media.txt"), Syndromes_media, fmt="%.10e")
    np.savetxt(os.path.join(output_dir, "FullMatch_media.txt"), FullMatch_media, fmt="%.10e")

    elapsed = time.time() - t0
    print(f"\nTempo total: {elapsed/60:.2f} min")

    return {
        "BER_media": BER_media,
        "FER_media": FER_media,
        "Iter_media": Iter_media,
        "Syndromes_media": Syndromes_media,
        "FullMatch_media": FullMatch_media,
        "tempo_s": elapsed
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    resultados = run_parallel_simulation(
        realizacoes=realizacoes,
        bits_de_representacao=bits_de_representacao,
        mu0=mu0,
        sigma=sigma,
        tau=tau,
        H=H,
        seed_start=8759023,
        total_runs=4,
        n_workers=2,   # ajuste conforme sua máquina
        output_dir="results_parallel"
    )

    print("\nSimulação concluída.")
    print("BER média:", resultados["BER_media"])
    print("FER média:", resultados["FER_media"])
    print("Número médio de iterações:", resultados["Iter_media"])
    print("Número médio de síndromes igualadas:", resultados["Syndromes_media"])


"""Pe_media = Pe_media/max_iterations
print(f'Pe media: ', Pe_media)
Capacidade_media = 1-Entropia
print(f"Capacidade média: {Capacidade_media}.")

mean_of_iterations = mean_of_iterations / max_iterations
cont_sucessos = cont_sucessos / max_iterations
success_full = success_full / max_iterations

np.savetxt('Mean_of_iterations_60000.txt', mean_of_iterations, fmt='%f')
np.savetxt('Cont_sucess_60000.txt', cont_sucessos, fmt='%f')
np.savetxt('Sequencias_igualadas_60000.txt', success_full, fmt='%f')"""