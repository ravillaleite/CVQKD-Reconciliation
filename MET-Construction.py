#Funções para o cálculo de N e M compatíveis com o polinômio e a taxa fornecidos.

import math
from fractions import Fraction
from typing import Dict, Tuple, List, Optional
import numpy as np
import random

DegreeVec = Tuple[int, ...]
PolyNode = Dict[DegreeVec, float]

#O _ no nome das funções indica que são funções internas
#Reconstruir números racionais a partir de aproximações numéricas
#Evitar erros acumulados de ponto flutuante

def _as_fraction(x: float, max_den: int = 10_000) -> Fraction:
    # converte float -> fração racional aproximada
    return Fraction(x).limit_denominator(max_den)

# ------------------------------------------------------------------

#Testa a consistência das probabilidades fornecidas no polinômio

def _normalize(poly: PolyNode) -> Dict[DegreeVec, Fraction]:
    if not poly:
        raise ValueError("Polinômio vazio.")
    s = sum(poly.values())
    if s <= 0:
        raise ValueError("Soma das probabilidades <= 0.")
    out = {}
    for k, v in poly.items():
        if v < 0:
            raise ValueError("Probabilidade negativa no polinômio.")
        out[k] = _as_fraction(v / s)
    return out

# ----------------------------------------------------------------------

def _dim(poly: Dict[DegreeVec, Fraction]) -> int:
    dims = {len(k) for k in poly.keys()}
    if len(dims) != 1:
        raise ValueError("Vetores de graus com dimensões inconsistentes no polinômio.")
    return next(iter(dims))

# -----------------------------------------------------------------------

#Calcula os valores esperados (média) de graus para cada tipo de edge

def _expected_degree_by_type(poly: Dict[DegreeVec, Fraction], T: int) -> List[Fraction]:
    E = [Fraction(0, 1) for _ in range(T)] #T são os tipos de edges: 1, 2, 3...
    for d, p in poly.items():
        for t in range(T):
            E[t] += p * d[t]
    #print(f"Valores esperados de graus para cada tipo de edge: {E}")
    return E

# -----------------------------------------------------------------------

# Funções auxiliares

def _lcm(a: int, b: int) -> int:
    return abs(a*b) // math.gcd(a, b)


def _lcm_list(nums: List[int]) -> int:
    out = 1
    for n in nums:
        out = _lcm(out, n)
    return out


def _term_denominators(poly: Dict[DegreeVec, Fraction]) -> List[int]:
    return [p.denominator for p in poly.values()]


def symbolic_met_compatibility_report(
    v_poly: PolyNode,
    p_poly: PolyNode,
    max_den: int = 10_000
) -> dict:
    """
    Verificação simbólica/estrutural de compatibilidade MET para polinômios node-perspective.

    Retorna um dicionário com:
      - T
      - E_var_by_type, E_chk_by_type
      - ratios_M_over_N_by_type
      - is_asymptotically_consistent
      - suggested_base_N_multiple (para contagens exatas dos termos)
    """
    v = _normalize(v_poly)
    p = _normalize(p_poly)
    #p = p_poly

    T_v = _dim(v)
    T_p = _dim(p)
    if T_v != T_p:
        raise ValueError(f"Dimensão diferente: v tem T={T_v}, p tem T={T_p}.")
    T = T_v

    # presença de tipos
    def types_present(poly):
        pres = [False]*T
        for d, prob in poly.items():
            if prob == 0:
                continue
            for t in range(T):
                if d[t] > 0:
                    pres[t] = True
        return pres

    pres_v = types_present(v)
    pres_p = types_present(p)

    missing = [t for t in range(T) if pres_v[t] != pres_p[t]]
    if missing:
        # não necessariamente impossível, mas normalmente sinaliza ensemble degenerado
        pass

    Ev = _expected_degree_by_type(v, T)
    Ec = _expected_degree_by_type(p, T)

    # razões M/N por tipo: Ev[t] / Ec[t] (quando Ec[t]>0)
    ratios = []
    for t in range(T):
        if Ec[t] == 0:
            ratios.append(None)
        else:
            ratios.append(Ev[t] / Ec[t])

    # consistência assintótica: todos ratios (não None) iguais
    defined = [r for r in ratios if r is not None]
    is_consistent = all(r == defined[0] for r in defined) if defined else False

    # N múltiplo para permitir contagens exatas por termo (quando desejável)
    base_N = _lcm_list(_term_denominators(v))
    base_M = _lcm_list(_term_denominators(p))

    return {
        "T": T,
        "E_var_by_type": Ev,
        "E_chk_by_type": Ec,
        "ratios_M_over_N_by_type": ratios,
        "is_asymptotically_consistent": is_consistent,
        "types_present_var": pres_v,
        "types_present_chk": pres_p,
        "suggested_base_N_multiple": base_N,
        "suggested_base_M_multiple": base_M,
    }


def suggest_NM_candidates(
    v_poly: PolyNode,
    p_poly: PolyNode,
    num_solutions: int = 3,
    N_min: int = 50,
    N_max: int = 5000,
    force_exact_term_counts: bool = True,
    target_rate: Optional[float] = None,
    max_den: int = 10_000
) -> List[Tuple[int, int]]:
    """
    Sugere pares (N, M) viáveis para construção:
      - usa o relatório simbólico para obter razão M/N assintótica
      - varre N e calcula M esperado por tipo, procurando M inteiro consistente
      - se target_rate for fornecida, força M = round(N*(1-target_rate)) e checa consistência

    Observação: isso não resolve restrições adicionais como decomposições (4a+9b etc).
    Essas restrições são "de suporte" (quais vetores existem no polinômio) e podem exigir
    checagem mais forte (que depende do suporte) — mas este método já pega a compatibilidade
    macro e as integridades básicas.
    """
    rep = symbolic_met_compatibility_report(v_poly, p_poly, max_den=max_den)
    if not rep["is_asymptotically_consistent"]:
        raise ValueError(
            "Ensemble MET inconsistente assintoticamente: as razões M/N por tipo não coincidem.\n"
            f"Ratios por tipo: {rep['ratios_M_over_N_by_type']}"
        )

    T = rep["T"]
    ratios = rep["ratios_M_over_N_by_type"]
    r = next(rt for rt in ratios if rt is not None)  # Fração

    baseN = rep["suggested_base_N_multiple"] if force_exact_term_counts else 1

    sols = []
    for N in range(max(N_min, baseN), N_max + 1):
        if N % baseN != 0:
            continue

        if target_rate is not None:
            M = int(round(N * (1.0 - target_rate)))
        else:
            # M ~ r*N
            M = int(r * N) if (r * N).denominator == 1 else None
            if M is None:
                continue

        # checa razão por tipo: Ec[t]*M == Ev[t]*N
        Ev = rep["E_var_by_type"]
        Ec = rep["E_chk_by_type"]

        ok = True
        for t in range(T):
            if Ec[t] == 0:
                # se Ec[t]==0, Ev[t] também precisa ser 0 para não haver aresta desse tipo
                if Ev[t] != 0:
                    ok = False
                    break
            else:
                if Ec[t] * M != Ev[t] * N:
                    ok = False
                    break

        if ok:
            sols.append((N, M))
            if len(sols) >= num_solutions:
                break

    if len(sols) < num_solutions:
        # retorna o que encontrou, mas avisa via exceção ou deixa parcial
        # aqui opto por retornar parcial sem quebrar
        return sols

    return sols

#Funções para fornecer dv_list e dc_list a serem aplicados no PEG-MET

def make_vectors_from_exact_counts(poly_node: PolyNode, N: int) -> List[DegreeVec]:
    """
    Converte um polinômio node-perspective em uma lista de N vetores de graus.
    Exige que N * p(d) seja inteiro para todos os termos (aqui fazemos por arredondamento controlado).
    """
    #poly = normalize_poly_node(poly_node)
    poly = _normalize(poly_node)
    degs = list(poly.keys())
    probs = np.array([poly[d] for d in degs], dtype=float)

    raw = probs * N
    counts = np.floor(raw).astype(int)
    rem = N - int(counts.sum())

    # distribui o restante pelos maiores resíduos
    residues = raw - counts
    order = np.argsort(-residues)
    for k in range(rem):
        counts[order[k]] += 1

    out = []
    for d, c in zip(degs, counts):
        out.extend([d] * int(c))
    if len(out) != N:
        raise RuntimeError("Falha ao construir lista de vetores com tamanho N.")
    random.shuffle(out)
    return out

def edge_counts_by_type(vectors: List[DegreeVec], T: int) -> List[int]:
    E = [0] * T
    for d in vectors:
        if len(d) != T:
            raise ValueError("Dimensão do vetor de graus != T.")
        for t in range(T):
            E[t] += int(d[t])
    return E

def degrees_by_type_from_vectors(vectors: List[DegreeVec], T: int) -> List[List[int]]:
    N = len(vectors)
    dv_by_type = [[0] * N for _ in range(T)]
    for j, d in enumerate(vectors):
        for t in range(T):
            dv_by_type[t][j] = int(d[t])
    return dv_by_type

def balance_check_vectors_to_match_edges(
    check_poly_node: PolyNode,
    M: int,
    target_edges_by_type: List[int],
    T: int,
    seed: Optional[int] = None,
    max_iter: int = 50000
) -> List[DegreeVec]:
    """
    Gera M vetores de graus para checks (node-perspective) e ajusta por "troca local"
    para casar EXATAMENTE os totais de sockets por tipo.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    poly = _normalize(check_poly_node)
    #poly = check_poly_node
    support = list(poly.keys())
    weights = [poly[d] for d in support]

    # amostra inicial multinomial
    probs = np.array(weights, dtype=float)
    counts = np.random.multinomial(M, probs / probs.sum())
    checks = []
    for d, c in zip(support, counts):
        checks.extend([d] * int(c))
    while len(checks) < M:
        checks.append(random.choices(support, weights=weights, k=1)[0])
    checks = checks[:M]
    random.shuffle(checks)

    cur_edges = edge_counts_by_type(checks, T)

    def err(e):
        return sum(abs(e[t] - target_edges_by_type[t]) for t in range(T))

    cur_err = err(cur_edges)
    if cur_err == 0:
        return checks

    for _ in range(max_iter):
        if cur_err == 0:
            break

        idx = random.randrange(M)
        old = checks[idx]

        # tenta algumas propostas e pega a melhor
        best_new = old
        best_edges = cur_edges
        best_err = cur_err

        for __ in range(8):
            cand = random.choices(support, weights=weights, k=1)[0]
            new_edges = cur_edges[:]  # cópia
            for t in range(T):
                new_edges[t] += int(cand[t]) - int(old[t])
            new_err = err(new_edges)
            if new_err < best_err:
                best_err = new_err
                best_new = cand
                best_edges = new_edges
                if best_err == 0:
                    break

        if best_new != old:
            checks[idx] = best_new
            cur_edges = best_edges
            cur_err = best_err

    if cur_err != 0:
        raise ValueError(
            "Não consegui casar os totais de sockets por tipo.\n"
            f"Target: {target_edges_by_type}\n"
            f"Atual : {cur_edges}\n"
            f"Diff  : {[cur_edges[t]-target_edges_by_type[t] for t in range(T)]}\n"
            "Sugestões: ajuste M, use N maior, ou revise p(x)."
        )

    return checks

# ============================================================
# 2) PEG tipado simultâneo (BFS no grafo unido)
# ============================================================

def build_ldpc_met_peg(
    N: int,
    M: int,
    dv_by_type: List[List[int]],   # [t][j]
    dc_by_type: List[List[int]],   # [t][i]
    seed: Optional[int] = None
):
    """
    Construção MET-LDPC com sockets tipados simultaneamente.

    Regras:
      - aresta do tipo t só conecta sockets tipo t
      - BFS usa H_union (união de todos os tipos já colocados) para evitar ciclos curtos mistos
      - PROÍBE múltiplos tipos conectarem o mesmo par (check,var):
            exige H_union[c, j] == 0 no momento de inserir a aresta
        (evita arestas paralelas entre os mesmos nós, mesmo que tipadas)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    T = len(dv_by_type)
    if any(len(dv_by_type[t]) != N for t in range(T)):
        raise ValueError("dv_by_type dimensão inconsistente.")
    if any(len(dc_by_type[t]) != M for t in range(T)):
        raise ValueError("dc_by_type dimensão inconsistente.")

    # consistência total de sockets por tipo
    for t in range(T):
        if sum(dv_by_type[t]) != sum(dc_by_type[t]):
            raise ValueError(
                f"Tipo {t}: sum(dv)={sum(dv_by_type[t])} != sum(dc)={sum(dc_by_type[t])}"
            )

    H_types = [np.zeros((M, N), dtype=np.uint8) for _ in range(T)]
    H_union  = np.zeros((M, N), dtype=np.uint8)

    row_deg_t = [np.zeros(M, dtype=int) for _ in range(T)]

    def bfs_visited_checks(start_var: int) -> set:
        visited_v = {start_var}
        visited_c = set()
        frontier_v = {start_var}

        while True:
            # vars -> checks
            next_c = set()
            for v in frontier_v:
                next_c |= set(np.where(H_union[:, v] == 1)[0])
            next_c -= visited_c
            if not next_c:
                break
            visited_c |= next_c

            # checks -> vars
            next_v = set()
            for c in next_c:
                next_v |= set(np.where(H_union[c, :] == 1)[0])
            next_v -= visited_v
            if not next_v:
                break
            visited_v |= next_v
            frontier_v = next_v

        return visited_c

    def socket_schedule_for_var(j: int) -> List[int]:
        # lista de tipos com multiplicidades = número de sockets por tipo
        sched = []
        for t in range(T):
            sched.extend([t] * int(dv_by_type[t][j]))
        # embaralha para intercalar tipos
        random.shuffle(sched)
        return sched

    for j in range(N):
        sched = socket_schedule_for_var(j)

        for t in sched:
            visited_c = bfs_visited_checks(j)

            # candidatos: capacidade no tipo t, fora dos alcançados, e sem aresta já existente no par (c,j)
            candidates = [
                c for c in range(M)
                if row_deg_t[t][c] < dc_by_type[t][c]
                and c not in visited_c
                and H_union[c, j] == 0
            ]

            if not candidates:
                # fallback: ignora BFS, mas mantém capacidade e evita aresta paralela
                candidates = [
                    c for c in range(M)
                    if row_deg_t[t][c] < dc_by_type[t][c]
                    and H_union[c, j] == 0
                ]

            if not candidates:
                raise ValueError(
                    f"Sem candidatos para conectar var {j} no tipo {t}. "
                    "Possível inconsistência ou falta de flexibilidade (N pequeno)."
                )

            best_c = min(candidates, key=lambda c: row_deg_t[t][c])

            H_types[t][best_c, j] = 1
            row_deg_t[t][best_c] += 1
            H_union[best_c, j] = 1

    return H_types, H_union

# ============================================================
# 3) QC lifting por tipo + união
# ============================================================

def circulant_identity(Z: np.uint8, shift: int) -> np.ndarray:
    I = np.eye(Z, dtype=np.uint8)
    return np.roll(I, shift, axis=1)


def qc_lift_types(
    H_types: List[np.ndarray],
    Z: np.uint8,
    seed: Optional[int] = None
):
    """
    Lifting QC de cada H^{(t)} protográfica.
    Cada 1 vira uma circulante I^{(shift)} com shift aleatório.

    Retorna:
      Hq_types: lista [t] de matrizes (MZ x NZ)
      Hq_union: união (OR) de todas (MZ x NZ)
      shifts:   dict com shifts[(t, i, j)] = shift escolhido
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    T = len(H_types)
    M, N = H_types[0].shape
    for t in range(T):
        if H_types[t].shape != (M, N):
            raise ValueError("Todas as H_types devem ter mesma dimensão (M x N).")

    Hq_types = [np.zeros((M * Z, N * Z), dtype=np.uint8) for _ in range(T)]
    Hq_union = np.zeros((M * Z, N * Z), dtype=np.uint8)
    shifts = {}

    # opcional: evitar colisões entre tipos no mesmo bloco (i,j) (não deveria ocorrer se você proibiu no PEG)
    for i in range(M):
        for j in range(N):
            occ = sum(int(H_types[t][i, j]) for t in range(T))
            if occ > 1:
                raise ValueError(f"Encontrado par (check {i}, var {j}) com múltiplos tipos no protógrafo.")

    for t in range(T):
        Ht = H_types[t]
        for i in range(M):
            for j in range(N):
                if Ht[i, j] == 1:
                    s = np.random.randint(0, Z)
                    shifts[(t, i, j)] = s
                    block = circulant_identity(Z, s)
                    Hq_types[t][i*Z:(i+1)*Z, j*Z:(j+1)*Z] = block
                    Hq_union[i*Z:(i+1)*Z, j*Z:(j+1)*Z] |= block

    return Hq_types, Hq_union, shifts



# ---------------------------------------------------------------
# ---------------------- EXECUÇÃO -------------------------------
# ---------------------------------------------------------------

d_edge = 3 #tipos de arestas presentes
#temos 3 tipos de nós de variável também, com a seguinte distribuição:
#cada coluna representa um tipo de aresta
#Mani et al. (ArXiv) ---------------------------------------------------------
poly_var_node = {
    (2, 51, 0): 0.02,
    (3, 60, 0): 0.02,
    (0,  0, 1): 0.96,
}

# p(x)=0.016 x1^4 + 0.004 x1^9 + 0.3 x2^3 x3^1 + 0.66 x2^2 x3^1
poly_chk_node = {
    (4, 0, 0): 0.016,
    (9, 0, 0): 0.004,
    (0, 3, 1): 0.300,
    (0, 2, 1): 0.660,
}

#Mani et al. (Pshysical Review A.)
"""poly_var_node = {
    (2, 52, 0): 0.0225,
    (3, 57, 0): 0.0175,
    (0,  0, 1): 0.96,
}

poly_chk_node = {
    (4, 0, 0): 0.0165,
    (9, 0, 0): 0.0035,
    (0, 3, 1): 0.2475,
    (0, 2, 1): 0.7125,
}"""


#Milicevic et al. Deu inconsistência na hora de fornecer M e N ------------
"""poly_var_node = {
    (2, 57, 0): 9/400,
    (3, 57, 0): 7/400,
    (0,  0, 1): 24/25,
}

poly_chk_node = {
    (3, 0, 0): 3/320,
    (7, 0, 0): 17/1600,
    (0, 2, 1): 3/5,
    (0, 3, 1): 9/25,
}"""


rep = symbolic_met_compatibility_report(poly_var_node, poly_chk_node)
print(rep)

pairs = suggest_NM_candidates(poly_var_node, poly_chk_node, num_solutions=15, N_min=50, N_max=5000, target_rate=0.02)
print(pairs)


seed = 20260220

random.seed(seed)
np.random.seed(seed)

# ------------ Escolha de N compatível e taxa ~ 0.02 ------------
# Com N=50, temos 0.02N = 1 nó para cada termo de 2%, e 48 nós para 96%.
# E com M=49, dá R = 1 - M/N = 0.02 (Slepian-Wolf via síndrome).
T = 3
N = 2500 #7500
M = 2450 #7350

#N = 20000
#M = 19600
#Z = 8  # fator QC (pode aumentar para melhor desempenho/ciclos maiores)

# ------------ Gerar vetores de graus das variáveis ------------
var_vectors = make_vectors_from_exact_counts(poly_var_node, N)
target_E = edge_counts_by_type(var_vectors, T)

#print(var_vectors)
print("Totais de sockets por tipo (lado variável):", target_E)


# ------------ Gerar vetores de graus dos checks casando sockets por tipo ------------
chk_vectors = balance_check_vectors_to_match_edges(
    check_poly_node=poly_chk_node,
    M=M,
    target_edges_by_type=target_E,
    T=T,
    seed=seed,
    max_iter=120000
)
print("Totais de sockets por tipo (lado check)   :", edge_counts_by_type(chk_vectors, T))

# ------------ dv/dc por tipo (automático) ------------
dv_by_type = degrees_by_type_from_vectors(var_vectors, T)  # [t][j]
dc_by_type = degrees_by_type_from_vectors(chk_vectors, T)  # [t][i]

#print(f"dv_by_type: {dv_by_type[2][0:5]}")
#print(f"dc_by_type: {dc_by_type[1][0:5]}")

# ------------ Construção MET-PEG protográfica ------------
H_types, H_union = build_ldpc_met_peg(
    N=N, M=M, dv_by_type=dv_by_type, dc_by_type=dc_by_type, seed=seed
)

for mat in range(0,T):
  print(f"H type {mat+1}:\n{H_types[mat][0:3][0:3]}")

print(f"H_union:\n{H_union[0:5][0:5]}")
# Checagens rápidas de consistência
for t in range(T):
    total_edges_t = int(H_types[t].sum())
    print(f"Protógrafo: tipo {t} total de arestas = {total_edges_t}") 

#Hproto = np.savetxt("H_union4.txt", H_union, fmt='%d')


# ------------ Lifting QC por tipo ------------
Z=3 #21, 50, 80
Hq_types, Hq_union, shifts = qc_lift_types(H_types, Z=Z, seed=seed)

print("H_union protográfica shape:", H_union.shape)
print("H_union QC shape          :", Hq_union.shape)
print("Densidade H_union QC      :", float(Hq_union.mean()))

# Se quiser inspecionar:
# print("H_union protográfica:\n", H_union)
# print("H_union QC:\n", Hq_union)

#np.savetxt("Hq_union_60000_R002.txt", Hq_union, fmt='%d') #Da mais de 1GB de arquivo, não vale a pena salvar


from scipy.sparse import csc_matrix, csr_matrix, coo_matrix

#np.savetxt("H_proto_Mani_25000.txt", H_union, fmt='%d')

H = csr_matrix(Hq_union)
#H = csr_matrix(H_union)
if type(H) == csr_matrix:
    H = csr_matrix.todense(H).astype(np.uint8)
elif type(H) == csc_matrix:
    H = csc_matrix.todense(H).astype(np.uint8)
elif type(H) == coo_matrix:
    H = coo_matrix.todense(H).astype(np.uint8)
else:
    H = H.astype(np.int8)

m, n = H.shape
print(H)
varDegrees = [int(np.sum(H[:, j])) for j in range(n)]
checkDegrees = [int(np.sum(H[i, :])) for i in range(m)]
maxColDeg = max(varDegrees)
maxRowDeg = max(checkDegrees)

with open("LDPC_DVBS2_52500b_R002.txt", "w") as f:
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


import logging as logg
import matplotlib.pyplot as plt
from numba.typed import List
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix

def plotBinaryMatrix(H):
    """
    Plot the binary matrix H with dots at positions where H[i,j] = 1.

    Parameters
    ----------
    H : ndarray of shape (m, n)
        Binary matrix.
    """
    H = np.asarray(H)
    rows, cols = np.where(H == 1)
    plt.scatter(cols, rows, s=0.05, color="blue")  # s controls dot size
    plt.gca().invert_yaxis()
    plt.xlabel("Column indexes")
    plt.ylabel("Row indexes")
    plt.title(f"Matrix: {H.shape[0]} $\\times$ {H.shape[1]}")
    plt.axis("square")
    plt.xlim(0, H.shape[1])
    plt.ylim(H.shape[0], 0)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

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



H = readAlist("LDPC_DVBS2_52500b_R002.txt")
plotBinaryMatrix(csr_matrix.todense(H))