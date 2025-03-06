from __future__ import print_function
from itertools import combinations
from collections import defaultdict
from pathlib import Path
import subprocess
import math
import sys
import pandas as pd

def apriori(dataset, min_support=0.5, verbose=False):
    """
    The Apriori algorithm will iteratively generate new candidate
    k-itemsets using the frequent (k-1)-itemsets found in the previous
    iteration.

    Parameters
    ----------
    dataset : list
        The dataset (a list of transactions) from which to generate
        candidate itemsets.

    min_support : float
        The minimum support threshold. Defaults to 0.5.

    Returns
    -------
    F : list
        The list of frequent itemsets.

    support_data : dict
        The support data for all candidate itemsets.
    """
    C1 = create_candidates(dataset)
    D = list(map(set, dataset))
    F1, support_data = get_freq(D, C1, min_support, verbose=False)  # get frequent 1-itemsets
    F = [F1]                                                        # list of frequent itemsets; initialized to frequent 1-itemsets
    k = 2                                                           # the itemset cardinality
    while (len(F[k - 2]) > 0):
        Ck = apriori_gen(F[k-2], k)                                 # generate candidate itemsets
        Fk, supK  = get_freq(D, Ck, min_support)                    # get frequent itemsets
        support_data.update(supK)                                   # update the support counts to reflect pruning
        F.append(Fk)                                                # add the frequent k-itemsets to the list of frequent itemsets
        k += 1

    if verbose:                                                     # Print a list of all the frequent itemsets.
        for kset in F:
            for item in kset:
                print(""                     + "{"                     + "".join(str(i) + ", " for i in iter(item)).rstrip(', ')                     + "}"                     + ":  sup = " + str(round(support_data[item], 3)))

    return F, support_data

def create_candidates(dataset, verbose=False):
    """
    Creates a list of candidate 1-itemsets from a list of transactions.
    Parameters
    ----------
    dataset : list
        The dataset (a list of transactions) from which to generate candidate
        itemsets.

    Returns
    -------
    The list of candidate itemsets (c1) passed as a frozenset (a set that is
    immutable and hashable).
    """
    c1 = set()
    for transaction in dataset:
        for item in transaction:
            c1.add(frozenset([item]))  # Avoid redundant checks
    return list(c1)

def get_freq(dataset, candidates, min_support, verbose=False):
    """
    This function separates the candidates itemsets into frequent itemset and infrequent itemsets based on the min_support,
	and returns all candidate itemsets that meet a minimum support threshold.

    Parameters
    ----------
    dataset : list
        The dataset (a list of transactions) from which to generate candidate
        itemsets.

    candidates : frozenset
        The list of candidate itemsets.

    min_support : float
        The minimum support threshold.

        Returns
    -------
    freq_list : list
        The list of frequent itemsets.

    support_data : dict
        The support data for all candidate itemsets.
    """

    freq_list = []
    support_data = defaultdict(int)

    # Count occurrences using a dictionary (faster lookup)
    for transaction in dataset:
        for itemset in candidates:
            if itemset.issubset(transaction):  # Faster than intersection check
                support_data[itemset] += 1

    # Filter candidates that meet min_support
    num_transactions = len(dataset)
    min_count = math.ceil(min_support * num_transactions)

    for itemset, count in support_data.items():
        if count >= min_count:
            freq_list.append(itemset)

    return freq_list, support_data

def apriori_gen(freq_sets, k):
    """
    Generates candidate itemsets (via the F_k-1 x F_k-1 method).

    This part generates new candidate k-itemsets based on the frequent
    (k-1)-itemsets found in the previous iteration.

    The apriori_gen function performs two operations:
    (1) Generate length k candidate itemsets from length k-1 frequent itemsets
    (2) Prune candidate itemsets containing subsets of length k-1 that are infrequent

    Parameters
    ----------
    freq_sets : list
        The list of frequent (k-1)-itemsets.

    k : integer
        The cardinality of the current itemsets being evaluated.

    Returns
    -------
    candidate_list : list
        The list of candidate itemsets.
    """
    
    candidate_list = []
    freq_set_list = list(freq_sets)  # Convert set to list for indexing

    for i in range(len(freq_set_list)):
        for j in range(i + 1, len(freq_set_list)):  
            L1 = list(freq_set_list[i])
            L2 = list(freq_set_list[j])
            L1.sort()
            L2.sort()
            
            # Generate only if first k-2 items are the same
            if L1[:k-2] == L2[:k-2]:
                new_candidate = freq_set_list[i] | freq_set_list[j]
                candidate_list.append(new_candidate)

    return candidate_list

def loadDataSet(fileName, delim=','):
    with open(fileName) as fr:
        stringArr = [list(filter(None, [item.strip() for item in line.strip().split(delim)])) for line in fr]
    return stringArr

def run_apriori(data_path, min_support, verbose=False):
    dataset = loadDataSet(data_path)
    F, support = apriori(dataset, min_support=min_support, verbose=verbose)
    return F, support

def bool_transfer(input):
    ''' Transfer the input to boolean type'''
    input = str(input)
    if input.lower() in ['t', '1', 'true' ]:
        return True
    elif input.lower() in ['f', '0', 'false']:
        return False
    else:
        raise ValueError('Input must be one of {T, t, 1, True, true, F, F, 0, False, false}')

if __name__ == '__main__':
    if len(sys.argv)==3:
        F, support = run_apriori(sys.argv[1], float(sys.argv[2]))
    elif len(sys.argv)==4:
        F, support = run_apriori(sys.argv[1], float(sys.argv[2]), bool_transfer(sys.argv[3]))
    else:
        raise ValueError('Usage: python apriori.py <data_path> <min_support> <is_verbose>')
    # print([tuple(s) for s in F])
    support = dict(zip([tuple(k) for k in support.keys()],[v for v in support.values()]))

    if len(sys.argv)==4:
        if bool_transfer(sys.argv[3]) == False:
            filename = Path(sys.argv[1]).stem + "_out"
            root_path = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).strip().decode())
            output_dir = root_path / "data" / "freq"  # Target folder
            output_dir.mkdir(parents=True, exist_ok=True)  # Ensure it exists
            output_file = output_dir / filename
            with open(output_file, mode = 'w',newline='') as file:
                for k,v in support.items():
                    if(len(k) > 1):
                        file.write(','.join(map(str,list(k))))
                        file.write(',support: ' + str(v))
                        file.write('\n')
            print(f"Saved frequent itemsets to {output_file}")



    '''
    Example: 
    py apriori.py data.txt 0.5 
    OR
    py apriori.py data.txt 0.5 True

    Run from the home directory
    '''