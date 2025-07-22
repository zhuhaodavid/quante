# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-07-23 00:05:42
# @Last Modified by:   hzhu
# @Last Modified time: 2025-07-23 01:37:08

from quspin.operators._make_hamiltonian import _consolidate_static
import numpy as np

def clean_static(static):
    """Clean the static list by removing duplicate terms and combining coefficients."""
    reduce_dic = {
        ('+', '+'): (),
        ('+', '-'): (('I', 0.5), ('z', 1.0)),
        ('+', 'z'): (('+', -0.5), ),
        ('+', 'I'): (('+', 1.), ),
        ('-', '-'): (),
        ('-', '+'): (('I', 0.5), ('z', -1.0)),
        ('-', 'z'): (('-', 0.5), ),
        ('-', 'I'): (('-', 1.), ),
        ('z', '+'): (('+', 0.5), ),
        ('z', '-'): (('-', -0.5), ),
        ('z', 'z'): (('I', 0.25), ),
        ('z', 'I'): (('z', 1.), ),
        ('I', '+'): (('+', 1.), ),
        ('I', '-'): (('-', 1.), ),
        ('I', 'z'): (('z', 1.), ),
        ('I', 'I'): (('I', 1.), ),
    }
    # first expand with pmz
    from quspin.basis import spin_basis_general
    basis = spin_basis_general(N=1, S='1/2', pauli=0)  # dummy call to load the module
    tmp, _ = basis.expanded_form(static)

    static_dict = {}
    for opnm, posn, coef in _consolidate_static(tmp):
        # remove first 'I' if it exists
        if opnm.startswith('I'):
            for i in range(len(opnm)):
                if opnm[i] != 'I':
                    opnm = opnm[i:]
                    posn = posn[i:]
                    break
        
        if len(opnm) == 0:
            return []
        
        res = [[[opnm[0]], [posn[0]], coef], ]

        for cur in range(len(posn)-1):
            o2, p2 = opnm[cur+1], posn[cur+1]
            if o2 == 'I':
                continue # 'I' is identity, skip it

            should_remove = []
            l = len(res)
            for i in range(l):
                opnm_i, posn_i, coef_i = res[i][0], res[i][1], res[i][2]
                if len(opnm_i) == 0:
                    res[i][0] = [o2]
                    res[i][1] = [p2]
                    continue
            
                o1, p1 = opnm_i[-1], posn_i[-1]
                if p1 == p2:
                    no = reduce_dic[(o1, o2)]
                    if len(no) == 0:
                        should_remove.append(i)
                    elif len(no) == 1:
                        if no[0][0] == 'I':
                            res[i][0] = opnm_i[:-1]
                            res[i][1] = posn_i[:-1]
                            res[i][2] *= no[0][1]
                        else:
                            res[i][0] = opnm_i[:-1] + [no[0][0]]
                            res[i][2] *= no[0][1]
                    else:
                        if no[0][0] == 'I':
                            res[i][0] = opnm_i[:-1]
                            res[i][1] = posn_i[:-1]
                            res[i][2] *= no[0][1]
                        else:
                            res[i][0] = opnm_i[:-1] + [no[0][0]]
                            res[i][2] *= no[0][1]
                        for j in range(1, len(no)):
                            if no[j][0] == 'I':
                                newopnm = opnm_i[:-1]
                                newposn = posn_i[:-1]
                            else:
                                newopnm = opnm_i[:-1] + [no[j][0]]
                                newposn = posn_i
                            newcoef = coef_i * no[j][1]
                            if len(newopnm) > 0:
                                res.append([newopnm, newposn, newcoef])
                else:
                    opnm_i.append(o2)
                    posn_i.append(p2)
        
            for i in reversed(should_remove):
                res.pop(i)
            
        for opnm_, posn_, J in res:
            if len(opnm_) == 0:
                opstr = 'I'
                indx = [0]
            else:
                inc_indx = [i for i, c in enumerate(opnm_) if c == 'I']
                opstr = ''.join(c for i, c in enumerate(opnm_) if i not in inc_indx)
                indx = [c for i, c in enumerate(posn_) if i not in inc_indx]

            indx.insert(0, J)
            if opstr in static_dict:
                static_dict[opstr].append(indx)
            else:
                static_dict[opstr] = [indx]
            
    res = [[str(key), list(value)] for key, value in static_dict.items()]
    res, _ = basis.expanded_form(res)
    return res


def clean_static2(static):
    """Clean the static list by removing duplicate terms and combining coefficients.
    
    This apply only of spin-1/2 operators!!! 
    """
    reduce_dic = {
        ('x', 'x'): ('I', 0.25),
        ('x', 'y'): ('z', 0.5j),
        ('x', 'z'): ('y', -0.5j),
        ('x', 'I'): ('x', 1.0),
        ('y', 'x'): ('z', -0.5j),
        ('y', 'y'): ('I', 0.25),
        ('y', 'z'): ('x', 0.5j),
        ('y', 'I'): ('y', 1.0),
        ('z', 'x'): ('y', 0.5j),
        ('z', 'y'): ('x', -0.5j),
        ('z', 'z'): ('I', 0.25),
        ('z', 'I'): ('z', 1.0),
        ('I', 'x'): ('x', 1.0),
        ('I', 'y'): ('y', 1.0),
        ('I', 'z'): ('z', 1.0),
        ('I', 'I'): ('I', 1.0),
    }
    static_dict = {}
    for opstr, indx, J in _consolidate_static(static):
        sortindx = np.argsort(indx, kind='stable')
        opstr = [opstr[i] for i in sortindx]
        indx = [indx[i] for i in sortindx]
        newopstr = [opstr[0]]
        newindx = [indx[0]]
        newcoef = J
        for i in range(len(opstr) - 1):
            o1, p1 = opstr[i], indx[i]
            o2, p2 = opstr[i+1], indx[i+1]
            if p1 == p2:
                no, cf = reduce_dic.get((o1, o2), (None, None))
                if no is None and cf is None:
                    raise ValueError(f"Unsupported operator combination: {o1}, {o2}")
                newopstr[-1] = no
                newcoef *= cf
            else:
                newopstr.append(o2)
                newindx.append(p2)
        notIindx = [i for i, c in enumerate(newopstr) if c != 'I']
        newopstr = ''.join(c for i, c in enumerate(newopstr) if i in notIindx)
        newindx = [c for i, c in enumerate(newindx) if i in notIindx]
        newindx.insert(0, newcoef)
        if len(newopstr) == 0:
            newopstr = 'I'
            newindx.append(0)
        if newopstr in static_dict:
            for j in static_dict[newopstr]:
                if all(a==b for a,b in zip(j[1:], newindx[1:])):
                    # If the new index is already in the list, skip adding it
                    j[0] += newcoef
                    break
            else:
                static_dict[newopstr].append(newindx)
        else:
            static_dict[newopstr] = [newindx]
    return [[str(key), list(value)] for key, value in static_dict.items()]


if __name__ == "__main__":
    # Example usage
    static = [["xyzzxy", [[1., 1, 2, 3, 4, 2, 3]]], ["xyzzxy", [[1., 1, 2, 3, 4, 2, 3]]]]
    print(clean_static2(static))