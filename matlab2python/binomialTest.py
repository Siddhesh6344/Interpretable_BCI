import numpy as np
from scipy.stats import binom

def binomial_test(s, n, p=0.5, sided='two'):
    """
    Performs a binomial test of the number of successes given a total number 
    of outcomes and a probability of success. Can be one or two-sided.
    
    Parameters:
    -----------
    s : array_like
        The observed number of successful outcomes
    n : array_like 
        The total number of outcomes (successful or not)
    p : array_like, optional
        The proposed probability of a successful outcome (default: 0.5)
    sided : str, optional
        Can be 'one', 'two' (default), or 'two, equal counts'
        
    Returns:
    --------
    pout : ndarray
        The probability of observing the resulting value of s or
        another value more extreme given n total outcomes with 
        a probability of success of p.
    """
    
    # Convert inputs to numpy arrays and ensure proper types
    s = np.floor(np.asarray(s)).astype(int)
    n = np.asarray(n)
    p = np.asarray(p)
    
    # Make arrays compatible (broadcast to same shape)
    s, n, p = np.broadcast_arrays(s, n, p)
    
    # Calculate expected values
    E = p * n
    
    # Determine which values are greater than expected
    greater_inds = s >= E
    pout = np.zeros_like(s, dtype=float)
    
    # Precision for rounding error tolerance
    prec = 1e-14
    
    sided_lower = sided.lower()
    
    if sided_lower in ['two', 'two, equal counts']:
        # Handle special case where p=0.5 and sided='two'
        if np.all(p == 0.5) and sided_lower == 'two':
            sided_lower = 'two, equal counts'
            
        dE = np.zeros_like(pout)
        
        # Calculate pout for greater indices first
        if np.any(greater_inds):
            # Probability of getting >= s successes
            pout[greater_inds] = 1 - binom.cdf(s[greater_inds] - 1, n[greater_inds], p[greater_inds])
            
            # Calculate difference from expected value
            dE[greater_inds] = s[greater_inds] - E[greater_inds]
            
            if sided_lower == 'two, equal counts':
                s2 = np.floor(E[greater_inds] - dE[greater_inds]).astype(int)
                
                # Only add if s2 >= 0 (negative successes impossible)
                valid_s2 = s2 >= 0
                if np.any(valid_s2):
                    mask = greater_inds.copy()
                    mask[greater_inds] = valid_s2
                    pout[mask] += binom.cdf(s2[valid_s2], n[mask], p[mask])
                
                # Adjust for cases where expected value is exactly equaled
                eq_inds = greater_inds & (dE == 0)
                if np.any(eq_inds):
                    pout[eq_inds] -= binom.pmf(E[eq_inds].astype(int), n[eq_inds], p[eq_inds])
                    
            else:  # sided_lower == 'two'
                greater_indices = np.where(greater_inds)[0]
                
                # Find values on other side with probability <= observed probability
                targy = binom.pmf(s[greater_inds], n[greater_inds], p[greater_inds])
                
                s2 = np.maximum(np.floor(E[greater_inds] - dE[greater_inds]).astype(int), 0)
                y = binom.pmf(s2, n[greater_inds], p[greater_inds])
                
                for i, idx in enumerate(greater_indices):
                    skip_p_add = False
                    
                    if y[i] <= targy[i]:
                        # Search forward until we find correct limit
                        while y[i] <= targy[i] and s2[i] < E[idx]:
                            s2[i] += 1
                            y[i] = binom.pmf(s2[i], n[idx], p[idx])
                        s2[i] -= 1  # Step back to last valid value
                    else:
                        # Search backward with precision check
                        while (y[i] - targy[i]) > prec and s2[i] < n[idx]:
                            s2[i] -= 1
                            y[i] = binom.pmf(s2[i], n[idx], p[idx])
                        
                        if (y[i] - targy[i]) > prec:
                            skip_p_add = True
                    
                    if not skip_p_add:
                        pout[idx] += binom.cdf(s2[i], n[idx], p[idx])
        
        # Calculate pout for lesser indices
        lesser_inds = ~greater_inds
        if np.any(lesser_inds):
            # Probability of getting <= s successes
            pout[lesser_inds] = binom.cdf(s[lesser_inds], n[lesser_inds], p[lesser_inds])
            
            # Calculate difference from expected value
            dE[lesser_inds] = E[lesser_inds] - s[lesser_inds]
            
            if sided_lower == 'two, equal counts':
                s2 = np.ceil(E[lesser_inds] + dE[lesser_inds]).astype(int)
                
                # Only add if s2 <= n
                valid_s2 = s2 <= n[lesser_inds]
                if np.any(valid_s2):
                    mask = lesser_inds.copy()
                    mask[lesser_inds] = valid_s2
                    pout[mask] += 1 - binom.cdf(s2[valid_s2] - 1, n[mask], p[mask])
                    
            else:  # sided_lower == 'two'
                lesser_indices = np.where(lesser_inds)[0]
                
                # Find values on other side with probability <= observed probability
                targy = binom.pmf(s[lesser_inds], n[lesser_inds], p[lesser_inds])
                
                s2 = np.minimum(np.ceil(E[lesser_inds] + dE[lesser_inds]).astype(int), n[lesser_inds])
                y = binom.pmf(s2, n[lesser_inds], p[lesser_inds])
                
                for i, idx in enumerate(lesser_indices):
                    skip_p_add = False
                    
                    if y[i] <= targy[i]:
                        # Search backward until we find correct limit
                        while y[i] <= targy[i] and s2[i] > E[idx]:
                            s2[i] -= 1
                            y[i] = binom.pmf(s2[i], n[idx], p[idx])
                        s2[i] += 1  # Step forward to last valid value
                    else:
                        # Search forward with precision check
                        while (y[i] - targy[i]) > prec and s2[i] < n[idx]:
                            s2[i] += 1
                            y[i] = binom.pmf(s2[i], n[idx], p[idx])
                        
                        if (y[i] - targy[i]) > prec:
                            skip_p_add = True
                    
                    if not skip_p_add:
                        pout[idx] += 1 - binom.cdf(s2[i] - 1, n[idx], p[idx])
    
    elif sided_lower == 'one':  # One-sided test
        if np.any(greater_inds):
            # Probability of getting >= s successes
            pout[greater_inds] = 1 - binom.cdf(s[greater_inds] - 1, n[greater_inds], p[greater_inds])
        
        if np.any(~greater_inds):
            # Probability of getting <= s successes
            pout[~greater_inds] = binom.cdf(s[~greater_inds], n[~greater_inds], p[~greater_inds])
    
    else:
        raise ValueError(f"Unknown sided value: {sided}. Must be 'one', 'two', or 'two, equal counts'.")
    
    return pout