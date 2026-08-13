E_ai[:,:] +=  +2.00000000  * extract_mat(r1, "vo", o, v); # we have
E_ai[:,:] = E_ai[:,:] .+  +2.00000000  * fixed_einsum("A,Aai->ai", extract_mat(L1, "VI", o, v), extract_mat(R2, "VIvo", o, v), optimize="optimal"); # added
E_ai[:,:] = E_ai[:,:] .+  +2.00000000  * fixed_einsum("A,Aai->ai", extract_mat(L1, "VI", o, v), extract_mat(s, "VIvo", o, v), optimize="optimal"); # we have
