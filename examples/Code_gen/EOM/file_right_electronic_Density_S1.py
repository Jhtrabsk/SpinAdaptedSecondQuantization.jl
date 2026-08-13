E_ai[:,:] +=  +2.00000000  * extract_mat(r1, "vo", o, v);
E_ji[:,:] = E_ji[:,:] .+  -1.00000000  * fixed_einsum("Aaj,Aai->ji", extract_mat(L2, "VIvo", o, v), extract_mat(R2, "VIvo", o, v), optimize="optimal"); # Added
E_ji[:,:] = E_ji[:,:] .+  -1.00000000  * fixed_einsum("Aaj,Aai->ji", extract_mat(L2, "VIvo", o, v), extract_mat(s, "VIvo", o, v), optimize="optimal"); # we have 
E_ba[:,:] = E_ba[:,:] .+  +1.00000000  * fixed_einsum("Aai,Abi->ba", extract_mat(L2, "VIvo", o, v), extract_mat(R2, "VIvo", o, v), optimize="optimal"); # Added
E_ba[:,:] = E_ba[:,:] .+  +1.00000000  * fixed_einsum("Aai,Abi->ba", extract_mat(L2, "VIvo", o, v), extract_mat(s, "VIvo", o, v), optimize="optimal"); # we have
E_ia[:,:] = E_ai[:,:] .+  +1.00000000  * fixed_einsum("Aai,A->ia", extract_mat(L2, "VIvo", o, v), extract_mat(R1, "VI", o, v), optimize="optimal"); # we have 
E_ia[:,:] = E_ia[:,:] .+  +2.00000000  * fixed_einsum("Abj,Abjai->ia", extract_mat(L2, "VIvo", o, v), extract_mat(R3, "VIvovo", o, v), optimize="optimal"); # Added 
E_ia[:,:] = E_ia[:,:] .+  -1.00000000  * fixed_einsum("Abj,Abiaj->ia", extract_mat(L2, "VIvo", o, v), extract_mat(R3, "VIvovo", o, v), optimize="optimal"); # Added
E_ia[:,:] = E_ia[:,:] .+  -1.00000000  * fixed_einsum("Abj,Aajbi->ia", extract_mat(L2, "VIvo", o, v), extract_mat(R3, "VIvovo", o, v), optimize="optimal"); # Added
E_ia[:,:] = E_ia[:,:] .+  +2.00000000  * fixed_einsum("Abj,Aaibj->ia", extract_mat(L2, "VIvo", o, v), extract_mat(R3, "VIvovo", o, v), optimize="optimal"); # Added
E_ia[:,:] = E_ia[:,:] .+  +2.00000000  * fixed_einsum("Abj,Abjai->ia", extract_mat(L2, "VIvo", o, v), extract_mat(s2, "VIvovo", o, v), optimize="optimal"); # we have 
E_ia[:,:] = E_ia[:,:] .+  -1.00000000  * fixed_einsum("Abj,Abiaj->ia", extract_mat(L2, "VIvo", o, v), extract_mat(s2, "VIvovo", o, v), optimize="optimal"); # we have
E_ia[:,:] = E_ia[:,:] .+  +1.00000000  * fixed_einsum("Abj,A,bjai->ia", extract_mat(L2, "VIvo", o, v), extract_mat(R1, "VI", o, v), extract_mat(u, "vovo", o, v), optimize="optimal"); # implemented 
E_ia[:,:] = E_ia[:,:] .+  +2.00000000  * fixed_einsum("Abj,bj,Aai->ia", extract_mat(L2, "VIvo", o, v), extract_mat(r1, "vo", o, v), extract_mat(s, "VIvo", o, v), optimize="optimal"); # implemented
E_ia[:,:] = E_ia[:,:] .+  -1.00000000  * fixed_einsum("Abj,bi,Aaj->ia", extract_mat(L2, "VIvo", o, v), extract_mat(r1, "vo", o, v), extract_mat(s, "VIvo", o, v), optimize="optimal"); # implemented 
E_ia[:,:] = E_ia[:,:] .+  -1.00000000  * fixed_einsum("Abj,aj,Abi->ia", extract_mat(L2, "VIvo", o, v), extract_mat(r1, "vo", o, v), extract_mat(s, "VIvo", o, v), optimize="optimal"); # implemented