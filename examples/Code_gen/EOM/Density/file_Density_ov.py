E_ai[:,:] = E_ai[:,:] .+  +2.00000000  * fixed_einsum("A,Aai->ai", extract_mat(p3, "VI", o, v), extract_mat(s, "VIvo", o, v), optimize="optimal");
E_ai[:,:] = E_ai[:,:] .+  +1.00000000  * fixed_einsum("bj,bjai->ai", extract_mat(c1, "vo", o, v), extract_mat(u, "vovo", o, v), optimize="optimal");
E_ai[:,:] = E_ai[:,:] .+  +2.00000000  * fixed_einsum("Abj,Abjai->ai", extract_mat(p, "VIvo", o, v), extract_mat(s2, "VIvovo", o, v), optimize="optimal");
E_ai[:,:] = E_ai[:,:] .+  -1.00000000  * fixed_einsum("Abj,Abiaj->ai", extract_mat(p, "VIvo", o, v), extract_mat(s2, "VIvovo", o, v), optimize="optimal");
E_ai[:,:] = E_ai[:,:] .+  -1.00000000  * fixed_einsum("Abjck,Abi,ajck->ai", extract_mat(p2, "VIvovo", o, v), extract_mat(s, "VIvo", o, v), extract_mat(t, "vovo", o, v), optimize="optimal");
E_ai[:,:] = E_ai[:,:] .+  -1.00000000  * fixed_einsum("Abjck,Aaj,bick->ai", extract_mat(p2, "VIvovo", o, v), extract_mat(s, "VIvo", o, v), extract_mat(t, "vovo", o, v), optimize="optimal");
E_ai[:,:] = E_ai[:,:] .+  -1.00000000  * fixed_einsum("Abjck,Aak,bjci->ai", extract_mat(p2, "VIvovo", o, v), extract_mat(s, "VIvo", o, v), extract_mat(t, "vovo", o, v), optimize="optimal");
E_ai[:,:] = E_ai[:,:] .+  -1.00000000  * fixed_einsum("Abjck,Aci,bjak->ai", extract_mat(p2, "VIvovo", o, v), extract_mat(s, "VIvo", o, v), extract_mat(t, "vovo", o, v), optimize="optimal");