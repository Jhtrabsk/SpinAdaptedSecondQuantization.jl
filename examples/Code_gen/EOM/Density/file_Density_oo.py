E_ji[:,:] = E_ji[:,:] .+  -1.00000000  * fixed_einsum("Aaj,Aai->ji", extract_mat(p, "VIvo", o, v), extract_mat(s, "VIvo", o, v), optimize="optimal");
E_ji[:,:] = E_ji[:,:] .+  -2.00000000  * fixed_einsum("akbj,akbi->ji", extract_mat(c2, "vovo", o, v), extract_mat(t, "vovo", o, v), optimize="optimal");
E_ji[:,:] = E_ji[:,:] .+  -1.00000000  * fixed_einsum("Aakbj,Aakbi->ji", extract_mat(p2, "VIvovo", o, v), extract_mat(s2, "VIvovo", o, v), optimize="optimal");
E_ji[:,:] = E_ji[:,:] .+  -1.00000000  * fixed_einsum("Aajbk,Aaibk->ji", extract_mat(p2, "VIvovo", o, v), extract_mat(s2, "VIvovo", o, v), optimize="optimal");
