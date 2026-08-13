E_CB[:,:] = E_CB[:,:] .+  +1.00000000  * fixed_einsum("aibj,ai,Cbj->CB", extract_mat(L3, "AIvovo", o, v), extract_mat(r1, "vo", o, v), extract_mat(s, "VIvo", o, v), optimize="optimal");
E_CB[:,:] = E_CB[:,:] .+  +1.00000000  * fixed_einsum("aibj,bj,Cai->CB", extract_mat(L3, "AIvovo", o, v), extract_mat(r1, "vo", o, v), extract_mat(s, "VIvo", o, v), optimize="optimal");
E_AI= E_AI .+  +1.00000000  * fixed_einsum("Aai,ai->A", extract_mat(L2, "AIvo", o, v), extract_mat(r1, "vo", o, v), optimize="optimal");
E_II = E_II .+  +1.00000000  * fixed_einsum("ai,ai->", extract_mat(l1, "vo", o, v), extract_mat(r1, "vo", o, v), optimize="optimal");
E_II = E_II .+  -1.00000000  * fixed_einsum("Daibj,ai,Dbj->", extract_mat(L3, "VIvovo", o, v), extract_mat(r1, "vo", o, v), extract_mat(s, "VIvo", o, v), optimize="optimal");
E_II = E_II .+  -1.00000000  * fixed_einsum("Daibj,bj,Dai->", extract_mat(L3, "VIvovo", o, v), extract_mat(r1, "vo", o, v), extract_mat(s, "VIvo", o, v), optimize="optimal");
E_IA = E_IA .+  +2.00000000  * fixed_einsum("aibj,ai,Abj->A", extract_mat(l2, "vovo", o, v), extract_mat(r1, "vo", o, v), extract_mat(s, "AIvo", o, v), optimize="optimal");
