E_CB +=  +1.00000000 ;
E_CB = E_CB.+  -1.00000000  * fixed_einsum("Dai,Dai->", extract_mat(p, "VIvo", o, v), extract_mat(s, "VIvo", o, v), optimize="optimal");
E_CB = E_CB.+  -1.00000000  * fixed_einsum("Daibj,Daibj->", extract_mat(p2, "VIvovo", o, v), extract_mat(s2, "VIvovo", o, v), optimize="optimal");
