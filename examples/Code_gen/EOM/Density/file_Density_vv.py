E_ba[:,:] = E_ba[:,:] .+  +1.00000000  * fixed_einsum("Aai,Abi->ba", extract_mat(p, "VIvo", o, v), extract_mat(s, "VIvo", o, v), optimize="optimal");
E_ba[:,:] = E_ba[:,:] .+  +1.00000000  * fixed_einsum("ciaj,cibj->ba", extract_mat(c2, "vovo", o, v), extract_mat(t, "vovo", o, v), optimize="optimal");
E_ba[:,:] = E_ba[:,:] .+  +0.50000000  * fixed_einsum("Aciaj,Acibj->ba", extract_mat(p2, "VIvovo", o, v), extract_mat(s2, "VIvovo", o, v), optimize="optimal");
E_ba[:,:] = E_ba[:,:] .+  +0.50000000  * fixed_einsum("Aaicj,Acjbi->ba", extract_mat(p2, "VIvovo", o, v), extract_mat(s2, "VIvovo", o, v), optimize="optimal");

