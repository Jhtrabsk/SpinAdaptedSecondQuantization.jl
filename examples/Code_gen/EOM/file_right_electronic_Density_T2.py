E_ai[:,:] +=  +2.00000000  * extract_mat(r1, "vo", o, v);
E_ia[:,:] = E_ia[:,:] .+  +2.00000000  * fixed_einsum("bjai,bj->ia", extract_mat(l2, "vovo", o, v), extract_mat(r1, "vo", o, v), optimize="optimal");
E_ji[:,:] = E_ji[:,:] .+  -4.00000000  * fixed_einsum("akbj,akbi->ji", extract_mat(l2, "vovo", o, v), extract_mat(r2, "vovo", o, v), optimize="optimal");
E_ba[:,:] = E_ba[:,:] .+  +4.00000000  * fixed_einsum("ciaj,cibj->ba", extract_mat(l2, "vovo", o, v), extract_mat(r2, "vovo", o, v), optimize="optimal");
E_ji[:,:] = E_ji[:,:] .+  -2.00000000  * fixed_einsum("akbj,akbi->ji", extract_mat(l2, "vovo", o, v), extract_mat(t, "vovo", o, v), optimize="optimal");
E_ba[:,:] = E_ba[:,:] .+  +2.00000000  * fixed_einsum("ciaj,cibj->ba", extract_mat(l2, "vovo", o, v), extract_mat(t, "vovo", o, v), optimize="optimal");
E_ai[:,:] = E_ai[:,:] .+  -2.00000000  * fixed_einsum("bjck,bi,ckaj->ai", extract_mat(l2, "vovo", o, v), extract_mat(r1, "vo", o, v), extract_mat(t, "vovo", o, v), optimize="optimal");
E_ai[:,:] = E_ai[:,:] .+  -2.00000000  * fixed_einsum("bjck,aj,bick->ai", extract_mat(l2, "vovo", o, v), extract_mat(r1, "vo", o, v), extract_mat(t, "vovo", o, v), optimize="optimal");
E_ai[:,:] = E_ai[:,:] .+  +2.00000000  * fixed_einsum("bjck,bj,ckai->ai", extract_mat(l2, "vovo", o, v), extract_mat(r1, "vo", o, v), extract_mat(u, "vovo", o, v), optimize="optimal");