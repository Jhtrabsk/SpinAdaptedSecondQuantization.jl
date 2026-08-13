E_ia[:,:] +=  +2.00000000  * extract_mat(r1, "vo", o, v);
E_ia[:,:] = E_ia[:,:] .+  +1.00000000  * fixed_einsum("Abjai,Abj->ia", extract_mat(L3, "VIvovo", o, v), extract_mat(R2, "VIvo", o, v), optimize="optimal"); # We have 
E_ia[:,:] = E_ia[:,:] .+  +1.00000000  * fixed_einsum("Aaibj,Abj->ia", extract_mat(L3, "VIvovo", o, v), extract_mat(R2, "VIvo", o, v), optimize="optimal"); # We have

E_ba[:,:] = E_ba[:,:] .+  +1.00000000  * fixed_einsum("Aciaj,Acibj->ba", extract_mat(L3, "VIvovo", o, v), extract_mat(R3, "VIvovo", o, v), optimize="optimal"); ## Added
E_ba[:,:] = E_ba[:,:] .+  +1.00000000  * fixed_einsum("Aciaj,Abjci->ba", extract_mat(L3, "VIvovo", o, v), extract_mat(R3, "VIvovo", o, v), optimize="optimal"); ## Added
E_ji[:,:] = E_ji[:,:] .+  -1.00000000  * fixed_einsum("Aakbj,Aakbi->ji", extract_mat(L3, "VIvovo", o, v), extract_mat(R3, "VIvovo", o, v), optimize="optimal"); ## Added
E_ji[:,:] = E_ji[:,:] .+  -1.00000000  * fixed_einsum("Aakbj,Abiak->ji", extract_mat(L3, "VIvovo", o, v), extract_mat(R3, "VIvovo", o, v), optimize="optimal"); ## Added
E_ji[:,:] = E_ji[:,:] .+  -1.00000000  * fixed_einsum("Aajbk,Aaibk->ji", extract_mat(L3, "VIvovo", o, v), extract_mat(R3, "VIvovo", o, v), optimize="optimal"); ## Added
E_ji[:,:] = E_ji[:,:] .+  -1.00000000  * fixed_einsum("Aajbk,Abkai->ji", extract_mat(L3, "VIvovo", o, v), extract_mat(R3, "VIvovo", o, v), optimize="optimal"); ## Added
E_ba[:,:] = E_ba[:,:] .+  +1.00000000  * fixed_einsum("Aaicj,Acjbi->ba", extract_mat(L3, "VIvovo", o, v), extract_mat(R3, "VIvovo", o, v), optimize="optimal"); ## Added
E_ba[:,:] = E_ba[:,:] .+  +1.00000000  * fixed_einsum("Aaicj,Abicj->ba", extract_mat(L3, "VIvovo", o, v), extract_mat(R3, "VIvovo", o, v), optimize="optimal"); ## Added

E_ba[:,:] = E_ba[:,:] .+  +1.00000000  * fixed_einsum("Aciaj,Acibj->ba", extract_mat(L3, "VIvovo", o, v), extract_mat(s2, "VIvovo", o, v), optimize="optimal"); # We have
E_ji[:,:] = E_ji[:,:] .+  -1.00000000  * fixed_einsum("Aakbj,Aakbi->ji", extract_mat(L3, "VIvovo", o, v), extract_mat(s2, "VIvovo", o, v), optimize="optimal"); # We have
E_ji[:,:] = E_ji[:,:] .+  -1.00000000  * fixed_einsum("Aajbk,Aaibk->ji", extract_mat(L3, "VIvovo", o, v), extract_mat(s2, "VIvovo", o, v), optimize="optimal"); # We have
E_ba[:,:] = E_ba[:,:] .+  +1.00000000  * fixed_einsum("Aaicj,Acjbi->ba", extract_mat(L3, "VIvovo", o, v), extract_mat(s2, "VIvovo", o, v), optimize="optimal"); # We have

E_ba[:,:] = E_ba[:,:] .+  +1.00000000  * fixed_einsum("Aciaj,A,cibj->ba", extract_mat(L3, "VIvovo", o, v), extract_mat(R1, "VI", o, v), extract_mat(t, "vovo", o, v), optimize="optimal"); # Added
E_ji[:,:] = E_ji[:,:] .+  -1.00000000  * fixed_einsum("Aakbj,A,akbi->ji", extract_mat(L3, "VIvovo", o, v), extract_mat(R1, "VI", o, v), extract_mat(t, "vovo", o, v), optimize="optimal"); 
E_ji[:,:] = E_ji[:,:] .+  -1.00000000  * fixed_einsum("Aajbk,A,aibk->ji", extract_mat(L3, "VIvovo", o, v), extract_mat(R1, "VI", o, v), extract_mat(t, "vovo", o, v), optimize="optimal"); 
E_ba[:,:] = E_ba[:,:] .+  +1.00000000  * fixed_einsum("Aaicj,A,cjbi->ba", extract_mat(L3, "VIvovo", o, v), extract_mat(R1, "VI", o, v), extract_mat(t, "vovo", o, v), optimize="optimal"); # Added
E_ba[:,:] = E_ba[:,:] .+  +1.00000000  * fixed_einsum("Aciaj,ci,Abj->ba", extract_mat(L3, "VIvovo", o, v), extract_mat(r1, "vo", o, v), extract_mat(s, "VIvo", o, v), optimize="optimal"); # Added 
E_ji[:,:] = E_ji[:,:] .+  -1.00000000  * fixed_einsum("Aakbj,ak,Abi->ji", extract_mat(L3, "VIvovo", o, v), extract_mat(r1, "vo", o, v), extract_mat(s, "VIvo", o, v), optimize="optimal"); 
E_ji[:,:] = E_ji[:,:] .+  -1.00000000  * fixed_einsum("Aajbk,bk,Aai->ji", extract_mat(L3, "VIvovo", o, v), extract_mat(r1, "vo", o, v), extract_mat(s, "VIvo", o, v), optimize="optimal"); 
E_ba[:,:] = E_ba[:,:] .+  +1.00000000  * fixed_einsum("Aaicj,cj,Abi->ba", extract_mat(L3, "VIvovo", o, v), extract_mat(r1, "vo", o, v), extract_mat(s, "VIvo", o, v), optimize="optimal"); # Added

E_ai[:,:] = E_ai[:,:] .+  -1.00000000  * fixed_einsum("Abjck,Abi,ajck->ai", extract_mat(L3, "VIvovo", o, v), extract_mat(R2, "VIvo", o, v), extract_mat(t, "vovo", o, v), optimize="optimal"); # can reuse Added
E_ai[:,:] = E_ai[:,:] .+  -1.00000000  * fixed_einsum("Abjck,Aaj,bick->ai", extract_mat(L3, "VIvovo", o, v), extract_mat(R2, "VIvo", o, v), extract_mat(t, "vovo", o, v), optimize="optimal"); # can reuse Added
E_ai[:,:] = E_ai[:,:] .+  -1.00000000  * fixed_einsum("Abjck,Aak,bjci->ai", extract_mat(L3, "VIvovo", o, v), extract_mat(R2, "VIvo", o, v), extract_mat(t, "vovo", o, v), optimize="optimal"); # can reuse Added
E_ai[:,:] = E_ai[:,:] .+  -1.00000000  * fixed_einsum("Abjck,Aci,bjak->ai", extract_mat(L3, "VIvovo", o, v), extract_mat(R2, "VIvo", o, v), extract_mat(t, "vovo", o, v), optimize="optimal"); # can reuse Added

E_ai[:,:] = E_ai[:,:] .+  +1.00000000  * fixed_einsum("Abjck,Abj,aick->ai", extract_mat(L3, "VIvovo", o, v), extract_mat(R2, "VIvo", o, v), extract_mat(u, "vovo", o, v), optimize="optimal"); ## Added
E_ai[:,:] = E_ai[:,:] .+  +1.00000000  * fixed_einsum("Abjck,Ack,bjai->ai", extract_mat(L3, "VIvovo", o, v), extract_mat(R2, "VIvo", o, v), extract_mat(u, "vovo", o, v), optimize="optimal"); ## Added

E_ai[:,:] = E_ai[:,:] .+  +2.00000000  * fixed_einsum("Abjck,bj,Aaick->ai", extract_mat(L3, "VIvovo", o, v), extract_mat(r1, "vo", o, v), extract_mat(s2, "VIvovo", o, v), optimize="optimal"); # Added
E_ai[:,:] = E_ai[:,:] .+  -1.00000000  * fixed_einsum("Abjck,bj,Aakci->ai", extract_mat(L3, "VIvovo", o, v), extract_mat(r1, "vo", o, v), extract_mat(s2, "VIvovo", o, v), optimize="optimal"); # Added
E_ai[:,:] = E_ai[:,:] .+  -1.00000000  * fixed_einsum("Abjck,bi,Aajck->ai", extract_mat(L3, "VIvovo", o, v), extract_mat(r1, "vo", o, v), extract_mat(s2, "VIvovo", o, v), optimize="optimal"); # Added
E_ai[:,:] = E_ai[:,:] .+  -1.00000000  * fixed_einsum("Abjck,aj,Abick->ai", extract_mat(L3, "VIvovo", o, v), extract_mat(r1, "vo", o, v), extract_mat(s2, "VIvovo", o, v), optimize="optimal"); # Added
E_ai[:,:] = E_ai[:,:] .+  -1.00000000  * fixed_einsum("Abjck,ak,Abjci->ai", extract_mat(L3, "VIvovo", o, v), extract_mat(r1, "vo", o, v), extract_mat(s2, "VIvovo", o, v), optimize="optimal"); # Added
E_ai[:,:] = E_ai[:,:] .+  -1.00000000  * fixed_einsum("Abjck,ci,Abjak->ai", extract_mat(L3, "VIvovo", o, v), extract_mat(r1, "vo", o, v), extract_mat(s2, "VIvovo", o, v), optimize="optimal"); # Added
E_ai[:,:] = E_ai[:,:] .+  +2.00000000  * fixed_einsum("Abjck,ck,Abjai->ai", extract_mat(L3, "VIvovo", o, v), extract_mat(r1, "vo", o, v), extract_mat(s2, "VIvovo", o, v), optimize="optimal"); # Added
E_ai[:,:] = E_ai[:,:] .+  -1.00000000  * fixed_einsum("Abjck,ck,Abiaj->ai", extract_mat(L3, "VIvovo", o, v), extract_mat(r1, "vo", o, v), extract_mat(s2, "VIvovo", o, v), optimize="optimal"); # Added

E_ai[:,:] = E_ai[:,:] .+  -2.00000000  * fixed_einsum("Abjck,bjak,Aci->ai", extract_mat(L3, "VIvovo", o, v), extract_mat(r2, "vovo", o, v), extract_mat(s, "VIvo", o, v), optimize="optimal"); ## Added
E_ai[:,:] = E_ai[:,:] .+  -2.00000000  * fixed_einsum("AbjAk,bjci,Aak->ai", extract_mat(L3, "VIvovo", o, v), extract_mat(r2, "vovo", o, v), extract_mat(s, "VIvo", o, v), optimize="optimal"); ## Added
E_ai[:,:] = E_ai[:,:] .+  +4.00000000  * fixed_einsum("Abjck,bjck,Aai->ai", extract_mat(L3, "VIvovo", o, v), extract_mat(r2, "vovo", o, v), extract_mat(s, "VIvo", o, v), optimize="optimal"); ## Added
E_ai[:,:] = E_ai[:,:] .+  -2.00000000  * fixed_einsum("Abjck,bick,Aaj->ai", extract_mat(L3, "VIvovo", o, v), extract_mat(r2, "vovo", o, v), extract_mat(s, "VIvo", o, v), optimize="optimal"); ## Added
E_ai[:,:] = E_ai[:,:] .+  -2.00000000  * fixed_einsum("Abjck,ajck,Abi->ai", extract_mat(L3, "VIvovo", o, v), extract_mat(r2, "vovo", o, v), extract_mat(s, "VIvo", o, v), optimize="optimal"); ## Added

E_ai[:,:] = E_ai[:,:] .+  -1.00000000  * fixed_einsum("Abjck,Abi,ajck->ai", extract_mat(L3, "VIvovo", o, v), extract_mat(s, "VIvo", o, v), extract_mat(t, "vovo", o, v), optimize="optimal"); # We have
E_ai[:,:] = E_ai[:,:] .+  -1.00000000  * fixed_einsum("Abjck,Aaj,bick->ai", extract_mat(L3, "VIvovo", o, v), extract_mat(s, "VIvo", o, v), extract_mat(t, "vovo", o, v), optimize="optimal"); # We have
E_ai[:,:] = E_ai[:,:] .+  -1.00000000  * fixed_einsum("Abjck,Aak,bjci->ai", extract_mat(L3, "VIvovo", o, v), extract_mat(s, "VIvo", o, v), extract_mat(t, "vovo", o, v), optimize="optimal"); # We have
E_ai[:,:] = E_ai[:,:] .+  -1.00000000  * fixed_einsum("Abjck,Aci,bjak->ai", extract_mat(L3, "VIvovo", o, v), extract_mat(s, "VIvo", o, v), extract_mat(t, "vovo", o, v), optimize="optimal"); # We have
