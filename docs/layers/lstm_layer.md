```# https://developer.apple.com/documentation/metalperformanceshaders/mpslstmdescriptor 
1. Let x_j be the input data (at time index t of sequence, j index containing quadruplet: batch index, x,y and feature index (x = y = 0 for matrices)).

2. Let h0_j be the recurrent input (previous output) data from previous time step (at time index t-1 of sequence).

3. Let h1_i be the output data produced at this time step.

4. Let c0_j be the previous memory cell data (at time index t-1 of sequence).

5. Let c1_i be the new memory cell data (at time index t-1 of sequence).

6. Let Wi_ij, Ui_ij, Vi_ij be the input gate weights for input, recurrent input, and memory cell (peephole) data, respectively.

7. Let bi_i be the bias for the input gate.

8. Let Wf_ij, Uf_ij, Vf_ij be the forget gate weights for input, recurrent input, and memory cell data, respectively.

9. Let bf_i be the bias for the forget gate.

10. Let Wo_ij, Uo_ij, Vo_ij be the output gate weights for input, recurrent input, and memory cell data, respectively.

11. Let bo_i be the bias for the output gate.

12. Let Wc_ij, Uc_ij, Vc_ij be the memory cell gate weights for input, recurrent input, and memory cell data, respectively.

13. Let bc_i be the bias for the memory cell gate.

14. Let gi(x), gf(x), go(x), gc(x) be the neuron activation function for the input, forget, output gate, and memory cell gate.

15. Let gh(x) be the activation function applied to result memory cell data.

```

The output of the LSTM layer is computed as follows:

```
I_i = gi(  Wi_ij * x_j  +  Ui_ij * h0_j  +  Vi_ij * c0_j  + bi_i  )
F_i = gf(  Wf_ij * x_j  +  Uf_ij * h0_j  +  Vf_ij * c0_j  + bf_i  )
C_i = gc(  Wc_ij * x_j  +  Uc_ij * h0_j  +  Vc_ij * c0_j  + bc_i  )

c1_i = F_i·c0_i  +  I_i·C_i

O_i = go(  Wo_ij * x_j  +  Uo_ij * h0_j  +  Vo_ij * c1_j  + bo_i  )
h1_i = O_i·gh( c1_i )
```