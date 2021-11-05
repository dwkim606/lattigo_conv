package main

import (
	"fmt"
	"math"
	"time"

	"github.com/ldsec/lattigo/v2/ckks"
)

// Eval Conv, BN, relu with Boot
// in_wid must be Po2, BN is fold with Kernel
// stride = true: apply [1,2,2,1] stride; false: [1,1,1,1]
// pack_pos: position to pack (0,1,2,3): only for strided case
func evalConv_BNRelu(cont *context, ct_input *ckks.Ciphertext, ker_in, bn_a, bn_b []float64, alpha float64, in_wid, ker_wid, pack_pos int, padding, stride, printResult bool) (ct_res *ckks.Ciphertext) {

	in_size := in_wid * in_wid
	batch := cont.N / in_size
	pad := (ker_wid - 1) / 2

	start = time.Now()
	b_coeffs := make([]float64, cont.N)
	for i := range bn_b {
		for j := 0; j < in_wid; j++ {
			for k := 0; k < in_wid; k++ {
				b_coeffs[i+(j+k*in_wid)*batch] = bn_b[i]
			}
		}
	}
	pl_ker := prepKer_in(cont.params, cont.encoder, ker_in, bn_a, in_wid, ker_wid, batch, batch, cont.ECD_LV)
	fmt.Printf("Plaintext (kernel) preparation, Done in %s \n", time.Since(start))

	fmt.Println("Ker1_in (1st part): ")
	prt_vec(ker_in)

	fmt.Println()
	fmt.Println("===============================================")
	fmt.Println("     			   EVALUATION					")
	fmt.Println("===============================================")
	fmt.Println()

	start = time.Now()
	ct_conv := conv_then_pack(cont.params, cont.pack_evaluator, ct_input, pl_ker, cont.pl_idx, batch)

	// for Batch Normalization (BN)
	pl_bn_b := ckks.NewPlaintext(cont.params, ct_conv.Level(), ct_conv.Scale) // contain plaintext values
	cont.encoder.EncodeCoeffs(b_coeffs, pl_bn_b)
	cont.encoder.ToNTT(pl_bn_b)
	cont.evaluator.Add(ct_conv, pl_bn_b, ct_conv)
	fmt.Printf("Conv (with BN) Done in %s \n", time.Since(start))

	pl_conv := ckks.NewPlaintext(cont.params, ct_conv.Level(), cont.params.Scale())
	cont.decryptor.Decrypt(ct_conv, pl_conv)
	val_preB := cont.encoder.DecodeCoeffs(pl_conv)

	fmt.Println("Boot in: ")
	fmt.Println("Precision of values vs. ciphertext")
	in_cfs := printDebugCfs(cont.params, ct_conv, val_preB, cont.decryptor, cont.encoder)

	fmt.Println("Bootstrapping... Ours (until CtoS):")
	start = time.Now()
	ct_boots := make([]*ckks.Ciphertext, 2)
	ct_boots[0], ct_boots[1], _ = cont.btp.BootstrappConv_CtoS(ct_conv)
	fmt.Printf("Done in %s \n", time.Since(start))
	fmt.Println("after Boot (CtoS): LV = ", ct_boots[0].Level(), " Scale = ", math.Log2(ct_boots[0].Scale))

	// Only for checking the correctness
	in_cfs1_preB := make([]float64, cont.params.Slots())
	in_cfs2_preB := make([]float64, cont.params.Slots())
	in_slots1 := make([]complex128, cont.params.Slots()) // first part of ceffs
	in_slots2 := make([]complex128, cont.params.Slots()) // second part of ceffs
	for i := range in_cfs1_preB {
		in_cfs1_preB[i] = in_cfs[reverseBits(uint32(i), cont.params.LogSlots())] // first part of coeffs
		in_cfs2_preB[i] = in_cfs[reverseBits(uint32(i), cont.params.LogSlots())+uint32(cont.params.Slots())]
		in_slots1[i] = complex(in_cfs1_preB[i]/math.Pow(2, float64(pow)), 0)
		in_slots2[i] = complex(in_cfs2_preB[i]/math.Pow(2, float64(pow)), 0)
	}
	ext1_tmp := keep_vec_fl(in_cfs1_preB, in_wid, pad, 0)
	ext2_tmp := keep_vec_fl(in_cfs2_preB, in_wid, pad, 1)
	for i := range in_cfs1_preB {
		in_cfs1_preB[i] = ext1_tmp[reverseBits(uint32(i), cont.params.LogSlots())]
		in_cfs2_preB[i] = ext2_tmp[reverseBits(uint32(i), cont.params.LogSlots())]
	}
	in_cfs_pBoot := append(in_cfs1_preB, in_cfs2_preB...)                                     // After rot(ext) and boot
	in_slots1 = printDebug(cont.params, ct_boots[0], in_slots1, cont.decryptor, cont.encoder) // Compare before & after CtoS
	in_slots2 = printDebug(cont.params, ct_boots[1], in_slots2, cont.decryptor, cont.encoder) // Compare before & after CtoS

	var iter int
	if padding {
		iter = 1
	} else {
		iter = 2
	}
	start = time.Now()
	for ul := 0; ul < iter; ul++ { // up & low parts
		ct_boots[ul] = evalReLU(cont.params, cont.evaluator, ct_boots[ul], alpha)
		cont.evaluator.MulByPow2(ct_boots[ul], pow, ct_boots[ul])
	}
	fmt.Printf("ReLU Done in %s \n", time.Since(start))

	values_ReLU := make([]complex128, len(in_slots1))
	for i := range values_ReLU {
		values_ReLU[i] = complex(math.Max(0, real(in_slots1[i])*math.Pow(2, float64(pow))), 0)
	}
	printDebug(cont.params, ct_boots[0], values_ReLU, cont.decryptor, cont.encoder)
	for i := range values_ReLU {
		values_ReLU[i] = complex(math.Max(0, real(in_slots2[i])*math.Pow(2, float64(pow))), 0)
	}
	printDebug(cont.params, ct_boots[1], values_ReLU, cont.decryptor, cont.encoder)

	ct_keep := make([]*ckks.Ciphertext, iter) // for extend (rotation) of ctxt_in

	start = time.Now()
	for ul := 0; ul < iter; ul++ {
		if stride {
			ct_keep[ul] = ext_ctxt(cont.evaluator, cont.encoder, ct_boots[ul], cont.r_idx[in_wid][pack_pos], cont.params)
		} else {
			ct_keep[ul] = keep_ctxt(cont.params, cont.evaluator, cont.encoder, ct_boots[ul], cont.ext_idx[in_wid][ul])
		}
	}
	if padding {
		ct_boots[1] = nil
		ct_res = cont.btp.BootstrappConv_StoC(ct_keep[0], ct_boots[1])
	} else {
		ct_res = cont.btp.BootstrappConv_StoC(ct_keep[0], ct_keep[1])
	}
	cont.evaluator.Rescale(ct_res, cont.params.Scale(), ct_res)

	fmt.Printf("Boot (StoC) Done in %s \n", time.Since(start))

	fmt.Printf("Boot out: ")
	printDebugCfs(cont.params, ct_res, in_cfs_pBoot, cont.decryptor, cont.encoder)

	start = time.Now()
	cont.decryptor.Decrypt(ct_res, pl_conv)
	cfs_tmp := cont.encoder.DecodeCoeffs(pl_conv)
	fmt.Printf("Decryption Done in %s \n", time.Since(start))

	if printResult {
		fmt.Println()
		fmt.Println("=========================================")
		fmt.Println("              DECRYPTION                 ")
		fmt.Println("=========================================")
		fmt.Println()

		fmt.Print("Result: \n")
		if stride {
			prt_mat(cfs_tmp, batch*4, 0)
		} else {
			prt_mat(cfs_tmp, batch, 0)
		}
	}

	return ct_res
}

// Eval Conv only
// in_wid must be Po2,
func evalConv_BN(cont *context, ct_input *ckks.Ciphertext, ker_in, bn_a, bn_b []float64, in_wid, ker_wid int, printResult bool) (ct_res *ckks.Ciphertext) {

	in_size := in_wid * in_wid
	batch := cont.N / in_size

	start = time.Now()
	b_coeffs := make([]float64, cont.N)
	for i := range bn_b {
		for j := 0; j < in_wid; j++ {
			for k := 0; k < in_wid; k++ {
				b_coeffs[i+(j+k*in_wid)*batch] = bn_b[i]
			}
		}
	}
	pl_ker := prepKer_in(cont.params, cont.encoder, ker_in, bn_a, in_wid, ker_wid, batch, batch, cont.ECD_LV)
	fmt.Printf("Plaintext (kernel) preparation, Done in %s \n", time.Since(start))

	fmt.Println("Ker1_in (1st part): ")
	prt_vec(ker_in)

	fmt.Println()
	fmt.Println("===============================================")
	fmt.Println("     			   EVALUATION					")
	fmt.Println("===============================================")
	fmt.Println()

	start = time.Now()
	ct_conv := conv_then_pack(cont.params, cont.pack_evaluator, ct_input, pl_ker, cont.pl_idx, batch)

	// for Batch Normalization (BN)
	pl_bn_b := ckks.NewPlaintext(cont.params, ct_conv.Level(), ct_conv.Scale) // contain plaintext values
	cont.encoder.EncodeCoeffs(b_coeffs, pl_bn_b)
	cont.encoder.ToNTT(pl_bn_b)
	ct_res = cont.evaluator.AddNew(ct_conv, pl_bn_b)
	fmt.Printf("Conv (with BN) Done in %s \n", time.Since(start))

	return ct_res
}
