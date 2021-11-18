package main

import (
	"fmt"
	"math"
	"time"

	"github.com/ldsec/lattigo/v2/ckks"
)

// take raw_in_wid then outputs appropriate kp_wid and out_batch
func set_Variables(batch, raw_in_wid, in_wid, ker_wid int, kind string) (kp_wid, out_batch, logN int, trans bool) {
	N := batch * in_wid * in_wid
	logN = 0
	for ; (1 << logN) < N; logN++ {
	}
	max_kp_wid := in_wid - ((ker_wid - 1) / 2) // max possible size of raw_in_wid

	switch kind {
	case "Conv", "StrConv":
		trans = false
		kp_wid = raw_in_wid
		out_batch = batch
		if kp_wid > max_kp_wid {
			fmt.Println("max raw_in_wid: ", max_kp_wid)
			panic("too large raw_in_wid.")
		}
	case "TransConv":
		trans = true
		kp_wid = 2 * raw_in_wid
		out_batch = batch / 4
		if kp_wid > max_kp_wid {
			fmt.Println("max raw_in_wid: ", max_kp_wid/2)
			panic("too large raw_in_wid.")
		}
	default:
		panic("Wrong kinds!")
	}

	return
}

// Eval Conv only, always assume max batch
// in_wid must be Po2 (also include padding),
// include kernel preparation
func evalConv_BN(cont *context, ct_input *ckks.Ciphertext, ker_in, bn_a, bn_b []float64, in_wid, ker_wid, real_ib, real_ob int, printResult, trans bool) (ct_res *ckks.Ciphertext) {
	max_batch := cont.N / (in_wid * in_wid)

	fmt.Println()
	fmt.Println("===============  (KER) PREPARATION  ===============")
	fmt.Println()
	start = time.Now()
	pl_ker := prep_Ker(cont.params, cont.encoder, ker_in, bn_a, in_wid, ker_wid, real_ib, real_ob, cont.ECD_LV, 0, trans)
	b_coeffs := make([]float64, cont.N)
	for i := range bn_b {
		for j := 0; j < in_wid; j++ {
			for k := 0; k < in_wid; k++ {
				b_coeffs[i+(j+k*in_wid)*max_batch] = bn_b[i]
			}
		}
	}
	scale_exp := cont.params.Scale() * cont.params.Scale() * float64(max_batch)
	pl_bn_b := ckks.NewPlaintext(cont.params, cont.ECD_LV, scale_exp) // contain plaintext values
	cont.encoder.EncodeCoeffs(b_coeffs, pl_bn_b)
	cont.encoder.ToNTT(pl_bn_b)
	fmt.Printf("Plaintext (kernel) preparation, Done in %s \n", time.Since(start))

	fmt.Println()
	fmt.Println("===============  EVALUATION  ===============")
	fmt.Println()

	start = time.Now()
	ct_res = conv_then_pack(cont.params, cont.pack_evaluator, ct_input, pl_ker, cont.pl_idx, max_batch, cont.ECD_LV, scale_exp)
	cont.evaluator.Add(ct_res, pl_bn_b, ct_res) // for Batch Normalization (BN)
	fmt.Printf("Conv (with BN) Done in %s \n", time.Since(start))

	return ct_res
}

// Eval Conv, BN, relu with Boot
// in_wid must be Po2, BN is fold with Kernel
// stride = true: apply [1,2,2,1] stride; false: [1,1,1,1]
// pack_pos: position to pack (0,1,2,3): only for strided case
// real_ib, real_ob: real number of batches (less or equal than max_batch)
func evalConv_BNRelu(cont *context, ct_input *ckks.Ciphertext, ker_in, bn_a, bn_b []float64, alpha float64, in_wid, ker_wid, real_ib, real_ob, pack_pos int, padding, stride, printResult bool) (ct_res *ckks.Ciphertext) {
	trans := false
	kp_wid := in_wid - ((ker_wid - 1) / 2)
	ct_conv := evalConv_BN(cont, ct_input, ker_in, bn_a, bn_b, in_wid, ker_wid, real_ib, real_ob, printResult, trans)

	val_preB := cont.encoder.DecodeCoeffs(cont.decryptor.DecryptNew(ct_conv))
	fmt.Println("Bootstrapping... Ours (until CtoS):")
	start = time.Now()
	ct_boots := make([]*ckks.Ciphertext, 2)
	ct_boots[0], ct_boots[1], _ = cont.btp.BootstrappConv_CtoS(ct_conv, float64(pow))
	fmt.Printf("Done in %s \n", time.Since(start))
	fmt.Println("after Boot (CtoS): LV = ", ct_boots[0].Level(), " Scale = ", math.Log2(ct_boots[0].Scale))

	// Only for checking the correctness
	in_cfs1_preB := make([]float64, cont.params.Slots())
	in_cfs2_preB := make([]float64, cont.params.Slots())
	in_slots1 := make([]complex128, cont.params.Slots()) // first part of ceffs
	in_slots2 := make([]complex128, cont.params.Slots()) // second part of ceffs
	for i := range in_cfs1_preB {
		in_cfs1_preB[i] = val_preB[reverseBits(uint32(i), cont.params.LogSlots())] // first part of coeffs
		in_cfs2_preB[i] = val_preB[reverseBits(uint32(i), cont.params.LogSlots())+uint32(cont.params.Slots())]
		in_slots1[i] = complex(in_cfs1_preB[i]/math.Pow(2, float64(pow)), 0)
		in_slots2[i] = complex(in_cfs2_preB[i]/math.Pow(2, float64(pow)), 0)
	}
	ext1_tmp := keep_vec(in_cfs1_preB, in_wid, kp_wid, 0)
	ext2_tmp := keep_vec(in_cfs2_preB, in_wid, kp_wid, 1)
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
			// dec_tmp := cont.decryptor.DecryptNew(ct_keep[ul])
			// cf_ttmp := cont.encoder.Decode(dec_tmp, cont.params.LogSlots())
			// fmt.Println("itermediate: ", cf_ttmp)
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

	return ct_res
}

// Eval Conv, BN, relu with Boot
// in_wid must be Po2 (also include padding)
// stride = true: apply [1,2,2,1] stride; false: [1,1,1,1]
// pack_pos: position to pack (0,1,2,3): only for strided case
// real_ib, real_ob: real number of batches (less or equal than max_batch)
func evalConv_BNRelu_new(cont *context, ct_input *ckks.Ciphertext, ker_in, bn_a, bn_b []float64, alpha float64, in_wid, kp_wid, ker_wid, real_ib, real_ob, pack_pos int, kind string, printResult bool) (ct_res *ckks.Ciphertext) {
	// kp_wid := in_wid - ((ker_wid - 1) / 2)
	iter := 2 // for full packing (contrary to half packing)
	var trans, stride bool
	switch kind {
	case "Conv":
		trans = false
		stride = false
	case "StrConv":
		trans = false
		stride = true
	case "TransConv":
		trans = true
		stride = false
	}

	ct_conv := evalConv_BN(cont, ct_input, ker_in, bn_a, bn_b, in_wid, ker_wid, real_ib, real_ob, printResult, trans)

	cfs_preB := cont.encoder.DecodeCoeffs(cont.decryptor.DecryptNew(ct_conv))
	fmt.Println("Bootstrapping... Ours (until CtoS):")
	start = time.Now()
	ct_boots := make([]*ckks.Ciphertext, 2)
	ct_boots[0], ct_boots[1], _ = cont.btp.BootstrappConv_CtoS(ct_conv, float64(pow))
	fmt.Printf("Done in %s \n", time.Since(start))
	fmt.Println("after Boot (CtoS): LV = ", ct_boots[0].Level(), " Scale = ", math.Log2(ct_boots[0].Scale))

	// Only for checking the correctness (for CtoS)
	slot1, slot2 := debugCtoS(cont, cfs_preB)
	slot1 = printDebug(cont.params, ct_boots[0], slot1, cont.decryptor, cont.encoder) // Compare before & after CtoS
	slot2 = printDebug(cont.params, ct_boots[1], slot2, cont.decryptor, cont.encoder) // Compare before & after CtoS

	start = time.Now()
	for ul := 0; ul < iter; ul++ { // up & low parts
		ct_boots[ul] = evalReLU(cont.params, cont.evaluator, ct_boots[ul], alpha)
		cont.evaluator.MulByPow2(ct_boots[ul], pow, ct_boots[ul])
	}
	fmt.Printf("ReLU Done in %s \n", time.Since(start))

	// Only for checking the correctness (for ReLU)
	relu1, relu2 := debugReLU(cont, slot1, slot2, alpha)
	relu1 = printDebug(cont.params, ct_boots[0], relu1, cont.decryptor, cont.encoder)
	relu2 = printDebug(cont.params, ct_boots[1], relu2, cont.decryptor, cont.encoder)
	cfs_postB := debugStoC(cont, relu1, relu2, in_wid, kp_wid, kind)

	start = time.Now()
	ct_keep := make([]*ckks.Ciphertext, iter) // for extend (rotation) of ctxt_in
	for ul := 0; ul < iter; ul++ {
		if trans || stride {
			ct_keep[ul] = ext_ctxt(cont.evaluator, cont.encoder, ct_boots[ul], cont.r_idx[in_wid][ul], cont.params)
		} else {
			ct_keep[ul] = keep_ctxt(cont.params, cont.evaluator, cont.encoder, ct_boots[ul], cont.ext_idx[in_wid][ul])
		}
	}
	if iter == 1 {
		ct_boots[1] = nil
		ct_res = cont.btp.BootstrappConv_StoC(ct_keep[0], ct_boots[1])
	} else {
		ct_res = cont.btp.BootstrappConv_StoC(ct_keep[0], ct_keep[1])
	}
	cont.evaluator.Rescale(ct_res, cont.params.Scale(), ct_res)

	fmt.Printf("Boot (StoC) Done in %s \n", time.Since(start))
	fmt.Printf("Boot out: ")
	// Only for checking the correctness (for StoC)
	printDebugCfs(cont.params, ct_res, cfs_postB, cont.decryptor, cont.encoder)

	return ct_res
}

func debugCtoS(cont *context, cfs_preB []float64) (slot1, slot2 []complex128) {
	preB_cfs1 := make([]float64, cont.params.Slots())
	preB_cfs2 := make([]float64, cont.params.Slots())
	slot1 = make([]complex128, cont.params.Slots()) // first part of ceffs
	slot2 = make([]complex128, cont.params.Slots()) // second part of ceffs
	for i := range preB_cfs1 {
		preB_cfs1[i] = cfs_preB[reverseBits(uint32(i), cont.params.LogSlots())] // first part of coeffs
		preB_cfs2[i] = cfs_preB[reverseBits(uint32(i), cont.params.LogSlots())+uint32(cont.params.Slots())]
		slot1[i] = complex(preB_cfs1[i]/math.Pow(2, float64(pow)), 0)
		slot2[i] = complex(preB_cfs2[i]/math.Pow(2, float64(pow)), 0)
	}
	return
}

func debugReLU(cont *context, slot1, slot2 []complex128, alpha float64) (relu1, relu2 []complex128) {
	relu1 = make([]complex128, len(slot1))
	relu2 = make([]complex128, len(slot1))
	for i := range relu1 {
		relu1[i] = complex((math.Max(0, real(slot1[i]))+math.Min(0, real(slot1[i])*alpha))*math.Pow(2, float64(pow)), 0)
		relu2[i] = complex((math.Max(0, real(slot2[i]))+math.Min(0, real(slot2[i])*alpha))*math.Pow(2, float64(pow)), 0)
	}
	return
}

func debugStoC(cont *context, slot1, slot2 []complex128, in_wid, kp_wid int, kind string) (cfs_postB []float64) {
	slot1_fl := make([]float64, len(slot1))
	slot2_fl := make([]float64, len(slot1))
	for i := range slot1 {
		slot1_fl[i] = real(slot1[i])
		slot2_fl[i] = real(slot2[i])
	}

	var tmp1, tmp2 []float64
	switch kind {
	case "Conv":
		tmp1 = keep_vec(slot1_fl, in_wid, kp_wid, 0)
		tmp2 = keep_vec(slot2_fl, in_wid, kp_wid, 1)
	case "StrConv":
		tmp1 = comprs_full(slot1_fl, in_wid, kp_wid, 0, 0)
		tmp2 = comprs_full(slot2_fl, in_wid, kp_wid, 0, 1)
	case "TransConv":
		tmp1 = extend_full(slot1_fl, in_wid, kp_wid, 0, 0)
		tmp2 = extend_full(slot2_fl, in_wid, kp_wid, 0, 1)
	}

	cfs_postB1 := make([]float64, len(slot1))
	cfs_postB2 := make([]float64, len(slot1))
	for i := range cfs_postB1 {
		cfs_postB1[i] = tmp1[reverseBits(uint32(i), cont.params.LogSlots())]
		cfs_postB2[i] = tmp2[reverseBits(uint32(i), cont.params.LogSlots())]
	}
	cfs_postB = append(cfs_postB1, cfs_postB2...) // After rot(ext) and boot
	return
}
