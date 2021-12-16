package main

import (
	"fmt"
	"math"
	"time"

	"github.com/ldsec/lattigo/v2/ckks"
)

// take raw_in_wid then outputs appropriate kp_wid and out_batch
// only for our convs Test (not for BL)
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

// apply rotation for strided conv (compress) or transposed conv (extend)
// the same rotation for all batches; use BSGS to reduce rotations
// assume that input batches are well-ordered. compress: (0,4) (1,5) (2,6) (3,7) to (0,1,2,...6,7) extend: (0,2,4,6,1,3,5,7) to (0,1) (2,3) (4,5) (6,7)
// rotation for each batch position (0 to 3) is applied after or before compress or extend, resp.
// total rotation = 2*in_wid*4 + (4-1); depth = 2
func evalRot_BL(cont *context, ct_input *ckks.Ciphertext, in_wid, pos int, trans bool) (ct_res *ckks.Ciphertext) {
	if trans {
		in_size := in_wid * in_wid
		cont.evaluator.Rotate(ct_input, pos*in_size, ct_input)
		ct_res = bsgs_ctxt(cont.evaluator, cont.encoder, ct_input, cont.m_idx[in_wid][0], cont.r_idx[in_wid][0], cont.params)
	} else {
		out_size := in_wid * in_wid / 4
		ct_res = bsgs_ctxt(cont.evaluator, cont.encoder, ct_input, cont.m_idx[in_wid][0], cont.r_idx[in_wid][0], cont.params)
		cont.evaluator.Rotate(ct_res, -pos*out_size, ct_res)
	}
	return
}

// Eval Conv only, always assume max batch
// in_wid must be Po2 (also include padding), includes kernel preparation
// norm == 1 : normal case, norm == 4 : in & out batch is (1,0,0,0,2,0,0,0,3,0,0,0,4,0,0,0)
func evalConv_BN_BL(cont *context, ct_input *ckks.Ciphertext, ker_in, bn_a, bn_b []float64, in_wid, ker_wid, real_ib, real_ob, pos, norm int, trans, printResult bool) (ct_res *ckks.Ciphertext) {
	in_size := in_wid * in_wid
	out_size := in_size
	max_batch := cont.N / (2 * in_size)

	fmt.Println()
	fmt.Println("===============  (KER) PREPARATION  ===============")
	fmt.Println()
	start = time.Now()
	max_ker_rs := reshape_ker_BL(ker_in, bn_a, ker_wid, real_ib, real_ob, max_batch, pos, norm, trans)
	scale_exp := cont.params.Scale() * cont.params.Scale()
	if trans {
		scale_exp = cont.params.Scale() * cont.params.Scale() * cont.params.Scale()
	}
	bn_b_slots := make([]complex128, cont.N/2)
	for i, elt := range bn_b {
		for j := 0; j < out_size; j++ {
			bn_b_slots[j+norm*out_size*i] = complex(elt, 0)
		}
	}

	pl_bn_b := ckks.NewPlaintext(cont.params, cont.ECD_LV, scale_exp)
	cont.encoder.EncodeNTT(pl_bn_b, bn_b_slots, cont.logN-1)
	fmt.Printf("Plaintext (kernel) preparation, Done in %s \n", time.Since(start))

	fmt.Println()
	fmt.Println("===============  EVALUATION  ===============")
	fmt.Println()
	start = time.Now()
	ct_inputs_rots := preConv_BL(cont.evaluator, ct_input, in_wid, ker_wid)
	fmt.Printf("preConv done in %s \n", time.Since(start))

	var rot_iters int
	if norm*real_ob == max_batch {
		rot_iters = real_ob
	} else {
		rot_iters = max_batch
	}
	for i := 0; i < rot_iters; i++ {
		ct_tmp := postConv_BL(cont.params, cont.encoder, cont.evaluator, ct_inputs_rots, in_wid, ker_wid, norm*i, max_ker_rs)
		if i == 0 {
			ct_res = ct_tmp
		} else {
			cont.evaluator.Add(ct_res, cont.evaluator.RotateNew(ct_tmp, norm*i*out_size), ct_res)
		}
	}

	if ct_res.Scale != scale_exp {
		panic("Different scale between pl_bn_b and ctxt")
	}
	cont.evaluator.Add(ct_res, pl_bn_b, ct_res)
	fmt.Printf("Conv (with BN) Done in %s \n", time.Since(start))

	return ct_res
}

// Eval Conv only, always assume max batch
// in_wid must be Po2 (also include padding),
// include kernel preparation
func evalConv_BNRelu_BL(cont *context, ct_input *ckks.Ciphertext, ker_in, bn_a, bn_b []float64, alpha float64, in_wid, ker_wid, real_ib, real_ob, norm int, strides, trans, printResult bool) (ct_res *ckks.Ciphertext) {
	var ct_input_rot, ct_conv *ckks.Ciphertext
	rot_time := time.Duration(0)
	eval_time := time.Duration(0)
	if trans {
		for pos := 0; pos < 4; pos++ {
			start = time.Now()
			ct_input_rot = evalRot_BL(cont, ct_input, in_wid, pos, trans)
			rot_time += time.Since(start)
			start = time.Now()
			if pos == 0 {
				ct_conv = evalConv_BN_BL(cont, ct_input_rot, ker_in, bn_a, bn_b, 2*in_wid, ker_wid, real_ib, real_ob, pos, norm, trans, printResult)
			} else {
				ct_tmp := evalConv_BN_BL(cont, ct_input_rot, ker_in, bn_a, bn_b, 2*in_wid, ker_wid, real_ib, real_ob, pos, norm, trans, printResult)
				cont.evaluator.Add(ct_conv, ct_tmp, ct_conv)
			}
			eval_time += time.Since(start)
		}
		fmt.Printf("Rotation (for transConv) Done in %s \n", rot_time)
		fmt.Printf("EvalConv total (for transConv) Done in %s \n", eval_time)
	} else {
		ct_conv = evalConv_BN_BL(cont, ct_input, ker_in, bn_a, bn_b, in_wid, ker_wid, real_ib, real_ob, 0, norm, trans, printResult)
		if strides {
			start = time.Now()
			ct_conv = evalRot_BL(cont, ct_conv, in_wid, 0, trans)
			fmt.Printf("Rotation (for strided Conv) Done in %s \n", time.Since(start))
		}
	}

	// if trans {
	// 	start = time.Now()
	// 	ct_input = evalRot_BL(cont, ct_input, in_wid, 0, trans)
	// 	fmt.Printf("Rotation (for transConv) Done in %s \n", time.Since(start))
	// }
	// ct_conv := evalConv_BN_BL(cont, ct_input, ker_in, bn_a, bn_b, in_wid, ker_wid, real_ib, real_ob, 0, trans, printResult)
	// if strides {
	// 	start = time.Now()
	// 	ct_conv = evalRot_BL(cont, ct_conv, in_wid, 0, trans)
	// 	fmt.Printf("Rotation (for strided Conv) Done in %s \n", time.Since(start))
	// }

	ct_conv.Scale = ct_conv.Scale * math.Pow(2, pow)
	vals_preB := cont.encoder.Decode(cont.decryptor.DecryptNew(ct_conv), cont.logN-1)
	fmt.Println("\n ========= Bootstrapping... (original) ========= ")
	start_boot := time.Now()
	// cont.evaluator.SetScale(ct_conv, ct_conv.Scale*8)
	// cont.evaluator.SetScale(ct_conv, ct_conv.Scale*math.Pow(2, pow))
	fmt.Println("initial (before boot): LV = ", ct_conv.Level(), " Scale = ", math.Log2(ct_conv.Scale))

	ct_boot := cont.btp.Bootstrapp(ct_conv)
	fmt.Printf("Done in %s \n", time.Since(start_boot))
	fmt.Println("after Boot: LV = ", ct_boot.Level(), " Scale = ", math.Log2(ct_boot.Scale))

	// Only for checking the correctness (for Boot)
	vals_postB := printDebug(cont.params, ct_boot, vals_preB, cont.decryptor, cont.encoder)
	vals_relu := make([]complex128, len(vals_postB))
	for i, elt := range vals_postB {
		vals_relu[i] = complex((math.Max(0, real(elt))+math.Min(0, real(elt)*alpha))*math.Pow(2, pow), 0)
	}

	start = time.Now()
	pl_scale := ckks.NewPlaintext(cont.params, ct_boot.Level(), math.Pow(2, 30)*float64(cont.params.Q()[14])*float64(cont.params.Q()[13])/ct_boot.Scale)
	val_scale := make([]complex128, cont.N/2)
	for i := range val_scale {
		val_scale[i] = complex(1.0, 0) // val_scale[i] = complex(1.0/math.Pow(2, pow), 0)
	}
	cont.encoder.EncodeNTT(pl_scale, val_scale, cont.logN-1)
	cont.evaluator.Mul(ct_boot, pl_scale, ct_boot)
	cont.evaluator.Rescale(ct_boot, cont.params.Scale(), ct_boot)

	fmt.Println("after Rescale: LV = ", ct_boot.Level(), " Scale = 2^", math.Log2(ct_boot.Scale))
	ct_res = evalReLU(cont.params, cont.evaluator, ct_boot, alpha)
	cont.evaluator.MulByPow2(ct_res, pow, ct_res)
	cont.evaluator.SetScale(ct_res, cont.params.Scale())
	fmt.Printf("Relu Done in %s \n", time.Since(start))
	printDebug(cont.params, ct_res, vals_relu, cont.decryptor, cont.encoder)

	if printResult {
		vals_tmp := cont.encoder.Decode(cont.decryptor.DecryptNew(ct_res), cont.logN-1)
		fmt.Print("Result: \n")
		if strides {
			prt_mat_BL(vals_tmp, (cont.N / (in_wid * in_wid / 2)), 3)
		} else {
			prt_mat_BL(vals_tmp, (cont.N / (2 * in_wid * in_wid)), 3)
		}
	}

	return ct_res
}

// reduce mean and final FC layer (in_batch -> 16)
// assume that ct_input has batch (1,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,3,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0)
// ker_fc is of size in_batch*10 and 1-dim from [64,10] shape
func evalRMFC_BL(cont *context, ct_input *ckks.Ciphertext, ker_fc, bias []float64, printResult bool) (ct_res *ckks.Ciphertext) {
	rs_ker := make([][]float64, 64)
	for i := 0; i < 64; i++ {
		rs_ker[i] = make([]float64, 16)
		for j := 0; j < 10; j++ {
			rs_ker[i][j] = ker_fc[j+i*10] / 64.0 // we will add 64 elts instead of averaging them
		}
	}

	// sum 64 elements instead of averaging them
	ct_avg := ct_input
	for i := 1; i < 64; i *= 2 {
		ct_avg = cont.evaluator.AddNew(ct_avg, cont.evaluator.RotateNew(ct_avg, i))
	}

	for i := 0; i < 16; i++ {
		tmp := make([]complex128, cont.N/2)
		for j := 0; j < 64; j++ {
			tmp[j*64*8] = complex(rs_ker[j][(j%16+16-i)%16], 0)
		}
		pl_ker := cont.encoder.EncodeNTTAtLvlNew(ct_avg.Level(), tmp, cont.logN-1)

		if i == 0 {
			ct_res = cont.evaluator.MulNew(ct_avg, pl_ker)
		} else {
			ct_tmp := cont.evaluator.MulNew(ct_avg, pl_ker)
			cont.evaluator.Add(ct_res, cont.evaluator.RotateNew(ct_tmp, i*64*8), ct_res)
		}
	}

	// final rotations to add up (4 = 64/16)
	for i := 1; i < 4; i *= 2 {
		ct_res = cont.evaluator.AddNew(ct_res, cont.evaluator.RotateNew(ct_res, i*16*64*8))
	}

	tmp := make([]complex128, cont.N/2)
	for j := 0; j < 10; j++ {
		tmp[j*64*8] = complex(bias[j], 0)
	}
	pl_bias := cont.encoder.EncodeNTTAtLvlNew(ct_res.Level(), tmp, cont.logN-1)
	cont.evaluator.Add(ct_res, pl_bias, ct_res)

	return
}

// reduce mean and final FC layer (in_batch -> 16)
// assume that ct_input has batch (1,0,0,0,2,0,0,0,3,0,0,0,4,0,0,0 || 0, ..., 0)
// ker_fc is of size in_batch*10 and 1-dim from [64,10] shape
func evalRMFC_BL_old(cont *context, ct_input *ckks.Ciphertext, ker_fc, bias []float64, printResult bool) (ct_res *ckks.Ciphertext) {
	rs_ker := make([][]float64, 64)
	for i := 0; i < 64; i++ {
		rs_ker[i] = make([]float64, 16)
		for j := 0; j < 10; j++ {
			rs_ker[i][j] = ker_fc[j+i*10] / 64.0 // we will add 64 elts instead of averaging them
		}
	}

	// sum 64 elements instead of averaging them
	ct_avg := ct_input
	for i := 1; i < 64; i *= 2 {
		ct_avg = cont.evaluator.AddNew(ct_avg, cont.evaluator.RotateNew(ct_avg, i))
	}

	fmt.Println()
	fmt.Println("===============  DECRYPTION  ===============")
	fmt.Println()
	start = time.Now()
	vals_tmp := cont.encoder.Decode(cont.decryptor.DecryptNew(ct_avg), cont.logN-1)
	fmt.Printf("Decryption Done in %s \n", time.Since(start))
	fmt.Print("Result: \n")
	prt_mat_BL(vals_tmp, 512, 0)

	for i := 0; i < 16; i++ {
		tmp := make([]complex128, cont.N/2)
		for j := 0; j < 64; j++ {
			tmp[j*64] = complex(rs_ker[j][(j%16+16-i)%16], 0)
		}
		pl_ker := cont.encoder.EncodeNTTAtLvlNew(ct_avg.Level(), tmp, cont.logN-1)

		if i == 0 {
			ct_res = cont.evaluator.MulNew(ct_avg, pl_ker)
		} else {
			ct_tmp := cont.evaluator.MulNew(ct_avg, pl_ker)
			cont.evaluator.Add(ct_res, cont.evaluator.RotateNew(ct_tmp, i*64), ct_res)
			cont.evaluator.Add(ct_res, cont.evaluator.RotateNew(ct_tmp, -(64-i)*64), ct_res)
		}
	}

	// final rotations to add up (4 = 64/16)
	for i := 1; i < 4; i *= 2 {
		ct_res = cont.evaluator.AddNew(ct_res, cont.evaluator.RotateNew(ct_res, i*16*64))
	}

	tmp := make([]complex128, cont.N/2)
	for j := 0; j < 10; j++ {
		tmp[j*64] = complex(bias[j], 0)
	}
	pl_bias := cont.encoder.EncodeNTTAtLvlNew(ct_res.Level(), tmp, cont.logN-1)
	cont.evaluator.Add(ct_res, pl_bias, ct_res)

	return
}

// Eval Conv only, always assume max batch
// in_wid must be Po2 (also include padding),
// include kernel preparation
// norm = 2 : in&out batches are (1,0,2,0,3,0,...)
func evalConv_BN(cont *context, ct_input *ckks.Ciphertext, ker_in, bn_a, bn_b []float64, in_wid, ker_wid, real_ib, real_ob, norm int, printResult, trans bool) (ct_res *ckks.Ciphertext) {
	max_batch := cont.N / (in_wid * in_wid)

	fmt.Println()
	fmt.Println("===============  (KER) PREPARATION  ===============")
	fmt.Println()
	start = time.Now()
	pl_ker := prep_Ker(cont.params, cont.encoder, ker_in, bn_a, in_wid, ker_wid, real_ib, real_ob, norm, cont.ECD_LV, 0, trans)
	b_coeffs := make([]float64, cont.N)
	for i := range bn_b {
		for j := 0; j < in_wid*in_wid; j++ {
			b_coeffs[norm*i+j*max_batch] = bn_b[i]
		}
	}
	scale_exp := cont.params.Scale() * cont.params.Scale() * float64(max_batch/norm)
	pl_bn_b := ckks.NewPlaintext(cont.params, cont.ECD_LV, scale_exp) // contain plaintext values
	cont.encoder.EncodeCoeffs(b_coeffs, pl_bn_b)
	cont.encoder.ToNTT(pl_bn_b)
	fmt.Printf("Plaintext (kernel) preparation, Done in %s \n", time.Since(start))

	fmt.Println()
	fmt.Println("===============  EVALUATION  ===============")
	fmt.Println()

	start = time.Now()
	ct_res = conv_then_pack(cont.params, cont.pack_evaluator, ct_input, pl_ker, cont.pl_idx, max_batch, norm, cont.ECD_LV, scale_exp)
	cont.evaluator.Add(ct_res, pl_bn_b, ct_res) // for Batch Normalization (BN)
	fmt.Printf("Conv (with BN) Done in %s \n", time.Since(start))

	return ct_res
}

// Eval Conv, BN, relu with Boot
// in_wid must be Po2, BN is fold with Kernel
// stride = true: apply [1,2,2,1] stride; false: [1,1,1,1]
// pack_pos: position to pack (0,1,2,3): only for strided case
// real_ib, real_ob: real number of batches (less or equal than max_batch)
func evalConv_BNRelu(cont *context, ct_input *ckks.Ciphertext, ker_in, bn_a, bn_b []float64, alpha float64, in_wid, ker_wid, real_ib, real_ob, norm, pack_pos int, padding, stride, printResult bool) (ct_res *ckks.Ciphertext) {
	trans := false
	kp_wid := in_wid - ((ker_wid - 1) / 2)
	ct_conv := evalConv_BN(cont, ct_input, ker_in, bn_a, bn_b, in_wid, ker_wid, real_ib, real_ob, norm, printResult, trans)
	ct_conv.Scale = ct_conv.Scale * math.Pow(2, pow)
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

	// Only for checking the correctness (for ReLU)
	relu1, relu2 := debugReLU(cont, slot1, slot2, alpha)
	relu1 = printDebug(cont.params, ct_boots[0], relu1, cont.decryptor, cont.encoder)
	relu2 = printDebug(cont.params, ct_boots[1], relu2, cont.decryptor, cont.encoder)
	var kind string
	if stride {
		kind = "StrConv"
	} else {
		kind = "Conv"
	}
	cfs_postB := debugStoC(cont, relu1, relu2, in_wid, kp_wid, kind)

	// needs to be modified for pack_pos consideration!!
	start = time.Now()
	ct_keep := make([]*ckks.Ciphertext, iter) // for extend (rotation) of ctxt_in
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
	printDebugCfs(cont.params, ct_res, cfs_postB, cont.decryptor, cont.encoder)

	return ct_res
}

// Eval Conv, BN, relu with Boot
// in_wid must be Po2 (also include padding)
// stride = true: apply [1,2,2,1] stride; false: [1,1,1,1]
// pack_pos: position to pack (0,1,2,3): only for strided case
// real_ib, real_ob: real number of batches (less or equal than max_batch)
func evalConv_BNRelu_new(cont *context, ct_input *ckks.Ciphertext, ker_in, bn_a, bn_b []float64, alpha float64, in_wid, kp_wid, ker_wid, real_ib, real_ob, norm, pack_pos int, kind string, printResult bool) (ct_res *ckks.Ciphertext) {
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

	ct_conv := evalConv_BN(cont, ct_input, ker_in, bn_a, bn_b, in_wid, ker_wid, real_ib, real_ob, norm, printResult, trans)
	ct_conv.Scale = ct_conv.Scale * math.Pow(2, pow)
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
		slot1[i] = complex(preB_cfs1[i], 0)
		slot2[i] = complex(preB_cfs2[i], 0)
		// slot1[i] = complex(preB_cfs1[i]/math.Pow(2, float64(pow)), 0)
		// slot2[i] = complex(preB_cfs2[i]/math.Pow(2, float64(pow)), 0)
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
