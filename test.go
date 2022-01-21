package main

import (
	"fmt"
	"math"
	"strconv"
	"time"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/rlwe"
)

func testDCGAN() {
	alpha := 0.3
	print := false
	logN := 16
	in_wids := []int{4, 8, 16, 32}
	kp_wids := []int{4, 8, 16, 32}
	batch := []int{512, 128, 64, 1}
	max_batch := []int{1024, 256, 64, 16}
	ker_wid := 5
	N := (1 << logN)
	padding := true // assume that each input is encoded with padding (e.g. 10//00)
	// st_batch := N / (2 * in_wid * 2 * in_wid) // We also consider zero-paddings  // must be adjusted when in_wid is not power of 2
	// end_batch := 4
	ECD_LV := 1

	// parameter generation
	cont := newContext(logN, ker_wid, in_wids, kp_wids, padding, "DCGAN")
	// cont := newContext(logN, ECD_LV, in_wids, padding, "DCGAN")

	for iter := 1; iter < 2; iter++ {
		name_iter := fmt.Sprintf("%04d", iter)

		input := readTxt("./DCGAN_inputs/input_"+name_iter+".txt", 0)
		ext_input := inputExt(input, logN, in_wids[0], false) // Takes arranged input (assume intermediate layers)  // print only outputs first (st_batch) batches

		start := time.Now()
		plain_in := ckks.NewPlaintext(cont.params, 1, cont.params.Scale()) // contain plaintext values
		cont.encoder.EncodeCoeffs(ext_input, plain_in)
		ctxt_in := cont.encryptor.EncryptNew(plain_in)

		// zeros := make([]complex128, params.Slots())
		// plain_in = encoder.EncodeNew(zeros, params.LogSlots())
		// ctxt0 := encryptor.EncryptNew(plain_in)

		fmt.Printf("Encryption: Done in %s \n", time.Since(start))

		if print {
			fmt.Println("vec size: ", N)
			fmt.Println("input width: ", in_wids)
			fmt.Println("kernel width: ", ker_wid)
			fmt.Println("num batches (in 1 ctxt with padding): ", max_batch[0])
		}

		ker1 := readTxt("./DCGAN_variables/conv1.txt", 0)
		a1 := readTxt("./DCGAN_variables/a1.txt", 0)
		b1 := readTxt("./DCGAN_variables/b1.txt", 0)
		b1_coeffs := make([]float64, N)
		for i := range b1 {
			for j := 0; j < in_wids[1]; j++ {
				for k := 0; k < in_wids[1]; k++ {
					b1_coeffs[i+(j+k*in_wids[1]*2)*max_batch[1]] = b1[i]
				}
			}
		}

		pl_ker := prepKer_in_trans(cont.params, cont.encoder, cont.encryptor, ker1, a1, in_wids[0], ker_wid, max_batch[0], max_batch[1], batch[0], batch[1], ECD_LV)

		fmt.Print("Boot in: ")
		fmt.Println()
		fmt.Println("Precision of values vs. ciphertext")
		in_cfs := printDebugCfs(cont.params, ctxt_in, ext_input, cont.decryptor, cont.encoder)

		fmt.Println("Bootstrapping... Ours (until CtoS):")
		start = time.Now()
		ctxt1, ctxt2, _ := cont.btp.BootstrappConv_CtoS(ctxt_in)
		fmt.Printf("Done in %s \n", time.Since(start))
		fmt.Println("after Boot: LV = ", ctxt1.Level(), " Scale = ", math.Log2(ctxt1.Scale))

		// Only for checking the correctness
		in_cfs_1_pBoot := make([]float64, cont.params.Slots())
		in_cfs_2_pBoot := make([]float64, cont.params.Slots())
		in_slots := make([]complex128, cont.params.Slots()) // first part of ceffs
		for i := range in_cfs_1_pBoot {
			in_cfs_1_pBoot[i] = in_cfs[reverseBits(uint32(i), cont.params.LogSlots())] // first part of coeffs
			in_cfs_2_pBoot[i] = in_cfs[reverseBits(uint32(i), cont.params.LogSlots())+uint32(cont.params.Slots())]
			in_slots[i] = complex(in_cfs_1_pBoot[i]/math.Pow(2, float64(pow)), 0)
		}
		ext1_tmp := extend_full_nhf(in_cfs_1_pBoot, in_wids[1], 3, true, true)
		ext2_tmp := extend_full_nhf(in_cfs_2_pBoot, in_wids[1], 3, true, true)
		for i := range in_cfs_1_pBoot {
			in_cfs_1_pBoot[i] = ext1_tmp[reverseBits(uint32(i), cont.params.LogSlots())]
			in_cfs_2_pBoot[i] = ext2_tmp[reverseBits(uint32(i), cont.params.LogSlots())]
		}
		in_cfs_pBoot := append(in_cfs_1_pBoot, in_cfs_2_pBoot...) // After rot(ext) and boot

		in_slots = printDebug(cont.params, ctxt1, in_slots, cont.decryptor, cont.encoder) // Compare before & after CtoS

		start = time.Now()
		// evaluator.MultByConst(ctxt1, 1.000000001, ctxt1)
		// evaluator.DropLevel(ctxt1, 10)
		cont.evaluator.MulByPow2(ctxt1, pow, ctxt1)
		cont.evaluator.DropLevel(ctxt1, ctxt1.Level()-3)

		// ctxt1 = evalReLU(cont.params, cont.evaluator, ctxt1, 1.0)
		fmt.Printf("NO ReLU Done in %s \n", time.Since(start))

		values_ReLU := make([]complex128, len(in_slots))
		for i := range values_ReLU {
			values_ReLU[i] = complex(math.Pow(2, float64(pow)), 0) * in_slots[i] // complex(math.Max(0, real(in_slots[i])), 0)
		}
		printDebug(cont.params, ctxt1, values_ReLU, cont.decryptor, cont.encoder)

		ext_ctxt1 := make([]*ckks.Ciphertext, 4) // for extend (rotation) of ctxt_in
		// ext_ctxt2 := make([]*ckks.Ciphertext, 4)  // do not need if we use po2 inputs dims
		ciphertext := make([]*ckks.Ciphertext, 4) // after Bootstrapping

		ctxt2 = nil
		// cont.evaluator.DropLevel(ctxt2, ctxt2.Level()-2)
		start = time.Now()
		for pos := 0; pos < 4; pos++ {
			ext_ctxt1[pos] = ext_ctxt(cont.evaluator, cont.encoder, ctxt1, cont.r_idx[in_wids[1]][pos], cont.params)
			// fmt.Println(ext_ctxt1[pos].Level(), ctxt2.Level(), ext_ctxt1[pos].Scale, ctxt2.Scale)
			// ext_ctxt2[pos] = ext_ctxt(cont.evaluator, cont.encoder, ctxt2, r_idx[pos], m_idx[pos], cont.params)
			ciphertext[pos] = cont.btp.BootstrappConv_StoC(ext_ctxt1[pos], ctxt2)
			cont.evaluator.Rescale(ciphertext[pos], cont.params.Scale(), ciphertext[pos])
		}
		fmt.Printf("Boot (StoC) Done in %s \n", time.Since(start))

		fmt.Printf("Boot out: ")
		// for i := range in_cfs_pBoot {
		// 	in_cfs_pBoot[i] = math.Max(0, in_cfs_pBoot[i])
		// }
		printDebugCfs(cont.params, ciphertext[3], in_cfs_pBoot, cont.decryptor, cont.encoder)

		ctxt_result := conv_then_pack_trans(cont.params, cont.pack_evaluator, ciphertext, pl_ker, cont.pl_idx, max_batch[1])

		// for Batch Normalization (BN)
		plain_in = ckks.NewPlaintext(cont.params, ctxt_result.Level(), ctxt_result.Scale) // contain plaintext values
		cont.encoder.EncodeCoeffs(b1_coeffs, plain_in)
		cont.encoder.ToNTT(plain_in)
		cont.evaluator.Add(ctxt_result, plain_in, ctxt_result)

		plain_out := ckks.NewPlaintext(cont.params, ctxt_result.Level(), cont.params.Scale())
		start = time.Now()
		cont.decryptor.Decrypt(ctxt_result, plain_out)
		pre_boot := cont.encoder.DecodeCoeffs(plain_out)
		cfs_tmp := reshape_conv_out(cont.encoder.DecodeCoeffs(plain_out), in_wids[1], max_batch[1])

		if print {
			fmt.Println()
			fmt.Println("=========================================")
			fmt.Println("              DECRYPTION                 ")
			fmt.Println("=========================================")
			fmt.Println()

			fmt.Print("Result: \n")
			prt_mat(cfs_tmp, max_batch[1], in_wids[1])
		}
		fmt.Printf("(Layer 1) Done in %s \n", time.Since(start))

		// // To see each matrix
		// cfs_tmp = cont.encoder.DecodeCoeffs(plain_out)
		// int_tmpn := make([]int, N)
		// for i := range cfs_tmp {
		// 	int_tmpn[i] = int(cfs_tmp[i])
		// }
		// fmt.Print("Output: \n")
		// for b := 0; b < batch[1]; b++ {
		// 	print_vec("output ("+strconv.Itoa(b)+")", int_tmpn, in_wid[2], b)
		// }

		// // Layer 1 done

		fmt.Println()
		fmt.Println("=========================================")
		fmt.Println("              LAYER 2	                 ")
		fmt.Println("=========================================")
		fmt.Println()

		ker2 := readTxt("./DCGAN_variables/conv2.txt", 0)
		a2 := readTxt("./DCGAN_variables/a2.txt", 0)
		b2 := readTxt("./DCGAN_variables/b2.txt", 0)
		b2_coeffs := make([]float64, N)
		for i := range b2 {
			for j := 0; j < in_wids[2]; j++ {
				for k := 0; k < in_wids[2]; k++ {
					b2_coeffs[i+(j+k*in_wids[2]*2)*max_batch[2]] = b2[i]
				}
			}
		}

		pl_ker = prepKer_in_trans(cont.params, cont.encoder, cont.encryptor, ker2, a2, in_wids[1], ker_wid, max_batch[1], max_batch[2], batch[1], batch[2], ECD_LV)

		// fmt.Print("Boot in: ")
		// fmt.Println()
		// fmt.Println("Precision of values vs. ciphertext")
		// in_cfs = printDebugCfs(cont.params, ctxt_result, pre_boot, cont.decryptor, cont.encoder)
		in_cfs = pre_boot
		ctxt_in.Copy(ctxt_result)
		// ctxt_in.SetScalingFactor(ctxt_in.Scale * 64)

		fmt.Println("Bootstrapping... Ours (until CtoS):")
		start = time.Now()
		ctxt1, _, _ = cont.btp.BootstrappConv_CtoS(ctxt_in)
		fmt.Printf("Done in %s \n", time.Since(start))

		// Only for checking the correctness
		for i := range in_cfs_1_pBoot {
			in_cfs_1_pBoot[i] = in_cfs[reverseBits(uint32(i), cont.params.LogSlots())] // first part of coeffs
			in_cfs_2_pBoot[i] = 0                                                      // in_cfs[reverseBits(uint32(i), cont.params.LogSlots())+uint32(cont.params.Slots())]
			in_slots[i] = complex(in_cfs_1_pBoot[i]/math.Pow(2, float64(pow)), 0)
		}
		ext1_tmp = extend_full_nhf(in_cfs_1_pBoot, in_wids[2], 0, true, true)
		ext2_tmp = extend_full_nhf(in_cfs_2_pBoot, in_wids[2], 0, true, true)
		for i := range in_cfs_1_pBoot {
			in_cfs_1_pBoot[i] = ext1_tmp[reverseBits(uint32(i), cont.params.LogSlots())]
			in_cfs_2_pBoot[i] = ext2_tmp[reverseBits(uint32(i), cont.params.LogSlots())]
		}
		in_cfs_pBoot = append(in_cfs_1_pBoot, in_cfs_2_pBoot...) // After rot(ext) and boot
		in_slots = printDebug(cont.params, ctxt1, in_slots, cont.decryptor, cont.encoder)

		start = time.Now()

		plain_ch := ckks.NewPlaintext(cont.params, ctxt1.Level(), cont.params.Scale())
		cont.decryptor.Decrypt(ctxt1, plain_ch)
		check := cont.encoder.Decode(plain_ch, logN-1)
		max := 0.0
		avg := 0.0
		for _, val := range check {
			rval := real(val)
			if math.Abs(rval) > math.Abs(max) {
				max = rval
			}
			avg += rval
		}
		avg = 2 * avg / float64(N)
		fmt.Println("max valu: ", max)
		fmt.Println("avg valu: ", avg)

		// evaluator.MulByPow2(ctxt1, pow, ctxt1)
		// evaluator.DropLevel(ctxt1, ctxt1.Level()-3)

		ctxt1 = evalReLU(cont.params, cont.evaluator, ctxt1, alpha)
		cont.evaluator.MulByPow2(ctxt1, pow, ctxt1)
		fmt.Printf("ReLU Done in %s \n", time.Since(start))

		for i := range values_ReLU {
			values_ReLU[i] = complex(math.Pow(2, float64(pow)), 0) * complex(math.Max(0, real(in_slots[i]))+alpha*math.Min(0, real(in_slots[i])), 0)
		}
		printDebug(cont.params, ctxt1, values_ReLU, cont.decryptor, cont.encoder)

		start = time.Now()
		for pos := 0; pos < 4; pos++ {
			ext_ctxt1[pos] = ext_ctxt(cont.evaluator, cont.encoder, ctxt1, cont.r_idx[in_wids[2]][pos], cont.params)
			// ext_ctxt2[pos] = ext_ctxt(evaluator, cont.encoder, ctxt2, r_idx1[pos], m_idx1[pos], cont.params)
			ciphertext[pos] = cont.btp.BootstrappConv_StoC(ext_ctxt1[pos], ctxt2)
			cont.evaluator.Rescale(ciphertext[pos], cont.params.Scale(), ciphertext[pos])
		}
		fmt.Printf("Boot (StoC) Done in %s \n", time.Since(start))

		fmt.Printf("Boot out: ")
		for i := range in_cfs_pBoot {
			in_cfs_pBoot[i] = math.Max(0, in_cfs_pBoot[i]) + alpha*math.Min(0, in_cfs_pBoot[i])
		}
		printDebugCfs(cont.params, ciphertext[0], in_cfs_pBoot, cont.decryptor, cont.encoder)

		ctxt_result = conv_then_pack_trans(cont.params, cont.pack_evaluator, ciphertext, pl_ker, cont.pl_idx, max_batch[2])

		// for BN
		plain_in = ckks.NewPlaintext(cont.params, ctxt_result.Level(), ctxt_result.Scale) // contain plaintext values
		cont.encoder.EncodeCoeffs(b2_coeffs, plain_in)
		cont.encoder.ToNTT(plain_in)
		cont.evaluator.Add(ctxt_result, plain_in, ctxt_result)

		start = time.Now()
		cont.decryptor.Decrypt(ctxt_result, plain_out)
		pre_boot = cont.encoder.DecodeCoeffs(plain_out)
		cfs_tmp = reshape_conv_out(cont.encoder.DecodeCoeffs(plain_out), in_wids[2], max_batch[2])

		if print {
			fmt.Println()
			fmt.Println("=========================================")
			fmt.Println("              DECRYPTION                 ")
			fmt.Println("=========================================")
			fmt.Println()
			fmt.Print("Result: \n")
			prt_mat(cfs_tmp, max_batch[2], in_wids[2])
		}
		fmt.Printf("(Layer 2) Done in %s \n", time.Since(start))

		fmt.Println()
		fmt.Println("=========================================")
		fmt.Println("              LAYER 3	                 ")
		fmt.Println("=========================================")
		fmt.Println()

		ker3 := readTxt("./DCGAN_variables/conv3.txt", 0)
		a3 := make([]float64, batch[3])
		for i := range a3 {
			a3[i] = 1
		}
		pl_ker = prepKer_in_trans(cont.params, cont.encoder, cont.encryptor, ker3, a3, in_wids[2], ker_wid, max_batch[2], max_batch[3], batch[2], batch[3], ECD_LV)

		// fmt.Print("Boot in: ")
		// fmt.Println()
		// fmt.Println("Precision of values vs. ciphertext")
		// in_cfs = printDebugCfs(cont.params, ctxt_result, pre_boot, cont.decryptor, cont.encoder)
		in_cfs = pre_boot
		ctxt_in.Copy(ctxt_result)
		// ctxt_in.SetScalingFactor(ctxt_in.Scale * 16)

		fmt.Println("Bootstrapping... Ours (until CtoS):")
		start = time.Now()
		ctxt1, _, _ = cont.btp.BootstrappConv_CtoS(ctxt_in)
		fmt.Printf("Done in %s \n", time.Since(start))

		// Only for checking the correctness
		for i := range in_cfs_1_pBoot {
			in_cfs_1_pBoot[i] = in_cfs[reverseBits(uint32(i), cont.params.LogSlots())] // first part of coeffs
			in_cfs_2_pBoot[i] = 0                                                      // in_cfs[reverseBits(uint32(i), cont.params.LogSlots())+uint32(cont.params.Slots())]
			in_slots[i] = complex(in_cfs_1_pBoot[i]/math.Pow(2, float64(pow)), 0)
		}
		ext1_tmp = extend_full_nhf(in_cfs_1_pBoot, in_wids[3], 0, true, true)
		ext2_tmp = extend_full_nhf(in_cfs_2_pBoot, in_wids[3], 0, true, true)
		for i := range in_cfs_1_pBoot {
			in_cfs_1_pBoot[i] = ext1_tmp[reverseBits(uint32(i), cont.params.LogSlots())]
			in_cfs_2_pBoot[i] = ext2_tmp[reverseBits(uint32(i), cont.params.LogSlots())]
		}
		in_cfs_pBoot = append(in_cfs_1_pBoot, in_cfs_2_pBoot...) // After rot(ext) and boot
		in_slots = printDebug(cont.params, ctxt1, in_slots, cont.decryptor, cont.encoder)

		start = time.Now()
		// evaluator.MultByConst(ctxt1, 1.000000001, ctxt1)
		// evaluator.DropLevel(ctxt1, 10)

		plain_ch = ckks.NewPlaintext(cont.params, ctxt1.Level(), cont.params.Scale())
		cont.decryptor.Decrypt(ctxt1, plain_ch)
		check = cont.encoder.Decode(plain_ch, logN-1)
		max = 0.0
		avg = 0.0
		for _, val := range check {
			rval := real(val)
			if math.Abs(rval) > math.Abs(max) {
				max = rval
			}
			avg += rval
		}
		avg = 2 * avg / float64(N)
		fmt.Println("max valu: ", max)
		fmt.Println("avg valu: ", avg)

		ctxt1 = evalReLU(cont.params, cont.evaluator, ctxt1, alpha)
		cont.evaluator.MulByPow2(ctxt1, pow, ctxt1)
		fmt.Printf("ReLU Done in %s \n", time.Since(start))

		for i := range values_ReLU {
			values_ReLU[i] = complex(math.Pow(2, float64(pow)), 0) * complex(math.Max(0, real(in_slots[i]))+alpha*math.Min(0, real(in_slots[i])), 0)
		}
		printDebug(cont.params, ctxt1, values_ReLU, cont.decryptor, cont.encoder)

		start = time.Now()
		for pos := 0; pos < 4; pos++ {
			ext_ctxt1[pos] = ext_ctxt(cont.evaluator, cont.encoder, ctxt1, cont.r_idx[in_wids[3]][pos], cont.params)
			// ext_ctxt2[pos] = ext_ctxt(evaluator, cont.encoder, ctxt2, r_idx2[pos], m_idx2[pos], cont.params)
			ciphertext[pos] = cont.btp.BootstrappConv_StoC(ext_ctxt1[pos], ctxt2)
			cont.evaluator.Rescale(ciphertext[pos], cont.params.Scale(), ciphertext[pos])
		}
		fmt.Printf("Boot (StoC) Done in %s \n", time.Since(start))

		fmt.Printf("Boot out: ")
		for i := range in_cfs_pBoot {
			in_cfs_pBoot[i] = math.Max(0, in_cfs_pBoot[i]) + alpha*math.Min(0, in_cfs_pBoot[i])
		}
		printDebugCfs(cont.params, ciphertext[0], in_cfs_pBoot, cont.decryptor, cont.encoder)

		ctxt_result = conv_then_pack_trans(cont.params, cont.pack_evaluator, ciphertext, pl_ker, cont.pl_idx, max_batch[3])
		// ctxt_result = conv_then_pack(cont.params, pack_evaluator, ciphertext, pl_ker, plain_idx, batch[2])

		start = time.Now()
		cont.decryptor.Decrypt(ctxt_result, plain_out)
		pre_boot = cont.encoder.DecodeCoeffs(plain_out)
		cfs_tmp = reshape_conv_out(cont.encoder.DecodeCoeffs(plain_out), in_wids[3], max_batch[3])

		if print {
			fmt.Println()
			fmt.Println("=========================================")
			fmt.Println("              DECRYPTION                 ")
			fmt.Println("=========================================")
			fmt.Println()
			fmt.Print("Result: \n")
			prt_mat(cfs_tmp, max_batch[3], in_wids[3])
		}
		fmt.Printf("(Layer 3) Done in %s \n", time.Since(start))

		output := make([]float64, in_wids[3]*in_wids[3])
		for i := range output {
			output[i] = cfs_tmp[max_batch[3]*i]
		}
		writeTxt("./DCGAN_result/result_"+name_iter+".txt", output)
	}
}

// Fast Conv without boot, Assume full batch with Po2 in_wid & N
// Normal Conv without output modification (e.g., trimming or expanding)
// Assume that the input is 0 padded according to kernel size: only in_wid - (ker_wid-1)/2 elements in row and columns are nonzero
// Also support non-full batching case
func testConv_noBoot(kind string, printResult bool) (ct_result *ckks.Ciphertext) {
	raw_in_batch := 8 // same as python
	raw_in_wid := 15  // same as python
	in_batch := 8     // needs to be divided by 4 (to pack the output of transConv)
	in_wid := 16
	norm := in_batch / raw_in_batch
	ker_wid := 3

	// set basic variables for above input variables
	kp_wid, out_batch, logN, trans := set_Variables(in_batch, raw_in_wid, in_wid, ker_wid, kind)
	raw_out_batch := out_batch / norm

	raw_input := make([]float64, raw_in_wid*raw_in_wid*raw_in_batch)
	ker_in := make([]float64, raw_in_batch*raw_out_batch*ker_wid*ker_wid)
	bn_a := make([]float64, raw_out_batch)
	bn_b := make([]float64, raw_out_batch)
	for i := range raw_input {
		raw_input[i] = 1.0 * float64(i) / float64(len(raw_input))
	}
	for i := range ker_in {
		ker_in[i] = 1.0 * float64(i) / float64(len(ker_in))
	}
	for i := range bn_a {
		bn_a[i] = 1.0
		bn_b[i] = 0.0
	}

	// generate Context: params, Keys, rotations, general plaintexts
	cont := newContext(logN, ker_wid, []int{in_wid}, []int{kp_wid}, false, kind)
	fmt.Println("vec size: log2 = ", cont.logN)
	fmt.Println("raw input width: ", raw_in_wid)
	fmt.Println("kernel width: ", ker_wid)
	fmt.Println("num raw batches in & out: ", raw_in_batch, ", ", raw_out_batch)

	// input encryption
	fmt.Println()
	fmt.Println("===============  ENCRYPTION  ===============")
	fmt.Println()
	input := prep_Input(raw_input, raw_in_wid, in_wid, cont.N, norm, trans, printResult)
	start = time.Now()
	plain_tmp := ckks.NewPlaintext(cont.params, cont.ECD_LV, cont.params.Scale()) // contain plaintext values
	cont.encoder.EncodeCoeffs(input, plain_tmp)
	ctxt_input := cont.encryptor.EncryptNew(plain_tmp)
	fmt.Printf("Encryption done in %s \n", time.Since(start))

	// Kernel Prep & Conv (+BN) Evaluation
	ct_result = evalConv_BN(cont, ctxt_input, ker_in, bn_a, bn_b, in_wid, ker_wid, raw_in_batch, raw_out_batch, norm, float64(1<<30), printResult, trans)

	fmt.Println()
	fmt.Println("===============  DECRYPTION  ===============")
	fmt.Println()
	start = time.Now()
	// cont.new_decryptor.Decrypt(ct_result, plain_tmp)
	cont.decryptor.Decrypt(ct_result, plain_tmp)
	cfs_tmp := cont.encoder.DecodeCoeffs(plain_tmp)
	fmt.Printf("Decryption Done in %s \n", time.Since(start))

	if printResult {
		fmt.Print("Result: \n")
		prt_mat(cfs_tmp, out_batch, 3)

		for b := 0; b < in_batch; b++ {
			print_vec("input ("+strconv.Itoa(b)+")", input, in_wid, b)
		}
		for b := 0; b < out_batch; b++ {
			print_vec("output ("+strconv.Itoa(b)+")", cfs_tmp, in_wid, b)
		}
	}

	return ct_result
}

// Fast Conv without boot, Assume full batch with Po2 in_wid & N
// Normal Conv without output modification (e.g., trimming or expanding)
// Assume that the input is 0 padded according to kernel size: only in_wid - (ker_wid-1)/2 elements in row and columns are nonzero
func testConv_noBoot_old(kind string, printResult bool) (ct_result *ckks.Ciphertext) {
	in_batch := 8   // needs to be divided by 4 (to pack the output of transConv)
	raw_in_wid := 4 // same as python
	in_wid := 8
	ker_wid := 3
	norm := 1

	// set basic variables for above input variables
	kp_wid, out_batch, logN, trans := set_Variables(in_batch, raw_in_wid, in_wid, ker_wid, kind)

	raw_input := make([]float64, raw_in_wid*raw_in_wid*in_batch)
	ker_in := make([]float64, in_batch*out_batch*ker_wid*ker_wid)
	bn_a := make([]float64, out_batch)
	bn_b := make([]float64, out_batch)
	for i := range raw_input {
		raw_input[i] = 1.0 * float64(i) / float64(len(raw_input))
	}
	for i := range ker_in {
		ker_in[i] = 1.0 - 1.0*float64(i)/float64(len(ker_in))
	}
	for i := range bn_a {
		bn_a[i] = 1.0
		bn_b[i] = 0.0
	}

	// generate Context: params, Keys, rotations, general plaintexts
	cont := newContext(logN, ker_wid, []int{in_wid}, []int{kp_wid}, false, kind)
	fmt.Println("vec size: log2 = ", cont.logN)
	fmt.Println("raw input width: ", raw_in_wid)
	fmt.Println("kernel width: ", ker_wid)
	fmt.Println("num batches in & out: ", in_batch, ", ", out_batch)

	// input encryption
	fmt.Println()
	fmt.Println("===============  ENCRYPTION  ===============")
	fmt.Println()
	input := prep_Input(raw_input, raw_in_wid, in_wid, cont.N, norm, trans, printResult)
	start = time.Now()
	plain_tmp := ckks.NewPlaintext(cont.params, cont.ECD_LV, cont.params.Scale()) // contain plaintext values
	cont.encoder.EncodeCoeffs(input, plain_tmp)
	ctxt_input := cont.encryptor.EncryptNew(plain_tmp)
	fmt.Printf("Encryption done in %s \n", time.Since(start))

	// Kernel Prep & Conv (+BN) Evaluation
	ct_result = evalConv_BN(cont, ctxt_input, ker_in, bn_a, bn_b, in_wid, ker_wid, in_batch, out_batch, norm, float64(1<<30), printResult, trans)

	fmt.Println()
	fmt.Println("===============  DECRYPTION  ===============")
	fmt.Println()
	start = time.Now()
	cont.decryptor.Decrypt(ct_result, plain_tmp)
	cfs_tmp := cont.encoder.DecodeCoeffs(plain_tmp)
	fmt.Printf("Decryption Done in %s \n", time.Since(start))

	if printResult {
		fmt.Print("Result: \n")
		prt_mat(cfs_tmp, out_batch, 3)

		for b := 0; b < in_batch; b++ {
			print_vec("input ("+strconv.Itoa(b)+")", input, in_wid, b)
		}
		for b := 0; b < out_batch; b++ {
			print_vec("output ("+strconv.Itoa(b)+")", cfs_tmp, in_wid, b)
		}
	}

	return ct_result
}

// Eval Conv, BN, relu with Boot
// always set pad := (ker_wid-1)/2 to simulate full packing
// in_wid must be Po2
// For trans, assume that input: full bached ciphertext, outputs 1/4 batched 1ctxt due to expansion
func testConv_BNRelu(kind string, printResult bool) {
	raw_in_batch := 256 // same as python
	raw_in_wid := 14    // same as python
	in_batch := 256     // needs to be divided by 4 (to pack the output of transConv)
	in_wid := 16
	norm := in_batch / raw_in_batch
	ker_wid := 5
	alpha := 0.0 // for ReLU: 0.0 , leakyReLU : 0.3
	pack_pos := 0

	// set basic variables for above input variables
	kp_wid, out_batch, logN, trans := set_Variables(in_batch, raw_in_wid, in_wid, ker_wid, kind)
	raw_out_batch := out_batch / norm
	// fmt.Println(raw_out_batch)

	raw_input := make([]float64, raw_in_wid*raw_in_wid*raw_in_batch)
	ker_in := make([]float64, raw_in_batch*raw_out_batch*ker_wid*ker_wid)
	bn_a := make([]float64, raw_out_batch)
	bn_b := make([]float64, raw_out_batch)
	for i := range raw_input {
		raw_input[i] = 0.1 * float64(i) / float64(len(raw_input))
	}
	for i := range ker_in {
		ker_in[i] = 0.5 * float64(i) / float64(len(ker_in))
	}
	for i := range bn_a {
		bn_a[i] = 1.0
		bn_b[i] = 0.0
	}

	// generate Context: params, Keys, rotations, general plaintexts
	cont := newContext(logN, ker_wid, []int{in_wid}, []int{kp_wid}, true, kind)
	fmt.Println("vec size: log2 = ", cont.logN)
	fmt.Println("raw input width: ", raw_in_wid)
	fmt.Println("kernel width: ", ker_wid)
	fmt.Println("num batches in & out: ", raw_in_batch, ", ", raw_out_batch)

	// input encryption
	fmt.Println()
	fmt.Println("===============  ENCRYPTION  ===============")
	fmt.Println()
	input := prep_Input(raw_input, raw_in_wid, in_wid, cont.N, norm, trans, printResult)
	start = time.Now()
	plain_tmp := ckks.NewPlaintext(cont.params, cont.ECD_LV, cont.params.Scale()) // contain plaintext values
	cont.encoder.EncodeCoeffs(input, plain_tmp)
	ctxt_input := cont.encryptor.EncryptNew(plain_tmp)
	fmt.Printf("Encryption done in %s \n", time.Since(start))

	// Kernel Prep & Conv (+BN) Evaluation

	fast_pack := false
	if kind == "StrConv_fast" {
		fast_pack = true
	}
	ct_result := evalConv_BNRelu_new(cont, ctxt_input, ker_in, bn_a, bn_b, alpha, in_wid, kp_wid, ker_wid, raw_in_batch, raw_out_batch, norm, pack_pos, 2, kind, fast_pack, printResult)

	fmt.Println()
	fmt.Println("===============  DECRYPTION  ===============")
	fmt.Println()
	start = time.Now()
	cfs_tmp := cont.encoder.DecodeCoeffs(cont.decryptor.DecryptNew(ct_result))
	fmt.Printf("Decryption Done in %s \n", time.Since(start))

	if printResult {
		fmt.Print("Result: \n")
		prt_mat(cfs_tmp, out_batch, 0)

		for b := 0; b < in_batch; b++ {
			print_vec("input ("+strconv.Itoa(b)+")", input, in_wid, b)
		}
		if kind == "StrConv" {
			out_batch = 4 * out_batch
		}
		for b := 0; b < out_batch; b++ {
			// needs to be adjusted for conv/ strided/ trans
			switch kind {
			case "Conv":
				print_vec("output ("+strconv.Itoa(b)+")", cfs_tmp, in_wid, b)
			case "StrConv":
				print_vec("output ("+strconv.Itoa(b)+")", cfs_tmp, in_wid/2, b)
			case "TransConv":
				print_vec("output ("+strconv.Itoa(b)+")", cfs_tmp, in_wid*2, b)
			}
		}
	}
}

func testResNet_in(st, end int) {
	// For ResNet, we use padding: i.e., in_wid**2 element is contained in (2*in_wid)**2 sized block
	// So ReLU, keep or rot, StoC done only on the 1st part of the CtoS ciphertexts
	weight_dir := "weight_ker7_h5/"
	ker_name := "ker7"
	logN := 16
	num_blc1 := 3                   // 3 // 5 // 7
	num_blc2 := 1                   // 1 // 3 // 5
	num_blc3 := 1                   // 1 // 3 // 5
	raw_in_wids := []int{32, 16, 8} // same as python
	real_batch := []int{16, 32, 64} // same as python
	ker_wid := 7
	padding := true
	fast_pack := true
	in_wids := make([]int, len(raw_in_wids))
	kp_wids := make([]int, len(raw_in_wids))
	for i, elt := range raw_in_wids {
		if padding {
			in_wids[i] = 2 * elt
			kp_wids[i] = elt
		} else {
			in_wids[i] = elt
		}
	}
	cont := newContext(logN, ker_wid, in_wids, kp_wids, true, "Resnet")
	ker_size := ker_wid * ker_wid
	max_batch := make([]int, len(real_batch)) // the max batch
	for i := range max_batch {
		max_batch[i] = cont.N / (in_wids[i] * in_wids[i])
	}
	alpha := 0.0 // 0.3 => leakyrelu

	for iter := st; iter < end; iter++ {
		norm := 1
		fmt.Println("Running ", iter, "-th iter... ker size: ", ker_wid)
		image := readTxt("test_data/test_image_"+strconv.Itoa(iter)+".csv", 32*32*3)
		input := make([]float64, cont.N)
		k := 0
		for i := 0; i < in_wids[0]; i++ {
			for j := 0; j < in_wids[0]; j++ {
				for b := 0; b < 3; b++ {
					if (i < raw_in_wids[0]) && (j < raw_in_wids[0]) {
						input[i*in_wids[0]*max_batch[0]+j*max_batch[0]+b] = image[k]
						k++
					}
				}
			}
		}
		fmt.Println("Input: ")
		prt_mat(input, max_batch[0], 3)
		fmt.Println("vec size: ", cont.N)
		fmt.Println("input width: ", raw_in_wids)
		fmt.Println("kernel width: ", ker_wid)
		fmt.Println("num batches: ", real_batch)
		fmt.Println("Input matrix: ")
		prt_vec(input)

		start = time.Now()
		pl_input := ckks.NewPlaintext(cont.params, cont.ECD_LV, cont.params.Scale()) // contain plaintext values
		cont.encoder.EncodeCoeffs(input, pl_input)
		ct_input := cont.encryptor.EncryptNew(pl_input)
		fmt.Printf("Encryption done in %s \n", time.Since(start))

		timings := make([]float64, 6)
		begin_start := time.Now()
		new_start := time.Now()

		// ResNet Block 1
		ct_layer := ct_input
		prt_result := false
		for i := 1; i <= num_blc1; i++ {
			if i == num_blc1 {
				prt_result = true
			}
			bn_a := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-a.csv", real_batch[0])
			bn_b := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-b.csv", real_batch[0])
			if i == 1 {
				ker_in := readTxt(weight_dir+"w0-conv.csv", 3*real_batch[0]*ker_size)
				ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in, bn_a, bn_b, alpha, in_wids[0], kp_wids[0], ker_wid, 3, real_batch[0], norm, 0, 1, "Conv", fast_pack, prt_result)
			} else {
				ker_in := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-conv.csv", real_batch[0]*real_batch[0]*ker_size)
				ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in, bn_a, bn_b, alpha, in_wids[0], kp_wids[0], ker_wid, real_batch[0], real_batch[0], norm, 0, 1, "Conv", fast_pack, prt_result)
			}
			fmt.Println("Block1, Layer ", i, "done!")
		}
		fmt.Println("done.")
		timings[0] = time.Since(new_start).Seconds()
		new_start = time.Now()
		ker_in12 := readTxt(weight_dir+"w"+strconv.Itoa(num_blc1)+"-conv.csv", real_batch[0]*real_batch[1]*ker_size)
		ker_in12_0 := make([]float64, len(ker_in12)/2)
		ker_in12_1 := make([]float64, len(ker_in12)/2)
		for k := 0; k < ker_size; k++ {
			for i := 0; i < real_batch[0]; i++ {
				for j := 0; j < real_batch[1]/2; j++ {
					ker_in12_0[k*real_batch[0]*real_batch[1]/2+(i*real_batch[1]/2+j)] = ker_in12[k*real_batch[0]*real_batch[1]+(i*real_batch[1]+2*j)]   // [i][2*j]
					ker_in12_1[k*real_batch[0]*real_batch[1]/2+(i*real_batch[1]/2+j)] = ker_in12[k*real_batch[0]*real_batch[1]+(i*real_batch[1]+2*j+1)] // [i][2*j+1]
				}
			}
		}
		bn_a := readTxt(weight_dir+"w"+strconv.Itoa(num_blc1)+"-a.csv", real_batch[1])
		bn_b := readTxt(weight_dir+"w"+strconv.Itoa(num_blc1)+"-b.csv", real_batch[1])
		bn_a2_0 := make([]float64, real_batch[1]/2)
		bn_a2_1 := make([]float64, real_batch[1]/2)
		bn_b2_0 := make([]float64, real_batch[1]/2)
		bn_b2_1 := make([]float64, real_batch[1]/2)
		for i := range bn_b2_0 {
			bn_a2_0[i] = bn_a[2*i]
			bn_a2_1[i] = bn_a[2*i+1]
			bn_b2_0[i] = bn_b[2*i]
			bn_b2_1[i] = bn_b[2*i+1]
		}
		ct_result1 := evalConv_BNRelu_new(cont, ct_layer, ker_in12_0, bn_a2_0, bn_b2_0, alpha, in_wids[0], kp_wids[0], ker_wid, real_batch[0], real_batch[1]/2, norm, 0, 1, "StrConv", fast_pack, false)
		ct_result2 := evalConv_BNRelu_new(cont, ct_layer, ker_in12_1, bn_a2_1, bn_b2_1, alpha, in_wids[0], kp_wids[0], ker_wid, real_batch[0], real_batch[1]/2, norm, 2, 1, "StrConv", fast_pack, false)
		ct_result := cont.evaluator.AddNew(ct_result1, ct_result2)
		fmt.Println("Block1 to 2 done!")
		max_bat := cont.N / (in_wids[1] * in_wids[1])
		res_ttmp := cont.encoder.DecodeCoeffs(cont.decryptor.DecryptNew(ct_result))
		prt_mat_norm(res_ttmp, max_bat, 2, 3, true)

		timings[1] = time.Since(new_start).Seconds()
		new_start = time.Now()

		// ResNet Block 2
		ct_layer = ct_result
		prt_result = false
		norm = 2
		for i := 1; i <= num_blc2; i++ {
			if i == num_blc2 {
				prt_result = true
			}
			bn_a2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blc1+i)+"-a.csv", real_batch[1])
			bn_b2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blc1+i)+"-b.csv", real_batch[1])
			ker_in2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blc1+i)+"-conv.csv", real_batch[1]*real_batch[1]*ker_size)

			ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in2, bn_a2, bn_b2, alpha, in_wids[1], kp_wids[1], ker_wid, real_batch[1], real_batch[1], norm, 0, 1, "Conv", fast_pack, prt_result)
			fmt.Println("Block2, Layer ", i, "done!")
		}
		timings[2] = time.Since(new_start).Seconds()
		new_start = time.Now()
		ker_in23 := readTxt(weight_dir+"w"+strconv.Itoa(num_blc1+num_blc2+1)+"-conv.csv", real_batch[1]*real_batch[2]*ker_size)
		bn_a3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blc1+num_blc2+1)+"-a.csv", real_batch[2])
		bn_b3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blc1+num_blc2+1)+"-b.csv", real_batch[2])
		ker_in23_new := make([]float64, 2*real_batch[1]*real_batch[2]*ker_size)
		for k := 0; k < ker_size; k++ {
			for i := 0; i < real_batch[1]; i++ {
				for j := 0; j < real_batch[2]; j++ {
					ker_in23_new[k*2*real_batch[1]*real_batch[2]+2*i*real_batch[2]+j] = ker_in23[k*real_batch[1]*real_batch[2]+i*real_batch[2]+j]
				}
			}
		}

		norm = 1
		ct_result = evalConv_BNRelu_new(cont, ct_layer, ker_in23_new, bn_a3, bn_b3, alpha, in_wids[1], kp_wids[1], ker_wid, 2*real_batch[1], real_batch[2], norm, 0, 1, "StrConv", fast_pack, prt_result)
		fmt.Println("Block2 to 3 done!")
		max_bat = cont.N / (in_wids[2] * in_wids[2])
		res_ttmp = cont.encoder.DecodeCoeffs(cont.decryptor.DecryptNew(ct_result))
		prt_mat_norm(res_ttmp, max_bat, 4, 3, true)
		timings[3] = time.Since(new_start).Seconds()
		new_start = time.Now()

		// ResNet Block 3
		ct_layer = ct_result
		prt_result = false
		norm = 4
		for i := 1; i <= num_blc3; i++ {
			if i == num_blc3 {
				prt_result = true
			}
			bn_a3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blc1+num_blc2+i+1)+"-a.csv", real_batch[2])
			bn_b3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blc1+num_blc2+i+1)+"-b.csv", real_batch[2])
			ker_in3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blc1+num_blc2+i+1)+"-conv.csv", real_batch[2]*real_batch[2]*ker_size)

			ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in3, bn_a3, bn_b3, alpha, in_wids[2], kp_wids[2], ker_wid, real_batch[2], real_batch[2], norm, 0, 1, "Conv", fast_pack, prt_result)
			fmt.Println("Block3, Layer ", i, "done!")
		}
		timings[4] = time.Since(new_start).Seconds()

		new_start = time.Now()
		ker_inf := readTxt(weight_dir+"final-fckernel.csv", real_batch[2]*10)
		ker_inf_ := make([]float64, 9*9*real_batch[2]*10)
		for i := range ker_inf {
			for b := 0; b < 9*9; b++ {
				if (b%9 != 0) && (b/9 != 0) {
					ker_inf_[i+b*real_batch[2]*10] = ker_inf[i]
				}
			}
		}
		bn_af := make([]float64, 10)
		for i := range bn_af {
			bn_af[i] = 1.0 / (8 * 8) // for reduce mean on 8*8 elements
		}
		bn_bf := readTxt(weight_dir+"final-fcbias.csv", 10)
		norm = 4
		ct_result = evalConv_BN_sep(cont, ct_layer, ker_inf_, bn_af, bn_bf, in_wids[2], 9, real_batch[2], 10, norm, float64(1<<30), true, false)
		timings[5] = time.Since(new_start).Seconds()
		fmt.Println()
		fmt.Println("===============  DECRYPTION  ===============")
		fmt.Println()
		new_start = time.Now()
		cont.decryptor.Decrypt(ct_result, pl_input)
		res_tmp := cont.encoder.DecodeCoeffs(pl_input)
		fmt.Printf("Decryption Done in %s \n", time.Since(new_start))
		res_out := prt_mat_one_norm(res_tmp, max_batch[2], 4*4, 4, 4)
		fmt.Println("\n result: ", res_out[:10])
		writeTxt("class_result_"+ker_name+"/class_result_"+ker_name+"_"+strconv.Itoa(iter)+".csv", res_out[:10])

		fmt.Println("Blc1: ", timings[0], " sec")
		fmt.Println("Blc1->2: ", timings[1], " sec")
		fmt.Println("Blc2: ", timings[2], " sec")
		fmt.Println("Blc2->3: ", timings[3], " sec")
		fmt.Println("Blc3: ", timings[4], " sec")
		fmt.Println("Final (reduce_mean & FC): ", timings[5], " sec")
		fmt.Printf("Total done in %s \n", time.Since(begin_start))
	}
}

func testResNet_in_old(iter int) {
	// For ResNet, we use padding: i.e., in_wid**2 element is contained in (2*in_wid)**2 sized block
	// So ReLU, keep or rot, StoC done only on the 1st part of the CtoS ciphertexts
	weight_dir := "weight_ker5_h5/"
	ker_name := "ker5"
	norm := 1
	logN := 16
	num_blc1 := 5                   // 3 // 5 // 7
	num_blc2 := 3                   // 1 // 3 // 5
	num_blc3 := 3                   // 1 // 3 // 5
	raw_in_wids := []int{32, 16, 8} // same as python
	real_batch := []int{16, 32, 64} // same as python
	ker_wid := 5
	padding := true
	in_wids := make([]int, len(raw_in_wids))
	kp_wids := make([]int, len(raw_in_wids))
	for i, elt := range raw_in_wids {
		in_wids[i] = 2 * elt
		kp_wids[i] = elt
	}
	image := readTxt("test_data/test_image_"+strconv.Itoa(iter)+".csv", 32*32*3)
	cont := newContext(logN, ker_wid, in_wids, kp_wids, true, "Resnet")

	ker_size := ker_wid * ker_wid
	max_batch := make([]int, len(real_batch)) // the max batch
	for i := range max_batch {
		max_batch[i] = cont.N / (in_wids[i] * in_wids[i])
	}

	alpha := 0.0 // 0.3 => leakyrelu
	input := make([]float64, cont.N)
	k := 0
	for i := 0; i < in_wids[0]; i++ {
		for j := 0; j < in_wids[0]; j++ {
			for b := 0; b < max_batch[0]; b++ {
				if (i < raw_in_wids[0]) && (j < raw_in_wids[0]) && (b < 3) {
					input[i*in_wids[0]*max_batch[0]+j*max_batch[0]+b] = image[k]
					k++
				}
			}
		}
	}
	fmt.Println("Input: ")
	prt_mat(input, max_batch[0], 3)
	fmt.Println("vec size: ", cont.N)
	fmt.Println("input width: ", raw_in_wids)
	fmt.Println("kernel width: ", ker_wid)
	fmt.Println("num batches: ", real_batch)
	fmt.Println("Input matrix: ")
	prt_vec(input)

	start = time.Now()
	pl_input := ckks.NewPlaintext(cont.params, cont.ECD_LV, cont.params.Scale()) // contain plaintext values
	cont.encoder.EncodeCoeffs(input, pl_input)
	ct_input := cont.encryptor.EncryptNew(pl_input)
	fmt.Printf("Encryption done in %s \n", time.Since(start))

	timings := make([]float64, 6)
	begin_start := time.Now()
	new_start := time.Now()

	// ResNet Block 1
	// ct_layer := make([]*ckks.Ciphertext, num_blc1+1)
	ct_layer := ct_input
	prt_result := false
	for i := 1; i <= num_blc1; i++ {
		if i == num_blc1 {
			prt_result = true
		}
		bn_a := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-a.csv", real_batch[0])
		bn_b := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-b.csv", real_batch[0])
		if i == 1 {
			ker_in := readTxt(weight_dir+"w0-conv.csv", 3*real_batch[0]*ker_size)
			ct_layer = evalConv_BNRelu(cont, ct_layer, ker_in, bn_a, bn_b, alpha, in_wids[0], ker_wid, 3, real_batch[0], norm, 0, padding, false, prt_result)
		} else {
			ker_in := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-conv.csv", real_batch[0]*real_batch[0]*ker_size)
			ct_layer = evalConv_BNRelu(cont, ct_layer, ker_in, bn_a, bn_b, alpha, in_wids[0], ker_wid, real_batch[0], real_batch[0], norm, 0, padding, false, prt_result)
		}
		fmt.Println("Block1, Layer ", i, "done!")
	}
	fmt.Println("done.")
	timings[0] = time.Since(new_start).Seconds()
	new_start = time.Now()
	ker_in := readTxt(weight_dir+"w"+strconv.Itoa(num_blc1)+"-conv.csv", real_batch[0]*real_batch[1]*ker_size)
	ker_in_0 := make([]float64, len(ker_in)/2)
	ker_in_1 := make([]float64, len(ker_in)/2)
	for i := range ker_in_0 {
		ker_in_0[i] = ker_in[(i/(real_batch[1]/2))*real_batch[1]+i%(real_batch[1]/2)]
		ker_in_1[i] = ker_in[(i/(real_batch[1]/2))*real_batch[1]+i%(real_batch[1]/2)+real_batch[1]/2]
	}
	bn_a := readTxt(weight_dir+"w"+strconv.Itoa(num_blc1)+"-a.csv", real_batch[1])
	bn_b := readTxt(weight_dir+"w"+strconv.Itoa(num_blc1)+"-b.csv", real_batch[1])
	bn_a0 := bn_a[:real_batch[1]/2]
	bn_a1 := bn_a[real_batch[1]/2:]
	bn_b0 := bn_b[:real_batch[1]/2]
	bn_b1 := bn_b[real_batch[1]/2:]

	ct_result1 := evalConv_BNRelu(cont, ct_layer, ker_in_0, bn_a0, bn_b0, alpha, in_wids[0], ker_wid, real_batch[0], real_batch[1]/2, norm, 0, padding, true, prt_result)
	ct_result2 := evalConv_BNRelu(cont, ct_layer, ker_in_1, bn_a1, bn_b1, alpha, in_wids[0], ker_wid, real_batch[0], real_batch[1]/2, norm, 1, padding, true, prt_result)
	ct_result := cont.evaluator.AddNew(ct_result1, ct_result2)
	fmt.Println("Block1 to 2 done!")
	timings[1] = time.Since(new_start).Seconds()
	new_start = time.Now()

	// ResNet Block 2
	// ct_layer2 := make([]*ckks.Ciphertext, num_blc2+1)
	ct_layer2 := ct_result
	prt_result = false
	for i := 1; i <= num_blc2; i++ {
		if i == num_blc2 {
			prt_result = true
		}
		bn_a2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blc1+i)+"-a.csv", real_batch[1])
		bn_b2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blc1+i)+"-b.csv", real_batch[1])
		ker_in2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blc1+i)+"-conv.csv", real_batch[1]*real_batch[1]*ker_size)

		ct_layer2 = evalConv_BNRelu(cont, ct_layer2, ker_in2, bn_a2, bn_b2, alpha, in_wids[1], ker_wid, real_batch[1], real_batch[1], norm, 0, padding, false, prt_result)
		fmt.Println("Block2, Layer ", i, "done!")
	}
	timings[2] = time.Since(new_start).Seconds()
	new_start = time.Now()

	ker_in23 := readTxt(weight_dir+"w"+strconv.Itoa(num_blc1+num_blc2+1)+"-conv.csv", real_batch[1]*real_batch[2]*ker_size)
	bn_a3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blc1+num_blc2+1)+"-a.csv", real_batch[2])
	bn_b3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blc1+num_blc2+1)+"-b.csv", real_batch[2])

	ct_result = evalConv_BNRelu(cont, ct_layer2, ker_in23, bn_a3, bn_b3, alpha, in_wids[1], ker_wid, real_batch[1], real_batch[2], norm, 0, padding, true, prt_result)
	fmt.Println("Block2 to 3 done!")
	timings[3] = time.Since(new_start).Seconds()
	new_start = time.Now()

	// ResNet Block 3
	// ct_layer3 := make([]*ckks.Ciphertext, num_blc3+1)
	ct_layer3 := ct_result
	prt_result = false
	for i := 1; i <= num_blc3; i++ {
		if i == num_blc3 {
			prt_result = true
		}
		bn_a3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blc1+num_blc2+i+1)+"-a.csv", real_batch[2])
		bn_b3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blc1+num_blc2+i+1)+"-b.csv", real_batch[2])
		ker_in3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blc1+num_blc2+i+1)+"-conv.csv", real_batch[2]*real_batch[2]*ker_size)

		ct_layer3 = evalConv_BNRelu(cont, ct_layer3, ker_in3, bn_a3, bn_b3, alpha, in_wids[2], ker_wid, real_batch[2], real_batch[2], norm, 0, padding, false, prt_result)
		fmt.Println("Block3, Layer ", i, "done!")
	}
	timings[4] = time.Since(new_start).Seconds()

	new_start = time.Now()
	ker_inf := readTxt(weight_dir+"final-fckernel.csv", real_batch[2]*10)
	ker_inf_ := make([]float64, 9*9*real_batch[2]*10)
	for i := range ker_inf {
		for b := 0; b < 9*9; b++ {
			if (b%9 != 0) && (b/9 != 0) {
				ker_inf_[i+b*real_batch[2]*10] = ker_inf[i]
			}
		}
	}
	bn_af := make([]float64, 10)
	for i := range bn_af {
		bn_af[i] = 1.0 / (8 * 8) // for reduce mean on 8*8 elements
	}
	bn_bf := readTxt(weight_dir+"final-fcbias.csv", 10)
	ct_result = evalConv_BN(cont, ct_layer3, ker_inf_, bn_af, bn_bf, in_wids[2], 9, real_batch[2], 10, norm, float64(cont.params.Q()[0])/256.0, true, false)
	timings[5] = time.Since(new_start).Seconds()
	fmt.Println()
	fmt.Println("===============  DECRYPTION  ===============")
	fmt.Println()
	new_start = time.Now()
	cont.decryptor.Decrypt(ct_result, pl_input)
	res_tmp := cont.encoder.DecodeCoeffs(pl_input)
	fmt.Printf("Decryption Done in %s \n", time.Since(new_start))
	res_out := prt_mat_one(res_tmp, max_batch[2], 4, 4)
	fmt.Println("\n result: ", res_out[:10])
	writeTxt("class_result_"+ker_name+"/class_result_"+ker_name+"_"+strconv.Itoa(iter)+".csv", res_out)

	fmt.Println("Blc1: ", timings[0], " sec")
	fmt.Println("Blc1->2: ", timings[1], " sec")
	fmt.Println("Blc2: ", timings[2], " sec")
	fmt.Println("Blc2->3: ", timings[3], " sec")
	fmt.Println("Blc3: ", timings[4], " sec")
	fmt.Println("Final (reduce_mean & FC): ", timings[5], " sec")
	fmt.Printf("Total done in %s \n", time.Since(begin_start))
}

func testReduceMean() {
	norm := 1
	logN := 12
	raw_in_wids := []int{8, 8, 8}   // same as python
	real_batch := []int{16, 16, 16} // same as python
	in_wids := make([]int, len(raw_in_wids))
	kp_wids := make([]int, len(raw_in_wids))
	for i, elt := range raw_in_wids {
		in_wids[i] = 2 * elt
		kp_wids[i] = elt
	}
	cont := newContext(logN, 0, in_wids, kp_wids, true, "Resnet")

	max_batch := make([]int, len(real_batch)) // the max batch
	for i := range max_batch {
		max_batch[i] = cont.N / (in_wids[i] * in_wids[i])
	}

	input := make([]float64, cont.N)
	k := 0.0
	for i := 0; i < in_wids[0]; i++ {
		for j := 0; j < in_wids[0]; j++ {
			for b := 0; b < max_batch[0]; b++ {
				if (i < raw_in_wids[0]) && (j < raw_in_wids[0]) {
					input[i*in_wids[0]*max_batch[0]+j*max_batch[0]+b] = k
					k += 0.01
					// k += (1.0 / float64(real_batch[0]*(raw_in_wids[0])*(raw_in_wids[0])))
				}
			}
		}
	}
	fmt.Println("Input: ")
	prt_mat(input, max_batch[0], 0)
	pl_input := ckks.NewPlaintext(cont.params, cont.ECD_LV, cont.params.Scale()) // contain plaintext values
	cont.encoder.EncodeCoeffs(input, pl_input)
	ct_input := cont.encryptor.EncryptNew(pl_input)

	// ker_inf := readTxt("weight_h5/final-fckernel.csv", real_batch[2]*10)
	ker_inf := make([]float64, real_batch[2]*10)
	for i := range ker_inf {
		ker_inf[i] = 1.0 * float64(i)
	}
	ker_inf_ := make([]float64, 9*9*real_batch[2]*10)
	for i := range ker_inf {
		for b := 0; b < 9*9; b++ {
			if (b%9 != 0) && (b/9 != 0) {
				ker_inf_[i+b*real_batch[2]*10] = ker_inf[i]
			}
		}
	}
	bn_af := make([]float64, real_batch[2])
	for i := range bn_af {
		bn_af[i] = 1.0 / (8 * 8) // for reduce mean on 8*8 elements
	}
	// bn_bf := readTxt("weight_h5/final-fcbias.csv", 10)
	bn_bf := make([]float64, 10)
	for i := range bn_bf {
		bn_bf[i] = 10.0 * float64(i)
	}

	ct_result := evalConv_BN(cont, ct_input, ker_inf_, bn_af, bn_bf, in_wids[2], 9, real_batch[2], 10, norm, float64(cont.params.Q()[0])/256.0, true, false)
	cont.decryptor.Decrypt(ct_result, pl_input)
	res_tmp := cont.encoder.DecodeCoeffs(pl_input)
	prt_mat_one(res_tmp, max_batch[2], 4, 4)
}

func testReduceMean_norm() {
	norm := 4
	logN := 16
	raw_in_wids := []int{32, 16, 8} // same as python
	real_batch := []int{1, 1, 64}   // same as python
	in_wids := make([]int, len(raw_in_wids))
	kp_wids := make([]int, len(raw_in_wids))
	for i, elt := range raw_in_wids {
		in_wids[i] = 2 * elt
		kp_wids[i] = elt
	}
	cont := newContext(logN, 0, in_wids, kp_wids, true, "Resnet")

	max_batch := make([]int, len(real_batch)) // the max batch
	for i := range max_batch {
		max_batch[i] = cont.N / (in_wids[i] * in_wids[i])
	}

	input := make([]float64, cont.N)
	k := 0
	for i := 0; i < in_wids[2]; i++ {
		for j := 0; j < in_wids[2]; j++ {
			for b := 0; b < real_batch[2]; b++ {
				if (i < raw_in_wids[2]) && (j < raw_in_wids[2]) {
					input[i*in_wids[2]*max_batch[2]+j*max_batch[2]+norm*b] = float64(k) / 53.0
					k++
					if k == 53 {
						k = 0
					}
					// k += (1.0 / float64(real_batch[2]*(raw_in_wids[2])*(raw_in_wids[2])))
				}
			}
		}
	}
	fmt.Println("Input: ")
	prt_mat_norm(input, max_batch[2], norm, 3, true)
	pl_input := ckks.NewPlaintext(cont.params, cont.ECD_LV, cont.params.Scale()) // contain plaintext values
	cont.encoder.EncodeCoeffs(input, pl_input)
	ct_input := cont.encryptor.EncryptNew(pl_input)

	// ker_inf := readTxt("weight_h5/final-fckernel.csv", real_batch[2]*10)
	ker_inf := make([]float64, real_batch[2]*10)
	kk := 0
	for i := range ker_inf {
		// ker_inf[i] = 1.0 * float64(i)
		ker_inf[i] = float64(kk) / 13.0
		kk++
		if kk == 13 {
			kk = 0
		}
	}
	ker_inf_ := make([]float64, 9*9*real_batch[2]*10)
	for i := range ker_inf {
		for b := 0; b < 9*9; b++ {
			if (b%9 != 0) && (b/9 != 0) {
				ker_inf_[i+b*real_batch[2]*10] = ker_inf[i]
			}
		}
	}
	bn_af := make([]float64, 10)
	for i := range bn_af {
		bn_af[i] = 1.0 / (8 * 8) // for reduce mean on 8*8 elements
	}
	// bn_bf := readTxt("weight_h5/final-fcbias.csv", 10)
	bn_bf := make([]float64, 10)
	for i := range bn_bf {
		bn_bf[i] = 10.0 * float64(i)
	}

	ct_result := evalConv_BN_sep(cont, ct_input, ker_inf_, bn_af, bn_bf, in_wids[2], 9, real_batch[2], 10, norm, float64(1<<30), true, false)
	cont.decryptor.Decrypt(ct_result, pl_input)
	res_tmp := cont.encoder.DecodeCoeffs(pl_input)
	// prt_mat_norm(res_tmp, max_batch[2], norm, 4)
	prt_mat_one_norm(res_tmp, max_batch[2], 4*4, 4, 4)
}

func testResNet_ker23() {
	// For ResNet, we use padding: i.e., in_wid**2 element is contained in (2*in_wid)**2 sized block
	// So ReLU, keep or rot, StoC done only on the 1st part of the CtoS ciphertexts
	norm := 1
	logN := 14
	raw_in_wids := []int{32, 16, 8} // same as python
	real_batch := []int{4, 8, 16}   // same as python
	py_bn_a := []float64{0.2, 0.2, 0.2}
	// num_blc1 := 2
	num_blc2 := 1
	num_blc3 := 2
	ker_wid := 3
	padding := true
	fast_pack := true
	in_wids := make([]int, len(raw_in_wids))
	kp_wids := make([]int, len(raw_in_wids))
	for i, elt := range raw_in_wids {
		if padding {
			in_wids[i] = 2 * elt
			kp_wids[i] = elt
		} else {
			in_wids[i] = elt
		}
	}
	cont := newContext(logN, ker_wid, in_wids, kp_wids, true, "Resnet")

	ker_size := ker_wid * ker_wid
	max_batch := make([]int, len(real_batch)) // the max batch
	for i := range max_batch {
		max_batch[i] = cont.N / (in_wids[i] * in_wids[i])
	}

	alpha := 0.0 // 0.3 => leakyrelu
	input := make([]float64, cont.N)
	k := 0.0
	for i := 0; i < in_wids[1]; i++ {
		for j := 0; j < in_wids[1]; j++ {
			for b := 0; b < real_batch[1]; b++ {
				if (i < raw_in_wids[1]) && (j < raw_in_wids[1]) {
					input[i*in_wids[1]*max_batch[1]+j*max_batch[1]+2*b] = k
					k += 1.0 / float64(real_batch[1]*(raw_in_wids[1])*(raw_in_wids[1]))
				}
			}
		}
	}
	fmt.Println("Input: ")
	prt_mat(input, max_batch[1], 0)

	// for i := 0; i < in_wids[0]; i++ {
	// 	for j := 0; j < in_wids[0]; j++ {
	// 		for b := 0; b < 3; b++ {
	// 			if (i < raw_in_wids[0]) && (j < raw_in_wids[0]) {
	// 				input[i*in_wids[0]*max_batch[0]+j*max_batch[0]+b] = k
	// 				k -= 1.0 / float64(3*(raw_in_wids[0])*(raw_in_wids[0]))
	// 			}
	// 		}
	// 	}
	// }
	// fmt.Println("Input: ")
	// prt_mat(input, max_batch[0], 3)
	ker_in := make([]float64, 3*real_batch[0]*ker_size)
	for i := range ker_in {
		ker_in[i] = 0.25 * float64(i) / float64(len(ker_in))
	}
	ker_in1 := make([]float64, real_batch[0]*real_batch[0]*ker_size)
	for i := range ker_in1 {
		ker_in1[i] = 0.25 * float64(i) / float64(len(ker_in1))
	}
	ker_in12 := make([]float64, real_batch[0]*real_batch[1]*ker_size)
	for i := range ker_in12 {
		ker_in12[i] = 0.25 * float64(i) / float64(len(ker_in12))
	}
	ker_in12_0 := make([]float64, len(ker_in12)/2)
	ker_in12_1 := make([]float64, len(ker_in12)/2)
	for k := 0; k < ker_size; k++ {
		for i := 0; i < real_batch[0]; i++ {
			for j := 0; j < real_batch[1]/2; j++ {
				ker_in12_0[k*real_batch[0]*real_batch[1]/2+(i*real_batch[1]/2+j)] = ker_in12[k*real_batch[0]*real_batch[1]+(i*real_batch[1]+2*j)]   // [i][2*j]
				ker_in12_1[k*real_batch[0]*real_batch[1]/2+(i*real_batch[1]/2+j)] = ker_in12[k*real_batch[0]*real_batch[1]+(i*real_batch[1]+2*j+1)] // [i][2*j+1]
			}
		}
	}

	ker_in2 := make([]float64, real_batch[1]*real_batch[1]*ker_size)
	for i := range ker_in2 {
		ker_in2[i] = 0.25 * float64(i) / float64(len(ker_in2))
	}
	ker_in23 := make([]float64, real_batch[1]*real_batch[2]*ker_size)
	for i := range ker_in23 {
		ker_in23[i] = 0.25 * float64(100) / float64(len(ker_in23))
	}
	ker_in23_new := make([]float64, 2*real_batch[1]*real_batch[2]*ker_size)
	for k := 0; k < ker_size; k++ {
		for i := 0; i < real_batch[1]; i++ {
			for j := 0; j < real_batch[2]; j++ {
				ker_in23_new[k*2*real_batch[1]*real_batch[2]+2*i*real_batch[2]+j] = ker_in23[k*real_batch[1]*real_batch[2]+i*real_batch[2]+j]
			}
		}
	}

	ker_in3 := make([]float64, real_batch[2]*real_batch[2]*ker_size)
	for i := range ker_in3 {
		ker_in3[i] = 0.25 * float64(i) / float64(len(ker_in3))
	}
	bn_a := make([]float64, real_batch[0])
	bn_b := make([]float64, real_batch[0])
	for i := range bn_a {
		bn_a[i] = py_bn_a[0] // * float64(i) / float64(batch)
		bn_b[i] = 0.0        //0.1 * float64(i) // float64(real_batch[0])
	}
	bn_a2 := make([]float64, real_batch[1])
	bn_b2 := make([]float64, real_batch[1])
	for i := range bn_a2 {
		bn_a2[i] = py_bn_a[1] // * float64(i) / float64(batch)
		bn_b2[i] = 0.0        //0.1 * float64(i)
	}
	bn_a2_0 := make([]float64, real_batch[1]/2)
	bn_a2_1 := make([]float64, real_batch[1]/2)
	bn_b2_0 := make([]float64, real_batch[1]/2)
	bn_b2_1 := make([]float64, real_batch[1]/2)
	for i := range bn_b2_0 {
		bn_a2_0[i] = bn_a2[2*i]
		bn_a2_1[i] = bn_a2[2*i+1]
		bn_b2_0[i] = bn_b2[2*i]
		bn_b2_1[i] = bn_b2[2*i+1]
	}

	bn_a3 := make([]float64, real_batch[2])
	bn_b3 := make([]float64, real_batch[2])
	for i := range bn_a3 {
		bn_a3[i] = py_bn_a[2] // * float64(i) / float64(batch)
		bn_b3[i] = 0.0        //0.1 * float64(i)
	}

	fmt.Println("vec size: ", cont.N)
	fmt.Println("input width: ", raw_in_wids)
	fmt.Println("kernel width: ", ker_wid)
	fmt.Println("num batches: ", real_batch)
	fmt.Println("Input matrix: ")
	prt_vec(input)

	start = time.Now()
	pl_input := ckks.NewPlaintext(cont.params, cont.ECD_LV, cont.params.Scale()) // contain plaintext values
	cont.encoder.EncodeCoeffs(input, pl_input)
	ct_input := cont.encryptor.EncryptNew(pl_input)
	fmt.Printf("Encryption done in %s \n", time.Since(start))

	timings := make([]float64, 6)
	begin_start := time.Now()
	new_start := time.Now()

	// // ResNet Block 1
	// ct_layer := ct_input
	// prt_result := true
	// for i := 1; i <= num_blc1; i++ {
	// 	if i == num_blc1 {
	// 		prt_result = true
	// 	}
	// 	if i == 1 {
	// 		ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in, bn_a, bn_b, alpha, in_wids[0], kp_wids[0], ker_wid, 3, real_batch[0], norm, 0, 1, "Conv", prt_result)
	// 	} else {
	// 		ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in1, bn_a, bn_b, alpha, in_wids[0], kp_wids[0], ker_wid, real_batch[0], real_batch[0], norm, 0, 1, "Conv", prt_result)
	// 	}
	// 	fmt.Println("Block1, Layer ", i, "done!")
	// }
	// fmt.Println("done.")
	// timings[0] = time.Since(new_start).Seconds()
	// new_start = time.Now()

	// ct_result1 := evalConv_BNRelu_new(cont, ct_layer, ker_in12_0, bn_a2_0, bn_b2_0, alpha, in_wids[0], kp_wids[0], ker_wid, real_batch[0], real_batch[1]/2, norm, 0, 1, "StrConv", false)
	// ct_result2 := evalConv_BNRelu_new(cont, ct_layer, ker_in12_1, bn_a2_1, bn_b2_1, alpha, in_wids[0], kp_wids[0], ker_wid, real_batch[0], real_batch[1]/2, norm, 2, 1, "StrConv", false)
	// ct_result := cont.evaluator.AddNew(ct_result1, ct_result2)
	// fmt.Println("Block1 to 2 done!")
	// max_bat := cont.N / (in_wids[1] * in_wids[1])
	// res_ttmp := cont.encoder.DecodeCoeffs(cont.decryptor.DecryptNew(ct_result))
	// prt_mat_norm(res_ttmp, max_bat, 2, 3)

	timings[1] = time.Since(new_start).Seconds()
	new_start = time.Now()

	// ResNet Block 2
	ct_layer := ct_input
	prt_result := true
	norm = 2
	for i := 1; i <= num_blc2; i++ {
		if i == num_blc2 {
			prt_result = true
		}
		ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in2, bn_a2, bn_b2, alpha, in_wids[1], kp_wids[1], ker_wid, real_batch[1], real_batch[1], norm, 0, 1, "Conv", fast_pack, prt_result)
		fmt.Println("Block2, Layer ", i, "done!")
	}
	timings[2] = time.Since(new_start).Seconds()
	new_start = time.Now()
	norm = 1
	ct_result := evalConv_BNRelu_new(cont, ct_layer, ker_in23_new, bn_a3, bn_b3, alpha, in_wids[1], kp_wids[1], ker_wid, 2*real_batch[1], real_batch[2], norm, 0, 1, "StrConv", fast_pack, prt_result)
	fmt.Println("Block2 to 3 done!")
	max_bat := cont.N / (in_wids[2] * in_wids[2])
	res_ttmp := cont.encoder.DecodeCoeffs(cont.decryptor.DecryptNew(ct_result))
	prt_mat_norm(res_ttmp, max_bat, 4, 3, true)
	timings[3] = time.Since(new_start).Seconds()
	new_start = time.Now()

	// ResNet Block 3
	ct_layer = ct_result
	prt_result = true
	norm = 4
	for i := 1; i <= num_blc3; i++ {
		if i == num_blc3 {
			prt_result = true
		}
		ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in3, bn_a3, bn_b3, alpha, in_wids[2], kp_wids[2], ker_wid, real_batch[2], real_batch[2], norm, 0, 1, "Conv", fast_pack, prt_result)
		fmt.Println("Block3, Layer ", i, "done!")
	}
	timings[4] = time.Since(new_start).Seconds()

	new_start = time.Now()
	// // ker_inf := readTxt("weight_h5/final-fckernel.csv", real_batch[2]*10)
	// ker_inf := make([]float64, real_batch[2]*10)
	// for i := range ker_inf {
	// 	ker_inf[i] = 1.0 * float64(i)
	// }
	// ker_inf_ := make([]float64, 9*9*real_batch[2]*10)
	// for i := range ker_inf {
	// 	for b := 0; b < 9*9; b++ {
	// 		if (b%9 != 0) && (b/9 != 0) {
	// 			ker_inf_[i+b*real_batch[2]*10] = ker_inf[i]
	// 		}
	// 	}
	// }
	// bn_af := make([]float64, 10)
	// for i := range bn_af {
	// 	bn_af[i] = 1.0 / (8 * 8) // for reduce mean on 8*8 elements
	// }
	// // bn_bf := readTxt("weight_h5/final-fcbias.csv", 10)
	// bn_bf := make([]float64, 10)
	// for i := range bn_bf {
	// 	bn_bf[i] = 1.0 * float64(i)
	// }

	// ct_result = evalConv_BN(cont, ct_input, ker_inf_, bn_af, bn_bf, in_wids[2], 9, real_batch[2], norm, 10, true, false)
	cont.decryptor.Decrypt(ct_layer, pl_input)
	res_tmp := cont.encoder.DecodeCoeffs(pl_input)
	prt_mat_norm(res_tmp, max_batch[2], 4, 0, true)
	timings[5] = time.Since(new_start).Seconds()

	fmt.Println("Blc1: ", timings[0], " sec")
	fmt.Println("Blc1->2: ", timings[1], " sec")
	fmt.Println("Blc2: ", timings[2], " sec")
	fmt.Println("Blc2->3: ", timings[3], " sec")
	fmt.Println("Blc3: ", timings[4], " sec")
	fmt.Println("Final (reduce_mean & FC): ", timings[5], " sec")
	fmt.Printf("Total done in %s \n", time.Since(begin_start))
}

func testResNet() {
	// For ResNet, we use padding: i.e., in_wid**2 element is contained in (2*in_wid)**2 sized block
	// So ReLU, keep or rot, StoC done only on the 1st part of the CtoS ciphertexts
	norm := 1
	logN := 14
	raw_in_wids := []int{32, 16, 8} // same as python
	real_batch := []int{4, 8, 16}   // same as python
	py_bn_a := []float64{0.2, 0.2, 0.2}
	num_blc1 := 2
	num_blc2 := 2
	num_blc3 := 2
	ker_wid := 3
	padding := true
	fast_pack := true
	in_wids := make([]int, len(raw_in_wids))
	kp_wids := make([]int, len(raw_in_wids))
	for i, elt := range raw_in_wids {
		if padding {
			in_wids[i] = 2 * elt
			kp_wids[i] = elt
		} else {
			in_wids[i] = elt
		}
	}
	cont := newContext(logN, ker_wid, in_wids, kp_wids, true, "Resnet")

	ker_size := ker_wid * ker_wid
	max_batch := make([]int, len(real_batch)) // the max batch
	for i := range max_batch {
		max_batch[i] = cont.N / (in_wids[i] * in_wids[i])
	}

	alpha := 0.0 // 0.3 => leakyrelu
	input := make([]float64, cont.N)
	k := 1.0
	for i := 0; i < in_wids[0]; i++ {
		for j := 0; j < in_wids[0]; j++ {
			for b := 0; b < 3; b++ {
				if (i < raw_in_wids[0]) && (j < raw_in_wids[0]) {
					input[i*in_wids[0]*max_batch[0]+j*max_batch[0]+b] = k
					k -= 1.0 / float64(3*(raw_in_wids[0])*(raw_in_wids[0]))
				}
			}
		}
	}
	fmt.Println("Input: ")
	prt_mat(input, max_batch[0], 3)
	ker_in := make([]float64, 3*real_batch[0]*ker_size)
	for i := range ker_in {
		ker_in[i] = 0.25 * float64(i) / float64(len(ker_in))
	}
	ker_in1 := make([]float64, real_batch[0]*real_batch[0]*ker_size)
	for i := range ker_in1 {
		ker_in1[i] = 0.25 * float64(i) / float64(len(ker_in1))
	}
	ker_in12 := make([]float64, real_batch[0]*real_batch[1]*ker_size)
	for i := range ker_in12 {
		ker_in12[i] = 0.25 * float64(i) / float64(len(ker_in12))
	}
	ker_in12_0 := make([]float64, len(ker_in12)/2)
	ker_in12_1 := make([]float64, len(ker_in12)/2)
	for k := 0; k < ker_size; k++ {
		for i := 0; i < real_batch[0]; i++ {
			for j := 0; j < real_batch[1]/2; j++ {
				ker_in12_0[k*real_batch[0]*real_batch[1]/2+(i*real_batch[1]/2+j)] = ker_in12[k*real_batch[0]*real_batch[1]+(i*real_batch[1]+2*j)]   // [i][2*j]
				ker_in12_1[k*real_batch[0]*real_batch[1]/2+(i*real_batch[1]/2+j)] = ker_in12[k*real_batch[0]*real_batch[1]+(i*real_batch[1]+2*j+1)] // [i][2*j+1]
			}
		}
	}

	ker_in2 := make([]float64, real_batch[1]*real_batch[1]*ker_size)
	for i := range ker_in2 {
		ker_in2[i] = 0.25 * float64(i) / float64(len(ker_in2))
	}
	ker_in23 := make([]float64, real_batch[1]*real_batch[2]*ker_size)
	for i := range ker_in23 {
		ker_in23[i] = 0.25 * float64(100) / float64(len(ker_in23))
	}
	ker_in23_new := make([]float64, 2*real_batch[1]*real_batch[2]*ker_size)
	for k := 0; k < ker_size; k++ {
		for i := 0; i < real_batch[1]; i++ {
			for j := 0; j < real_batch[2]; j++ {
				ker_in23_new[k*2*real_batch[1]*real_batch[2]+2*i*real_batch[2]+j] = ker_in23[k*real_batch[1]*real_batch[2]+i*real_batch[2]+j]
			}
		}
	}

	ker_in3 := make([]float64, real_batch[2]*real_batch[2]*ker_size)
	for i := range ker_in3 {
		ker_in3[i] = 0.25 * float64(i) / float64(len(ker_in3))
	}
	bn_a := make([]float64, real_batch[0])
	bn_b := make([]float64, real_batch[0])
	for i := range bn_a {
		bn_a[i] = py_bn_a[0] // * float64(i) / float64(batch)
		bn_b[i] = 0.0        //0.1 * float64(i) // float64(real_batch[0])
	}
	bn_a2 := make([]float64, real_batch[1])
	bn_b2 := make([]float64, real_batch[1])
	for i := range bn_a2 {
		bn_a2[i] = py_bn_a[1] // * float64(i) / float64(batch)
		bn_b2[i] = 0.0        //0.1 * float64(i)
	}
	bn_a2_0 := make([]float64, real_batch[1]/2)
	bn_a2_1 := make([]float64, real_batch[1]/2)
	bn_b2_0 := make([]float64, real_batch[1]/2)
	bn_b2_1 := make([]float64, real_batch[1]/2)
	for i := range bn_b2_0 {
		bn_a2_0[i] = bn_a2[2*i]
		bn_a2_1[i] = bn_a2[2*i+1]
		bn_b2_0[i] = bn_b2[2*i]
		bn_b2_1[i] = bn_b2[2*i+1]
	}

	bn_a3 := make([]float64, real_batch[2])
	bn_b3 := make([]float64, real_batch[2])
	for i := range bn_a3 {
		bn_a3[i] = py_bn_a[2] // * float64(i) / float64(batch)
		bn_b3[i] = 0.0        //0.1 * float64(i)
	}

	fmt.Println("vec size: ", cont.N)
	fmt.Println("input width: ", raw_in_wids)
	fmt.Println("kernel width: ", ker_wid)
	fmt.Println("num batches: ", real_batch)
	fmt.Println("Input matrix: ")
	prt_vec(input)

	start = time.Now()
	pl_input := ckks.NewPlaintext(cont.params, cont.ECD_LV, cont.params.Scale()) // contain plaintext values
	cont.encoder.EncodeCoeffs(input, pl_input)
	ct_input := cont.encryptor.EncryptNew(pl_input)
	fmt.Printf("Encryption done in %s \n", time.Since(start))

	timings := make([]float64, 6)
	begin_start := time.Now()
	new_start := time.Now()

	// ResNet Block 1
	ct_layer := ct_input
	prt_result := true
	for i := 1; i <= num_blc1; i++ {
		if i == num_blc1 {
			prt_result = true
		}
		if i == 1 {
			ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in, bn_a, bn_b, alpha, in_wids[0], kp_wids[0], ker_wid, 3, real_batch[0], norm, 0, 1, "Conv", fast_pack, prt_result)
		} else {
			ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in1, bn_a, bn_b, alpha, in_wids[0], kp_wids[0], ker_wid, real_batch[0], real_batch[0], norm, 0, 1, "Conv", fast_pack, prt_result)
		}
		fmt.Println("Block1, Layer ", i, "done!")
	}
	fmt.Println("done.")
	timings[0] = time.Since(new_start).Seconds()
	new_start = time.Now()

	ct_result1 := evalConv_BNRelu_new(cont, ct_layer, ker_in12_0, bn_a2_0, bn_b2_0, alpha, in_wids[0], kp_wids[0], ker_wid, real_batch[0], real_batch[1]/2, norm, 0, 1, "StrConv", fast_pack, false)
	ct_result2 := evalConv_BNRelu_new(cont, ct_layer, ker_in12_1, bn_a2_1, bn_b2_1, alpha, in_wids[0], kp_wids[0], ker_wid, real_batch[0], real_batch[1]/2, norm, 2, 1, "StrConv", fast_pack, false)
	ct_result := cont.evaluator.AddNew(ct_result1, ct_result2)
	fmt.Println("Block1 to 2 done!")
	max_bat := cont.N / (in_wids[1] * in_wids[1])
	res_ttmp := cont.encoder.DecodeCoeffs(cont.decryptor.DecryptNew(ct_result))
	prt_mat_norm(res_ttmp, max_bat, 2, 3, true)

	timings[1] = time.Since(new_start).Seconds()
	new_start = time.Now()

	// ResNet Block 2
	ct_layer = ct_result
	prt_result = true
	norm = 2
	for i := 1; i <= num_blc2; i++ {
		if i == num_blc2 {
			prt_result = true
		}
		ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in2, bn_a2, bn_b2, alpha, in_wids[1], kp_wids[1], ker_wid, real_batch[1], real_batch[1], norm, 0, 1, "Conv", fast_pack, prt_result)
		fmt.Println("Block2, Layer ", i, "done!")
	}
	timings[2] = time.Since(new_start).Seconds()
	new_start = time.Now()
	norm = 1
	ct_result = evalConv_BNRelu_new(cont, ct_layer, ker_in23_new, bn_a3, bn_b3, alpha, in_wids[1], kp_wids[1], ker_wid, 2*real_batch[1], real_batch[2], norm, 0, 1, "StrConv", fast_pack, prt_result)
	fmt.Println("Block2 to 3 done!")
	max_bat = cont.N / (in_wids[2] * in_wids[2])
	res_ttmp = cont.encoder.DecodeCoeffs(cont.decryptor.DecryptNew(ct_result))
	prt_mat_norm(res_ttmp, max_bat, 4, 3, true)
	timings[3] = time.Since(new_start).Seconds()
	new_start = time.Now()

	// ResNet Block 3
	ct_layer = ct_result
	prt_result = true
	norm = 4
	for i := 1; i <= num_blc3; i++ {
		if i == num_blc3 {
			prt_result = true
		}
		ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in3, bn_a3, bn_b3, alpha, in_wids[2], kp_wids[2], ker_wid, real_batch[2], real_batch[2], norm, 0, 1, "Conv", fast_pack, prt_result)
		fmt.Println("Block3, Layer ", i, "done!")
	}
	timings[4] = time.Since(new_start).Seconds()

	new_start = time.Now()
	// // ker_inf := readTxt("weight_h5/final-fckernel.csv", real_batch[2]*10)
	// ker_inf := make([]float64, real_batch[2]*10)
	// for i := range ker_inf {
	// 	ker_inf[i] = 1.0 * float64(i)
	// }
	// ker_inf_ := make([]float64, 9*9*real_batch[2]*10)
	// for i := range ker_inf {
	// 	for b := 0; b < 9*9; b++ {
	// 		if (b%9 != 0) && (b/9 != 0) {
	// 			ker_inf_[i+b*real_batch[2]*10] = ker_inf[i]
	// 		}
	// 	}
	// }
	// bn_af := make([]float64, 10)
	// for i := range bn_af {
	// 	bn_af[i] = 1.0 / (8 * 8) // for reduce mean on 8*8 elements
	// }
	// // bn_bf := readTxt("weight_h5/final-fcbias.csv", 10)
	// bn_bf := make([]float64, 10)
	// for i := range bn_bf {
	// 	bn_bf[i] = 1.0 * float64(i)
	// }

	// ct_result = evalConv_BN(cont, ct_input, ker_inf_, bn_af, bn_bf, in_wids[2], 9, real_batch[2], norm, 10, true, false)
	cont.decryptor.Decrypt(ct_layer, pl_input)
	res_tmp := cont.encoder.DecodeCoeffs(pl_input)
	prt_mat_norm(res_tmp, max_batch[2], 4, 0, true)
	timings[5] = time.Since(new_start).Seconds()

	fmt.Println("Blc1: ", timings[0], " sec")
	fmt.Println("Blc1->2: ", timings[1], " sec")
	fmt.Println("Blc2: ", timings[2], " sec")
	fmt.Println("Blc2->3: ", timings[3], " sec")
	fmt.Println("Blc3: ", timings[4], " sec")
	fmt.Println("Final (reduce_mean & FC): ", timings[5], " sec")
	fmt.Printf("Total done in %s \n", time.Since(begin_start))
}

func testResNet_old() {
	norm := 1
	// For ResNet, we use padding: i.e., in_wid**2 element is contained in (2*in_wid)**2 sized block
	// So ReLU, keep or rot, StoC done only on the 1st part of the CtoS ciphertexts
	logN := 12
	raw_in_wids := []int{32, 16, 8} // same as python
	real_batch := []int{1, 2, 4}    // same as python
	py_bn_a := []float64{0.5, 0.5, 0.5}
	ker_wid := 3
	padding := true
	in_wids := make([]int, len(raw_in_wids))
	kp_wids := make([]int, len(raw_in_wids))
	for i, elt := range raw_in_wids {
		if padding {
			in_wids[i] = 2 * elt
			kp_wids[i] = elt
		} else {
			in_wids[i] = elt
		}
	}
	cont := newContext(logN, ker_wid, in_wids, kp_wids, true, "Resnet")
	// cont := newContext(logN, ECD_LV, in_wids, padding, "Resnet")

	ker_size := ker_wid * ker_wid
	max_batch := make([]int, len(real_batch)) // the max batch
	for i := range max_batch {
		max_batch[i] = cont.N / (in_wids[i] * in_wids[i])
	}

	alpha := 0.0 // 0.3 => leakyrelu
	input := make([]float64, cont.N)
	k := 0.0
	for i := 0; i < in_wids[0]; i++ {
		for j := 0; j < in_wids[0]; j++ {
			for b := 0; b < max_batch[0]; b++ {
				if (i < raw_in_wids[0]) && (j < raw_in_wids[0]) && (b < real_batch[0]) {
					input[i*in_wids[0]*max_batch[0]+j*max_batch[0]+b] = k
					k += (10.0 / float64(real_batch[0]*(raw_in_wids[0])*(raw_in_wids[0])))
				}
			}
		}
	}
	fmt.Println("Input: ")
	prt_mat(input, max_batch[0], 3)
	ker_in := make([]float64, real_batch[0]*real_batch[0]*ker_size)
	for i := range ker_in {
		ker_in[i] = 0.5 * float64(i) / float64(len(ker_in))
	}
	ker_in12 := make([]float64, real_batch[0]*real_batch[1]*ker_size)
	for i := range ker_in12 {
		ker_in12[i] = 0.5 * float64(i) / float64(len(ker_in12))
	}
	ker_in12_0 := make([]float64, len(ker_in12)/2)
	ker_in12_1 := make([]float64, len(ker_in12)/2)
	for i := range ker_in12_0 {
		ker_in12_0[i] = ker_in12[(i/(real_batch[1]/2))*real_batch[1]+i%(real_batch[1]/2)]
		ker_in12_1[i] = ker_in12[(i/(real_batch[1]/2))*real_batch[1]+i%(real_batch[1]/2)+real_batch[1]/2]
	}
	ker_in2 := make([]float64, real_batch[1]*real_batch[1]*ker_size)
	for i := range ker_in2 {
		ker_in2[i] = 0.5 * float64(i) / float64(len(ker_in2))
	}
	ker_in23 := make([]float64, real_batch[1]*real_batch[2]*ker_size)
	for i := range ker_in23 {
		ker_in23[i] = 0.5 * float64(i) / float64(len(ker_in23))
	}
	ker_in3 := make([]float64, real_batch[2]*real_batch[2]*ker_size)
	for i := range ker_in3 {
		ker_in3[i] = 0.5 * float64(i) / float64(len(ker_in3))
	}
	bn_a := make([]float64, real_batch[0])
	bn_b := make([]float64, real_batch[0])
	for i := range bn_a {
		bn_a[i] = py_bn_a[0] // * float64(i) / float64(batch)
		bn_b[i] = 0.0        //0.1 * float64(i) // float64(real_batch[0])
	}
	bn_a2 := make([]float64, real_batch[1])
	bn_b2 := make([]float64, real_batch[1])
	for i := range bn_a2 {
		bn_a2[i] = py_bn_a[1] // * float64(i) / float64(batch)
		bn_b2[i] = 0.0        //0.1 * float64(i)
	}
	bn_a3 := make([]float64, real_batch[2])
	bn_b3 := make([]float64, real_batch[2])
	for i := range bn_a3 {
		bn_a3[i] = py_bn_a[2] // * float64(i) / float64(batch)
		bn_b3[i] = 0.0        //0.1 * float64(i)
	}

	fmt.Println("vec size: ", cont.N)
	fmt.Println("input width: ", raw_in_wids)
	fmt.Println("kernel width: ", ker_wid)
	fmt.Println("num batches: ", real_batch)
	fmt.Println("Input matrix: ")
	prt_vec(input)

	start = time.Now()
	pl_input := ckks.NewPlaintext(cont.params, cont.ECD_LV, cont.params.Scale()) // contain plaintext values
	cont.encoder.EncodeCoeffs(input, pl_input)
	ct_input := cont.encryptor.EncryptNew(pl_input)
	fmt.Printf("Encryption done in %s \n", time.Since(start))

	timings := make([]float64, 6)
	begin_start := time.Now()
	new_start := time.Now()

	// ResNet Block 1
	num_blc1 := 1
	ct_layer := make([]*ckks.Ciphertext, num_blc1+1)
	ct_layer[0] = ct_input
	prt_result := false
	for i := 1; i <= num_blc1; i++ {
		if i == num_blc1 {
			prt_result = true
		}
		ct_layer[i] = evalConv_BNRelu(cont, ct_layer[i-1], ker_in, bn_a, bn_b, alpha, in_wids[0], ker_wid, max_batch[0], max_batch[0], norm, 0, padding, false, prt_result)
		fmt.Println("Block1, Layer ", i, "done!")
	}
	fmt.Println("done.")
	timings[0] = time.Since(new_start).Seconds()
	new_start = time.Now()

	ct_result1 := evalConv_BNRelu(cont, ct_layer[num_blc1], ker_in12_0, bn_a, bn_b, alpha, in_wids[0], ker_wid, real_batch[0], real_batch[1]/2, norm, 0, padding, true, prt_result)
	ct_result2 := evalConv_BNRelu(cont, ct_layer[num_blc1], ker_in12_1, bn_a, bn_b, alpha, in_wids[0], ker_wid, real_batch[0], real_batch[1]/2, norm, 1, padding, true, prt_result)
	ct_result := cont.evaluator.AddNew(ct_result1, ct_result2)
	fmt.Println("Block1 to 2 done!")
	timings[1] = time.Since(new_start).Seconds()
	new_start = time.Now()

	// ResNet Block 2
	num_blc2 := 1
	ct_layer2 := make([]*ckks.Ciphertext, num_blc2+1)
	ct_layer2[0] = ct_result
	prt_result = false
	for i := 1; i <= num_blc2; i++ {
		if i == num_blc2 {
			prt_result = true
		}
		ct_layer2[i] = evalConv_BNRelu(cont, ct_layer2[i-1], ker_in2, bn_a2, bn_b2, alpha, in_wids[1], ker_wid, real_batch[1], real_batch[1], norm, 0, padding, false, prt_result)
		fmt.Println("Block2, Layer ", i, "done!")
	}
	timings[2] = time.Since(new_start).Seconds()
	new_start = time.Now()

	ct_result = evalConv_BNRelu(cont, ct_layer2[num_blc2], ker_in23, bn_a3, bn_b3, alpha, in_wids[1], ker_wid, real_batch[1], real_batch[2], norm, 0, padding, true, prt_result)
	fmt.Println("Block2 to 3 done!")
	timings[3] = time.Since(new_start).Seconds()
	new_start = time.Now()

	// ResNet Block 3
	num_blc3 := 1
	ct_layer3 := make([]*ckks.Ciphertext, num_blc3+1)
	ct_layer3[0] = ct_result
	prt_result = false
	for i := 1; i <= num_blc3; i++ {
		if i == num_blc3 {
			prt_result = true
		}
		ct_layer3[i] = evalConv_BNRelu(cont, ct_layer3[i-1], ker_in3, bn_a3, bn_b3, alpha, in_wids[2], ker_wid, real_batch[2], real_batch[2], norm, 0, padding, false, prt_result)
		fmt.Println("Block3, Layer ", i, "done!")
	}
	timings[4] = time.Since(new_start).Seconds()

	new_start = time.Now()
	// ker_inf := readTxt("weight_h5/final-fckernel.csv", real_batch[2]*10)
	ker_inf := make([]float64, real_batch[2]*10)
	for i := range ker_inf {
		ker_inf[i] = 1.0 * float64(i)
	}
	ker_inf_ := make([]float64, 9*9*real_batch[2]*10)
	for i := range ker_inf {
		for b := 0; b < 9*9; b++ {
			if (b%9 != 0) && (b/9 != 0) {
				ker_inf_[i+b*real_batch[2]*10] = ker_inf[i]
			}
		}
	}
	bn_af := make([]float64, 10)
	for i := range bn_af {
		bn_af[i] = 1.0 / (8 * 8) // for reduce mean on 8*8 elements
	}
	// bn_bf := readTxt("weight_h5/final-fcbias.csv", 10)
	bn_bf := make([]float64, 10)
	for i := range bn_bf {
		bn_bf[i] = 1.0 * float64(i)
	}

	ct_result = evalConv_BN(cont, ct_input, ker_inf_, bn_af, bn_bf, in_wids[2], 9, real_batch[2], 10, norm, float64(1<<30), true, false)
	cont.decryptor.Decrypt(ct_result, pl_input)
	res_tmp := cont.encoder.DecodeCoeffs(pl_input)
	prt_mat_one(res_tmp, max_batch[2], 4, 4)
	timings[5] = time.Since(new_start).Seconds()

	fmt.Println("Blc1: ", timings[0], " sec")
	fmt.Println("Blc1->2: ", timings[1], " sec")
	fmt.Println("Blc2: ", timings[2], " sec")
	fmt.Println("Blc2->3: ", timings[3], " sec")
	fmt.Println("Blc3: ", timings[4], " sec")
	fmt.Println("Final (reduce_mean & FC): ", timings[5], " sec")
	fmt.Printf("Total done in %s \n", time.Since(begin_start))
}

// set input as in_wid * in_wid * batch, then zero padding other values.
func testBRrot() {
	batch := 4
	in_wid := 16
	kp_wid := 14
	pos := 2
	in_size := in_wid * in_wid
	N := in_size * batch
	logN := 0
	for ; (1 << logN) < N; logN++ {
	}
	// padding := false

	sm_input := make([]int, in_size) // each will be packed to input vector
	input := make([]int, N)
	input_up_rev := make([]int, N/2) // for upper or lowerpart
	input_lw_rev := make([]int, N/2) // for upper or lowerpart
	output := make([]int, N)

	// set input and desired output
	for b := 0; b < batch; b++ {
		for i := range sm_input {
			sm_input[i] = i + b*in_size
		}
		arrgvec(sm_input, input, b)
	}

	for b := 0; b < batch; b++ {
		print_vec_int("input ("+strconv.Itoa(b)+")", input, in_wid, b)
	}

	input_up := input[0 : N/2]
	input_lw := input[N/2 : N]
	for i := range input_up {
		input_up_rev[reverseBits(uint32(i), logN-1)] = input_up[i]
		input_lw_rev[reverseBits(uint32(i), logN-1)] = input_lw[i]
	}

	// output_up_rev := keep_vec(input_up_rev, in_wid, 4, 0)
	// output_lw_rev := keep_vec(input_lw_rev, in_wid, 4, 1)
	// output_up_rev := comprs_full_hf(input_up_rev, in_wid, kp_wid, pos, 0)
	// output_lw_rev := comprs_full_hf(input_lw_rev, in_wid, kp_wid, pos, 1)
	// output_up_rev := comprs_full_int(input_up_rev, in_wid, kp_wid, pos, 0)
	// output_lw_rev := comprs_full_int(input_lw_rev, in_wid, kp_wid, pos, 1)
	// // output_up_rev := extend_full_int(input_up_rev, in_wid, kp_wid, pos, 0)
	// output_lw_rev := extend_full_int(input_lw_rev, in_wid, kp_wid, pos, 1)
	output_up_rev := comprs_full_fast_int(input_up_rev, in_wid, kp_wid, pos, 0)
	output_lw_rev := comprs_full_fast_int(input_lw_rev, in_wid, kp_wid, pos, 1)

	// "Now, to do extract_full_hf"

	for i := range output_up_rev {
		output[reverseBits(uint32(i), logN-1)] = output_up_rev[i]
		output[uint32(N/2)+reverseBits(uint32(i), logN-1)] = output_lw_rev[i]
	}

	for b := 0; b < batch*4; b++ {
		print_vec_int("output ("+strconv.Itoa(b)+")", output, in_wid/2, b)
	}
	// fmt.Println(output)
}

func testCyc(logN, iter int, printResult bool) {

	N := (1 << logN)
	ECD_LV := 1

	// Schemes parameters are created from scratch
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:     logN,
		LogQ:     []int{log_out_scale + log_c_scale, log_in_scale},
		LogP:     []int{60},
		Sigma:    rlwe.DefaultSigma,
		LogSlots: logN - 1,
		Scale:    float64(1 << log_in_scale),
	})
	if err != nil {
		panic(err)
	}

	fmt.Println()
	fmt.Println("=========================================")
	fmt.Println("         INSTANTIATING SCHEME            ")
	fmt.Println("=========================================")
	fmt.Println()

	start = time.Now()

	kgen := ckks.NewKeyGenerator(params)
	sk := kgen.GenSecretKey()

	encryptor := ckks.NewEncryptor(params, sk)
	decryptor := ckks.NewDecryptor(params, sk)
	encoder := ckks.NewEncoder(params)
	rlk := kgen.GenRelinearizationKey(sk, 2)
	evaluator := ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk})

	fmt.Printf("Done in %s \n", time.Since(start))

	fmt.Println()
	fmt.Printf("CKKS parameters: logN = %d, logSlots = %d, logQP = %d, levels = %d, scale= %f, sigma = %f \n",
		params.LogN(), params.LogSlots(), params.LogQP(), params.MaxLevel()+1, params.Scale(), params.Sigma())

	fmt.Println()
	fmt.Println("=========================================")
	fmt.Println("   PLAINTEXT CREATION & ENCRYPTION       ")
	fmt.Println("=========================================")
	fmt.Println()

	start = time.Now()

	name_iter := fmt.Sprintf("%04d", iter)

	circ_rows := readTxt("./variables/rows.txt", 0)
	circ_mat := make([][]float64, 8)
	plain_mat := make([]*ckks.Plaintext, 8)
	for i := 0; i < len(circ_mat); i++ {
		circ_mat[i] = encode_circ(circ_rows[N/2*i:N/2*(i+1)], 16, N)
		plain_mat[i] = ckks.NewPlaintext(params, ECD_LV, params.Scale())
		encoder.EncodeCoeffs(circ_mat[i], plain_mat[i])
		encoder.ToNTT(plain_mat[i])
	}

	test_input := readTxt("./inputs/vec_input_"+name_iter+".txt", 0)
	enc_test_input := make([]*ckks.Ciphertext, 8)
	test_tmp := ckks.NewPlaintext(params, ECD_LV, params.Scale())
	for i := 0; i < len(enc_test_input); i++ {
		encoder.EncodeCoeffs(encode_circ_in(test_input, i, 16, N), test_tmp)
		enc_test_input[i] = encryptor.EncryptNew(test_tmp)
	}

	fmt.Printf("Done in %s \n", time.Since(start))
	fmt.Println("vec size: ", N)
	fmt.Println("Input matrix: ")
	prt_vec(test_input)
	fmt.Println("Matrix rows (1st part): ")
	for i := 0; i < len(circ_mat); i++ {
		prt_vec(circ_rows[N/2*i : N/2*(i+1)])
	}

	fmt.Println()
	fmt.Println("=========================================")
	fmt.Println("              DECRYPTION                 ")
	fmt.Println("=========================================")
	fmt.Println()

	start = time.Now()
	var test_result *ckks.Ciphertext
	for i := 0; i < len(enc_test_input); i++ {
		if i == 0 {
			test_result = evaluator.MulNew(enc_test_input[i], plain_mat[i])
		} else {
			evaluator.Add(test_result, evaluator.MulNew(enc_test_input[i], plain_mat[i]), test_result)
		}
	}

	decryptor.Decrypt(test_result, test_tmp)
	test_out := encoder.DecodeCoeffs(test_tmp)
	fmt.Print("Result: \n")
	fmt.Println(test_out)

	fmt.Printf("Done in %s \n", time.Since(start))
}

func testImagenet_final_fast() {
	// Do not use fast_pack
	// We use full packing: i.e., in_wid**2 element is contained in po2_in_wid**2 sized block <-> half padding of Resnet
	// So ReLU, keep or rot, StoC done on both the 1st & 2nd part of the CtoS ciphertexts
	logN := 14
	num_blc := 3
	norm := 1
	raw_in_wids := []int{14, 7}  // same as python
	real_batch := []int{64, 128} // same as python
	fin_out_batch := 60
	py_bn_a := []float64{1.0, 0.2}
	ker_wid := 3
	iter := 2
	in_wids := make([]int, len(raw_in_wids))
	kp_wids := make([]int, len(raw_in_wids))
	for i, elt := range raw_in_wids {
		in_wids[i] = 16 / int(1<<i)
		kp_wids[i] = elt
	}
	cont := newContext(logN, ker_wid, in_wids, kp_wids, true, "Imagenet_final_fast")

	ker_size := ker_wid * ker_wid
	max_batch := make([]int, len(real_batch)) // the max batch
	for i := range max_batch {
		max_batch[i] = cont.N / (in_wids[i] * in_wids[i])
	}

	alpha := 0.0 // 0.3 => leakyrelu
	raw_input := make([]float64, raw_in_wids[0]*raw_in_wids[0]*real_batch[0])
	for i := range raw_input {
		raw_input[i] = 1.0 - 1.0*float64(i)/float64(len(raw_input))
	}

	input := make([]float64, in_wids[0]*in_wids[0]*real_batch[0])
	for i := 0; i < raw_in_wids[0]; i++ {
		for j := 0; j < raw_in_wids[0]; j++ {
			for b := 0; b < real_batch[0]; b++ {
				input[i*in_wids[0]*real_batch[0]+j*real_batch[0]+b] = raw_input[i*raw_in_wids[0]*real_batch[0]+j*real_batch[0]+b]
			}
		}
	}
	fmt.Println("Input: ")
	prt_mat(input, max_batch[0], 3)

	ker_in12 := make([]float64, real_batch[0]*real_batch[1]*ker_size)
	for i := range ker_in12 {
		ker_in12[i] = 0.1 * float64(i) / float64(len(ker_in12))
	}
	ker_in2 := make([]float64, real_batch[1]*real_batch[1]*ker_size)
	for i := range ker_in2 {
		ker_in2[i] = 0.1 * float64(i) / float64(len(ker_in2))
	}
	ker_inf := make([]float64, real_batch[1]*60)
	for i := range ker_inf {
		ker_inf[i] = 1.0 * float64(i) / float64(len(ker_inf))
	}
	bn_af := make([]float64, real_batch[1]*2)
	for i := range bn_af {
		bn_af[i] = 1.0 / (7 * 7) // for reduce mean on 8*8 elements
	}
	bn_bf := make([]float64, real_batch[1]*2)
	for i := range bn_bf {
		bn_bf[i] = 0.0 //10.0 * float64(i)
	}

	ker_in12_0 := make([]float64, len(ker_in12)/2)
	ker_in12_1 := make([]float64, len(ker_in12)/2)
	for k := 0; k < ker_size; k++ {
		for i := 0; i < real_batch[0]; i++ {
			for j := 0; j < real_batch[1]/2; j++ {
				ker_in12_0[k*real_batch[0]*real_batch[1]/2+(i*real_batch[1]/2+j)] = ker_in12[k*real_batch[0]*real_batch[1]+(i*real_batch[1]+j)]                 // [i][j]
				ker_in12_1[k*real_batch[0]*real_batch[1]/2+(i*real_batch[1]/2+j)] = ker_in12[k*real_batch[0]*real_batch[1]+(i*real_batch[1]+real_batch[1]/2+j)] // [i][j+B/2]
			}
		}
	}
	ker_inf_ := make([]float64, 7*7*real_batch[1]*fin_out_batch)
	for b := 0; b < 7*7; b++ {
		for i := 0; i < real_batch[1]; i++ {
			for j := 0; j < fin_out_batch; j++ {
				ker_inf_[b*real_batch[1]*fin_out_batch+i*fin_out_batch+j] = ker_inf[i*fin_out_batch+j]
			}
		}
	}

	bn_a12 := make([]float64, real_batch[1])
	bn_b12 := make([]float64, real_batch[1])
	for i := range bn_a12 {
		bn_a12[i] = py_bn_a[0] // * float64(i) / float64(batch)
		bn_b12[i] = 0.0        //0.1 * float64(i)
	}
	bn_a2 := make([]float64, real_batch[1])
	bn_b2 := make([]float64, real_batch[1])
	for i := range bn_a2 {
		bn_a2[i] = py_bn_a[1] // * float64(i) / float64(batch)
		bn_b2[i] = 0.0        //0.1 * float64(i)
	}

	bn_a12_0 := make([]float64, real_batch[1]/2)
	bn_a12_1 := make([]float64, real_batch[1]/2)
	bn_b12_0 := make([]float64, real_batch[1]/2)
	bn_b12_1 := make([]float64, real_batch[1]/2)
	for i := range bn_b12_0 {
		bn_a12_0[i] = bn_a12[i]
		bn_a12_1[i] = bn_a12[i+real_batch[1]/2]
		bn_b12_0[i] = bn_b12[i]
		bn_b12_1[i] = bn_b12[i+real_batch[1]/2]
	}

	fmt.Println("vec size: ", cont.N)
	fmt.Println("input width: ", raw_in_wids)
	fmt.Println("kernel width: ", ker_wid)
	fmt.Println("num batches: ", real_batch)

	start = time.Now()
	pl_input := ckks.NewPlaintext(cont.params, cont.ECD_LV, cont.params.Scale()) // contain plaintext values
	cont.encoder.EncodeCoeffs(input, pl_input)
	ct_input := cont.encryptor.EncryptNew(pl_input)
	fmt.Printf("Encryption done in %s \n", time.Since(start))

	timings := make([]float64, 3)
	begin_start := time.Now()
	// input to block 2
	new_start := time.Now()
	ct_result1 := evalConv_BNRelu_new(cont, ct_input, ker_in12_0, bn_a12_0, bn_b12_0, alpha, in_wids[0], kp_wids[0], ker_wid, real_batch[0], real_batch[0], norm, 0, iter, "StrConv", false, false)
	ct_result2 := evalConv_BNRelu_new(cont, ct_input, ker_in12_1, bn_a12_1, bn_b12_1, alpha, in_wids[0], kp_wids[0], ker_wid, real_batch[0], real_batch[0], norm, 1, iter, "StrConv", false, false)
	ct_result := cont.evaluator.AddNew(ct_result1, ct_result2)
	fmt.Println("Block1 to 2 done!")
	res_tmp := cont.encoder.DecodeCoeffs(cont.decryptor.DecryptNew(ct_result))
	prt_mat_norm(res_tmp, max_batch[1], 1, 4, false)

	timings[0] = time.Since(new_start).Seconds()
	new_start = time.Now()

	// Block 2
	ct_layer := ct_result
	prt_result := false
	for i := 1; i <= num_blc; i++ {
		if i == num_blc {
			prt_result = true
		}
		ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in2, bn_a2, bn_b2, alpha, in_wids[1], kp_wids[1], ker_wid, real_batch[1], real_batch[1], norm, 0, iter, "Conv", false, prt_result)
		fmt.Println("Block2, Layer ", i, "done!")
	}
	timings[1] = time.Since(new_start).Seconds()
	new_start = time.Now()

	// RMFC
	ct_result = evalConv_BN(cont, ct_layer, ker_inf_, bn_af, bn_bf, in_wids[1], 7, real_batch[1], 60, 1, float64(1<<30), true, false)

	timings[2] = time.Since(new_start).Seconds()
	new_start = time.Now()
	cont.decryptor.Decrypt(ct_result, pl_input)
	res_tmp = cont.encoder.DecodeCoeffs(pl_input)
	fmt.Printf("Decryption done in %s \n", time.Since(new_start))
	prt_mat_one_norm(res_tmp, max_batch[1], 1, 4, 4)

	fmt.Println("Blc1->2: ", timings[0], " sec")
	fmt.Println("Blc2: ", timings[1], " sec")
	fmt.Println("Final (reduce_mean & FC): ", timings[2], " sec")
	fmt.Printf("Total done in %s \n", time.Since(begin_start))
}

func testImagenet_final() {
	// We use full packing: i.e., in_wid**2 element is contained in po2_in_wid**2 sized block <-> half padding of Resnet
	// So ReLU, keep or rot, StoC done on both the 1st & 2nd part of the CtoS ciphertexts
	logN := 14
	num_blc := 3
	norm := 1
	raw_in_wids := []int{14, 7}  // same as python
	real_batch := []int{64, 128} // same as python
	fin_out_batch := 60
	py_bn_a := []float64{1.0, 0.2}
	ker_wid := 3
	iter := 2
	fast_pack := false
	in_wids := make([]int, len(raw_in_wids))
	kp_wids := make([]int, len(raw_in_wids))
	for i, elt := range raw_in_wids {
		in_wids[i] = 16 / int(1<<i)
		kp_wids[i] = elt
	}
	cont := newContext(logN, ker_wid, in_wids, kp_wids, true, "Imagenet_final")

	ker_size := ker_wid * ker_wid
	max_batch := make([]int, len(real_batch)) // the max batch
	for i := range max_batch {
		max_batch[i] = cont.N / (in_wids[i] * in_wids[i])
	}

	alpha := 0.0 // 0.3 => leakyrelu
	raw_input := make([]float64, raw_in_wids[0]*raw_in_wids[0]*real_batch[0])
	for i := range raw_input {
		raw_input[i] = 1.0 - 1.0*float64(i)/float64(len(raw_input))
	}

	input := make([]float64, in_wids[0]*in_wids[0]*real_batch[0])
	for i := 0; i < raw_in_wids[0]; i++ {
		for j := 0; j < raw_in_wids[0]; j++ {
			for b := 0; b < real_batch[0]; b++ {
				input[i*in_wids[0]*real_batch[0]+j*real_batch[0]+b] = raw_input[i*raw_in_wids[0]*real_batch[0]+j*real_batch[0]+b]
			}
		}
	}
	fmt.Println("Input: ")
	prt_mat(input, max_batch[0], 3)

	ker_in12 := make([]float64, real_batch[0]*real_batch[1]*ker_size)
	for i := range ker_in12 {
		ker_in12[i] = 0.1 * float64(i) / float64(len(ker_in12))
	}
	ker_in2 := make([]float64, real_batch[1]*real_batch[1]*ker_size)
	for i := range ker_in2 {
		ker_in2[i] = 0.1 * float64(i) / float64(len(ker_in2))
	}
	ker_inf := make([]float64, real_batch[1]*60)
	for i := range ker_inf {
		ker_inf[i] = 1.0 * float64(i) / float64(len(ker_inf))
	}
	bn_af := make([]float64, real_batch[1]*2)
	for i := range bn_af {
		bn_af[i] = 1.0 / (7 * 7) // for reduce mean on 8*8 elements
	}
	bn_bf := make([]float64, real_batch[1]*2)
	for i := range bn_bf {
		bn_bf[i] = 0.0 //10.0 * float64(i)
	}

	ker_in12_0 := make([]float64, len(ker_in12)/2)
	ker_in12_1 := make([]float64, len(ker_in12)/2)
	for k := 0; k < ker_size; k++ {
		for i := 0; i < real_batch[0]; i++ {
			for j := 0; j < real_batch[1]/2; j++ {
				ker_in12_0[k*real_batch[0]*real_batch[1]/2+(i*real_batch[1]/2+j)] = ker_in12[k*real_batch[0]*real_batch[1]+(i*real_batch[1]+2*j)]   // [i][2*j]
				ker_in12_1[k*real_batch[0]*real_batch[1]/2+(i*real_batch[1]/2+j)] = ker_in12[k*real_batch[0]*real_batch[1]+(i*real_batch[1]+2*j+1)] // [i][2*j+1]
			}
		}
	}
	ker_inf_ := make([]float64, 7*7*2*real_batch[1]*fin_out_batch)
	for b := 0; b < 7*7; b++ {
		for i := 0; i < real_batch[1]; i++ {
			for j := 0; j < fin_out_batch; j++ {
				ker_inf_[b*2*real_batch[1]*60+2*i*fin_out_batch+j] = ker_inf[i*fin_out_batch+j]
			}
		}
	}

	bn_a12 := make([]float64, real_batch[1])
	bn_b12 := make([]float64, real_batch[1])
	for i := range bn_a12 {
		bn_a12[i] = py_bn_a[0] // * float64(i) / float64(batch)
		bn_b12[i] = 0.0        //0.1 * float64(i)
	}
	bn_a2 := make([]float64, real_batch[1])
	bn_b2 := make([]float64, real_batch[1])
	for i := range bn_a2 {
		bn_a2[i] = py_bn_a[1] // * float64(i) / float64(batch)
		bn_b2[i] = 0.0        //0.1 * float64(i)
	}

	bn_a12_0 := make([]float64, real_batch[1]/2)
	bn_a12_1 := make([]float64, real_batch[1]/2)
	bn_b12_0 := make([]float64, real_batch[1]/2)
	bn_b12_1 := make([]float64, real_batch[1]/2)
	for i := range bn_b12_0 {
		bn_a12_0[i] = bn_a12[2*i]
		bn_a12_1[i] = bn_a12[2*i+1]
		bn_b12_0[i] = bn_b12[2*i]
		bn_b12_1[i] = bn_b12[2*i+1]
	}

	fmt.Println("vec size: ", cont.N)
	fmt.Println("input width: ", raw_in_wids)
	fmt.Println("kernel width: ", ker_wid)
	fmt.Println("num batches: ", real_batch)

	start = time.Now()
	pl_input := ckks.NewPlaintext(cont.params, cont.ECD_LV, cont.params.Scale()) // contain plaintext values
	cont.encoder.EncodeCoeffs(input, pl_input)
	ct_input := cont.encryptor.EncryptNew(pl_input)
	fmt.Printf("Encryption done in %s \n", time.Since(start))

	timings := make([]float64, 3)
	begin_start := time.Now()
	// input to block 2
	new_start := time.Now()
	ct_result1 := evalConv_BNRelu_new(cont, ct_input, ker_in12_0, bn_a12_0, bn_b12_0, alpha, in_wids[0], kp_wids[0], ker_wid, real_batch[0], real_batch[0], norm, 0, iter, "StrConv", true, false)
	ct_result2 := evalConv_BNRelu_new(cont, ct_input, ker_in12_1, bn_a12_1, bn_b12_1, alpha, in_wids[0], kp_wids[0], ker_wid, real_batch[0], real_batch[0], norm, 2, iter, "StrConv", true, false)
	ct_result := cont.evaluator.AddNew(ct_result1, ct_result2)
	fmt.Println("Block1 to 2 done!")
	res_tmp := cont.encoder.DecodeCoeffs(cont.decryptor.DecryptNew(ct_result))
	prt_mat_norm(res_tmp, max_batch[1], 2, 4, false)

	timings[0] = time.Since(new_start).Seconds()
	new_start = time.Now()

	// Block 2
	ct_layer := ct_result
	prt_result := false
	norm = 2
	for i := 1; i <= num_blc; i++ {
		if i == num_blc {
			prt_result = true
		}
		ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in2, bn_a2, bn_b2, alpha, in_wids[1], kp_wids[1], ker_wid, real_batch[1], real_batch[1], norm, 0, iter, "Conv", fast_pack, prt_result)
		fmt.Println("Block2, Layer ", i, "done!")
	}
	timings[1] = time.Since(new_start).Seconds()
	new_start = time.Now()

	// RMFC
	ct_result = evalConv_BN(cont, ct_layer, ker_inf_, bn_af, bn_bf, in_wids[1], 7, real_batch[1]*2, 60, 1, float64(1<<30), true, false)

	timings[2] = time.Since(new_start).Seconds()
	new_start = time.Now()
	cont.decryptor.Decrypt(ct_result, pl_input)
	res_tmp = cont.encoder.DecodeCoeffs(pl_input)
	fmt.Printf("Decryption done in %s \n", time.Since(new_start))
	prt_mat_one_norm(res_tmp, max_batch[1], 1, 4, 4)

	fmt.Println("Blc1->2: ", timings[0], " sec")
	fmt.Println("Blc2: ", timings[1], " sec")
	fmt.Println("Final (reduce_mean & FC): ", timings[2], " sec")
	fmt.Printf("Total done in %s \n", time.Since(begin_start))
}

func testImagenet() {
	// We use full packing: i.e., in_wid**2 element is contained in po2_in_wid**2 sized block <-> half padding of Resnet
	// So ReLU, keep or rot, StoC done on both the 1st & 2nd part of the CtoS ciphertexts
	logN := 12
	num_blc1 := 1
	num_blc2 := 1
	norm := 1
	raw_in_wids := []int{28, 14, 7} // same as python
	real_batch := []int{8, 16, 32}  // same as python
	num_ctxts := []int{2, 1, 1}
	py_bn_a := []float64{0.2, 0.2}
	ker_wid := 3
	iter := 2
	fast_pack := false
	in_wids := make([]int, len(raw_in_wids))
	kp_wids := make([]int, len(raw_in_wids))
	for i, elt := range raw_in_wids {
		in_wids[i] = 32 / int(1<<i)
		kp_wids[i] = elt
	}
	cont := newContext(logN, ker_wid, in_wids, kp_wids, true, "Imagenet")

	ker_size := ker_wid * ker_wid
	max_batch := make([]int, len(real_batch)) // the max batch
	for i := range max_batch {
		max_batch[i] = cont.N / (in_wids[i] * in_wids[i])
	}

	alpha := 0.0 // 0.3 => leakyrelu
	input := make([]float64, raw_in_wids[0]*raw_in_wids[0]*real_batch[0])
	for i := range input {
		input[i] = 1.0 // 1.0 - 1.0*float64(i)/float64(len(input))
	}
	input1 := make([]float64, cont.N)
	input2 := make([]float64, cont.N)
	for i := 0; i < in_wids[0]; i++ {
		for j := 0; j < in_wids[0]; j++ {
			for b := 0; b < real_batch[0]/2; b++ {
				if (i < raw_in_wids[0]) && (j < raw_in_wids[0]) {
					input1[i*in_wids[0]*max_batch[0]+j*max_batch[0]+b] = input[i*raw_in_wids[0]*real_batch[0]+j*real_batch[0]+b]
					input2[i*in_wids[0]*max_batch[0]+j*max_batch[0]+b] = input[i*raw_in_wids[0]*real_batch[0]+j*real_batch[0]+b+real_batch[0]/2]
				}
			}
		}
	}

	fmt.Println("Input1: ")
	prt_mat(input1, max_batch[0], 7)
	fmt.Println("Input2: ")
	prt_mat(input2, max_batch[0], 7)

	ker_in0 := make([]float64, real_batch[0]*real_batch[1]*ker_size)
	for i := range ker_in0 {
		ker_in0[i] = 0.1 // 0.1 * float64(i) / float64(len(ker_in0))
	}
	ker_in1 := make([]float64, real_batch[1]*real_batch[1]*ker_size)
	for i := range ker_in1 {
		ker_in1[i] = 0.1 * float64(i) / float64(len(ker_in1))
	}
	ker_in12 := make([]float64, real_batch[1]*real_batch[2]*ker_size)
	for i := range ker_in12 {
		ker_in12[i] = 0.1 * float64(i) / float64(len(ker_in12))
	}
	ker_in2 := make([]float64, real_batch[2]*real_batch[2]*ker_size)
	for i := range ker_in2 {
		ker_in2[i] = 0.1 * float64(i) / float64(len(ker_in2))
	}
	ker_inf := make([]float64, real_batch[2]*60)
	for i := range ker_inf {
		ker_inf[i] = 1.0 * float64(i) / float64(len(ker_inf))
	}
	bn_af := make([]float64, real_batch[2]*2)
	for i := range bn_af {
		bn_af[i] = 1.0 / (7 * 7) // for reduce mean on 8*8 elements
	}
	bn_bf := make([]float64, real_batch[2]*2)
	for i := range bn_bf {
		bn_bf[i] = 0.0 //10.0 * float64(i)
	}

	ker_in0_sep := make([][][]float64, 2*num_ctxts[0]) // number of output ctxts
	for out := 0; out < 2*num_ctxts[0]; out++ {
		ker_in0_sep[out] = make([][]float64, num_ctxts[0]) // number of input ctxts
		for in := 0; in < num_ctxts[0]; in++ {
			ker_in0_sep[out][in] = make([]float64, len(ker_in0)/(2*num_ctxts[0]*num_ctxts[0]))
			for k := 0; k < ker_size; k++ {
				for i := 0; i < real_batch[0]/num_ctxts[0]; i++ {
					for j := 0; j < real_batch[1]/(2*num_ctxts[0]); j++ {
						ker_in0_sep[out][in][k*real_batch[0]*real_batch[1]/(2*num_ctxts[0]*num_ctxts[0])+(i*real_batch[1]/(2*num_ctxts[0])+j)] =
							ker_in0[k*real_batch[0]*real_batch[1]+(i+in*real_batch[0]/num_ctxts[0])*real_batch[1]+j+out*real_batch[1]/(2*num_ctxts[0])] // [i][2*j]
					}
				}
			}
		}
	}
	ker_in12_0 := make([]float64, len(ker_in12)/2)
	ker_in12_1 := make([]float64, len(ker_in12)/2)
	for k := 0; k < ker_size; k++ {
		for i := 0; i < real_batch[1]; i++ {
			for j := 0; j < real_batch[2]/2; j++ {
				ker_in12_0[k*real_batch[1]*real_batch[2]/2+(i*real_batch[2]/2+j)] = ker_in12[k*real_batch[1]*real_batch[2]+(i*real_batch[2]+2*j)]   // [i][2*j]
				ker_in12_1[k*real_batch[1]*real_batch[2]/2+(i*real_batch[2]/2+j)] = ker_in12[k*real_batch[1]*real_batch[2]+(i*real_batch[2]+2*j+1)] // [i][2*j+1]
			}
		}
	}
	ker_inf_ := make([]float64, 7*7*2*real_batch[2]*60)
	for b := 0; b < 7*7; b++ {
		for i := 0; i < real_batch[2]; i++ {
			for j := 0; j < 60; j++ {
				ker_inf_[b*2*real_batch[2]*60+2*i*60+j] = ker_inf[i*60+j]
			}
		}
	}

	bn_a0 := make([]float64, real_batch[1])
	bn_b0 := make([]float64, real_batch[1])
	for i := range bn_a0 {
		bn_a0[i] = py_bn_a[0] // * float64(i) / float64(batch)
		bn_b0[i] = 0.0        //0.1 * float64(i) // float64(real_batch[0])
	}
	bn_a1 := make([]float64, real_batch[1])
	bn_b1 := make([]float64, real_batch[1])
	for i := range bn_a1 {
		bn_a1[i] = py_bn_a[0] // * float64(i) / float64(batch)
		bn_b1[i] = 0.0        //0.1 * float64(i)
	}
	bn_a12 := make([]float64, real_batch[2])
	bn_b12 := make([]float64, real_batch[2])
	for i := range bn_a12 {
		bn_a12[i] = py_bn_a[1] // * float64(i) / float64(batch)
		bn_b12[i] = 0.0        //0.1 * float64(i)
	}
	bn_a2 := make([]float64, real_batch[2])
	bn_b2 := make([]float64, real_batch[2])
	for i := range bn_a2 {
		bn_a2[i] = py_bn_a[1] // * float64(i) / float64(batch)
		bn_b2[i] = 0.0        //0.1 * float64(i)
	}

	bn_a0_sep := make([][]float64, 2*num_ctxts[0])
	bn_b0_sep := make([][]float64, 2*num_ctxts[0])
	for out := 0; out < 2*num_ctxts[0]; out++ {
		bn_a0_sep[out] = make([]float64, real_batch[1]/4)
		bn_b0_sep[out] = make([]float64, real_batch[1]/4)
		for i := 0; i < real_batch[1]/4; i++ {
			bn_a0_sep[out][i] = bn_a0[i+out*real_batch[1]/4]
			bn_b0_sep[out][i] = bn_b0[i+out*real_batch[1]/4]
		}
	}
	bn_a12_0 := make([]float64, real_batch[2]/2)
	bn_a12_1 := make([]float64, real_batch[2]/2)
	bn_b12_0 := make([]float64, real_batch[2]/2)
	bn_b12_1 := make([]float64, real_batch[2]/2)
	for i := range bn_b12_0 {
		bn_a12_0[i] = bn_a12[2*i]
		bn_a12_1[i] = bn_a12[2*i+1]
		bn_b12_0[i] = bn_b12[2*i]
		bn_b12_1[i] = bn_b12[2*i+1]
	}

	fmt.Println("vec size: ", cont.N)
	fmt.Println("input width: ", raw_in_wids)
	fmt.Println("kernel width: ", ker_wid)
	fmt.Println("num batches: ", real_batch)
	fmt.Println("Input matrix: ")
	prt_vec(input)

	start = time.Now()
	ct_input := make([]*ckks.Ciphertext, 2)
	pl_input := ckks.NewPlaintext(cont.params, cont.ECD_LV, cont.params.Scale()) // contain plaintext values
	cont.encoder.EncodeCoeffs(input1, pl_input)
	ct_input[0] = cont.encryptor.EncryptNew(pl_input)
	cont.encoder.EncodeCoeffs(input2, pl_input)
	ct_input[1] = cont.encryptor.EncryptNew(pl_input)
	fmt.Printf("Encryption done in %s \n", time.Since(start))

	timings := make([]float64, 5)
	begin_start := time.Now()
	new_start := time.Now()
	// input to block 1
	prt_result := false
	var ct_result *ckks.Ciphertext
	for pos := 0; pos < 4; pos++ {
		ct_tmp := cont.evaluator.AddNew(evalConv_BNRelu_new(cont, ct_input[0], ker_in0_sep[pos][0], bn_a0_sep[pos], bn_b0_sep[pos], alpha, in_wids[0], kp_wids[0], ker_wid, max_batch[0], max_batch[0], norm, pos, iter, "StrConv", fast_pack, prt_result),
			evalConv_BNRelu_new(cont, ct_input[1], ker_in0_sep[pos][1], bn_a0_sep[pos], bn_b0_sep[pos], alpha, in_wids[0], kp_wids[0], ker_wid, max_batch[0], max_batch[0], norm, pos, iter, "StrConv", fast_pack, prt_result))
		if pos == 0 {
			ct_result = ct_tmp
		} else {
			ct_result = cont.evaluator.AddNew(ct_result, ct_tmp)
		}
	}
	timings[0] = time.Since(new_start).Seconds()
	fmt.Println("Block0 to 1 done!")
	res_ttmp := cont.encoder.DecodeCoeffs(cont.decryptor.DecryptNew(ct_result))
	prt_mat_norm(res_ttmp, max_batch[1], 1, 0, false)

	new_start = time.Now()

	// Block 1
	ct_layer := ct_result
	prt_result = false
	for i := 1; i <= num_blc1; i++ {
		if i == num_blc1 {
			prt_result = true
		}
		ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in1, bn_a1, bn_b1, alpha, in_wids[1], kp_wids[1], ker_wid, real_batch[1], real_batch[1], norm, 0, iter, "Conv", fast_pack, prt_result)
		fmt.Println("Block1, Layer ", i, "done!")
	}
	fmt.Println("Block1 done.")
	timings[1] = time.Since(new_start).Seconds()
	new_start = time.Now()

	ct_result1 := evalConv_BNRelu_new(cont, ct_layer, ker_in12_0, bn_a12_0, bn_b12_0, alpha, in_wids[1], kp_wids[1], ker_wid, real_batch[1], real_batch[1], norm, 0, iter, "StrConv", true, false)
	ct_result2 := evalConv_BNRelu_new(cont, ct_layer, ker_in12_1, bn_a12_1, bn_b12_1, alpha, in_wids[1], kp_wids[1], ker_wid, real_batch[1], real_batch[1], norm, 2, iter, "StrConv", true, false)
	ct_result = cont.evaluator.AddNew(ct_result1, ct_result2)
	fmt.Println("Block1 to 2 done!")
	res_ttmp = cont.encoder.DecodeCoeffs(cont.decryptor.DecryptNew(ct_result))
	prt_mat_norm(res_ttmp, max_batch[2], 2, 3, false)

	timings[2] = time.Since(new_start).Seconds()
	new_start = time.Now()

	// Block 2
	ct_layer = ct_result
	prt_result = false
	norm = 2
	for i := 1; i <= num_blc2; i++ {
		if i == num_blc2 {
			prt_result = true
		}
		ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in2, bn_a2, bn_b2, alpha, in_wids[2], kp_wids[2], ker_wid, real_batch[2], real_batch[2], norm, 0, iter, "Conv", fast_pack, prt_result)
		fmt.Println("Block2, Layer ", i, "done!")
	}
	timings[3] = time.Since(new_start).Seconds()
	new_start = time.Now()

	// RMFC
	ct_result = evalConv_BN(cont, ct_layer, ker_inf_, bn_af, bn_bf, in_wids[2], 7, real_batch[2]*2, 60, 1, float64(1<<30), true, false)

	timings[4] = time.Since(new_start).Seconds()
	new_start = time.Now()
	cont.decryptor.Decrypt(ct_result, pl_input)
	res_tmp := cont.encoder.DecodeCoeffs(pl_input)
	fmt.Printf("Decryption done in %s \n", time.Since(new_start))
	prt_mat_one_norm(res_tmp, max_batch[2], 1, 4, 4)
	// prt_mat_norm(res_tmp, max_batch[2], 1, 0, false)

	fmt.Println("Blc0->1: ", timings[0], " sec")
	fmt.Println("Blc1: ", timings[1], " sec")
	fmt.Println("Blc1->2: ", timings[2], " sec")
	fmt.Println("Blc2: ", timings[3], " sec")
	fmt.Println("Final (reduce_mean & FC): ", timings[4], " sec")
	fmt.Printf("Total done in %s \n", time.Since(begin_start))
}

func testImagenet_final_fast_in(st, end int) {
	// We use full packing: i.e., in_wid**2 element is contained in po2_in_wid**2 sized block <-> half padding of Resnet
	// So ReLU, keep or rot, StoC done on both the 1st & 2nd part of the CtoS ciphertexts
	ker_name := "ker5" // "ker5"
	ker_wid := 5
	weight_dir := "weight_imgnet_" + ker_name + "_h5/"
	logN := 16
	raw_in_wids := []int{14, 7}   // same as python
	real_batch := []int{256, 512} // same as python
	iter := 2
	in_wids := make([]int, len(raw_in_wids))
	kp_wids := make([]int, len(raw_in_wids))
	var num_blc int
	var st_weight_num int
	if ker_name == "ker3" {
		for i, elt := range raw_in_wids {
			in_wids[i] = 16 / int(1<<i)
			kp_wids[i] = elt
		}
		num_blc = 3
		st_weight_num = 13
	} else if ker_name == "ker5" {
		in_wids[0] = 16
		in_wids[1] = 8
		kp_wids[0] = 12
		kp_wids[1] = 6
		num_blc = 1
		st_weight_num = 11
	} else {
		panic("strange ker name!")
	}
	cont := newContext(logN, ker_wid, in_wids, kp_wids, true, "Imagenet_final_fast")

	ker_size := ker_wid * ker_wid
	max_batch := make([]int, len(real_batch)) // the max batch
	for i := range max_batch {
		max_batch[i] = cont.N / (in_wids[i] * in_wids[i])
	}
	alpha := 0.0 // 0.3 => leakyrelu

	for name_iter := st; name_iter < end; name_iter++ {
		weight_num := st_weight_num
		norm := 1
		fmt.Println("Start ", name_iter, "-th iter..")

		raw_input := readTxt("./Imagenet/"+ker_name+"_ct_in_one2/input_"+strconv.Itoa(name_iter)+".csv", raw_in_wids[0]*raw_in_wids[0]*real_batch[0])
		input := make([]float64, in_wids[0]*in_wids[0]*real_batch[0])
		for i := 0; i < raw_in_wids[0]; i++ {
			for j := 0; j < raw_in_wids[0]; j++ {
				for b := 0; b < real_batch[0]; b++ {
					input[i*in_wids[0]*real_batch[0]+j*real_batch[0]+b] = raw_input[i*raw_in_wids[0]*real_batch[0]+j*real_batch[0]+b]
				}
			}
		}
		fmt.Println("Input: ")
		prt_mat(input, max_batch[0], 3)

		ker_in12 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-conv.csv", real_batch[0]*real_batch[1]*ker_size)
		weight_num++
		bn_a12 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-a.csv", real_batch[1])
		bn_b12 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-b.csv", real_batch[1])

		ker_in12_0 := make([]float64, len(ker_in12)/2)
		ker_in12_1 := make([]float64, len(ker_in12)/2)
		for k := 0; k < ker_size; k++ {
			for i := 0; i < real_batch[0]; i++ {
				for j := 0; j < real_batch[1]/2; j++ {
					ker_in12_0[k*real_batch[0]*real_batch[1]/2+(i*real_batch[1]/2+j)] = ker_in12[k*real_batch[0]*real_batch[1]+(i*real_batch[1]+j)]                 // [i][j]
					ker_in12_1[k*real_batch[0]*real_batch[1]/2+(i*real_batch[1]/2+j)] = ker_in12[k*real_batch[0]*real_batch[1]+(i*real_batch[1]+real_batch[1]/2+j)] // [i][j+B/2]
				}
			}
		}
		bn_a12_0 := make([]float64, real_batch[1]/2)
		bn_a12_1 := make([]float64, real_batch[1]/2)
		bn_b12_0 := make([]float64, real_batch[1]/2)
		bn_b12_1 := make([]float64, real_batch[1]/2)
		for i := range bn_b12_0 {
			bn_a12_0[i] = bn_a12[i]
			bn_a12_1[i] = bn_a12[i+real_batch[1]/2]
			bn_b12_0[i] = bn_b12[i]
			bn_b12_1[i] = bn_b12[i+real_batch[1]/2]
		}

		fmt.Println("vec size: ", cont.N)
		fmt.Println("input width: ", raw_in_wids)
		fmt.Println("kernel width: ", ker_wid)
		fmt.Println("num batches: ", real_batch)

		start = time.Now()
		pl_input := ckks.NewPlaintext(cont.params, cont.ECD_LV, cont.params.Scale()) // contain plaintext values
		cont.encoder.EncodeCoeffs(input, pl_input)
		ct_input := cont.encryptor.EncryptNew(pl_input)
		fmt.Printf("Encryption done in %s \n", time.Since(start))

		timings := make([]float64, 3)
		begin_start := time.Now()
		// input to block 2
		new_start := time.Now()
		ct_result1 := evalConv_BNRelu_new(cont, ct_input, ker_in12_0, bn_a12_0, bn_b12_0, alpha, in_wids[0], kp_wids[0], ker_wid, real_batch[0], real_batch[0], norm, 0, iter, "StrConv", false, false)
		ct_result2 := evalConv_BNRelu_new(cont, ct_input, ker_in12_1, bn_a12_1, bn_b12_1, alpha, in_wids[0], kp_wids[0], ker_wid, real_batch[0], real_batch[0], norm, 1, iter, "StrConv", false, false)
		ct_result := cont.evaluator.AddNew(ct_result1, ct_result2)
		fmt.Println("Block1 to 2 done!")
		res_tmp := cont.encoder.DecodeCoeffs(cont.decryptor.DecryptNew(ct_result))
		prt_mat_norm(res_tmp, max_batch[1], 1, 4, false)

		timings[0] = time.Since(new_start).Seconds()
		new_start = time.Now()

		// Block 2
		ct_layer := ct_result
		prt_result := false
		for i := 1; i <= num_blc; i++ {
			ker_in2 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-conv.csv", real_batch[1]*real_batch[1]*ker_size)
			weight_num++
			bn_a2 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-a.csv", real_batch[1])
			bn_b2 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-b.csv", real_batch[1])
			if i == num_blc {
				prt_result = true
			}
			ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in2, bn_a2, bn_b2, alpha, in_wids[1], kp_wids[1], ker_wid, real_batch[1], real_batch[1], norm, 0, iter, "Conv", false, prt_result)
			fmt.Println("Block2, Layer ", i, "done!")
		}
		timings[1] = time.Since(new_start).Seconds()
		new_start = time.Now()

		// RMFC
		fin_out_batch := 1000
		ker_inf := readTxt(weight_dir+"fc.csv", real_batch[1]*fin_out_batch)
		bn_af := make([]float64, real_batch[1]*2)
		if ker_wid == 3 {
			for i := range bn_af {
				bn_af[i] = 1.0 / (7 * 7) // for reduce mean on 8*8 elements
			}
		} else {
			for i := range bn_af {
				bn_af[i] = 1.0 / (6 * 6) // for reduce mean on 8*8 elements
			}
		}
		bn_bf := make([]float64, real_batch[1]*2)
		for i := range bn_bf {
			bn_bf[i] = 0.0 //10.0 * float64(i)
		}
		ker_inf_ := make([]float64, 7*7*real_batch[1]*fin_out_batch)
		for b := 0; b < 7*7; b++ {
			for i := 0; i < real_batch[1]; i++ {
				for j := 0; j < fin_out_batch; j++ {
					ker_inf_[b*real_batch[1]*fin_out_batch+i*fin_out_batch+j] = ker_inf[i*fin_out_batch+j]
				}
			}
		}
		ct_result = evalConv_BN(cont, ct_layer, ker_inf_, bn_af, bn_bf, in_wids[1], 7, real_batch[1], 1000, 1, float64(1<<30), true, false)

		timings[2] = time.Since(new_start).Seconds()
		new_start = time.Now()
		cont.decryptor.Decrypt(ct_result, pl_input)
		res_tmp = cont.encoder.DecodeCoeffs(pl_input)
		fmt.Printf("Decryption done in %s \n", time.Since(new_start))
		final_result := prt_mat_one_norm(res_tmp, max_batch[1], 1, 4, 4)
		writeTxt("imgnet_class_result_"+ker_name+"_final_fast/imgnet_enc_class_result_"+ker_name+"_"+strconv.Itoa(name_iter)+".csv", final_result[:1000])

		fmt.Println("Blc1->2: ", timings[0], " sec")
		fmt.Println("Blc2: ", timings[1], " sec")
		fmt.Println("Final (reduce_mean & FC): ", timings[2], " sec")
		fmt.Printf("Total done in %s \n", time.Since(begin_start))
	}
}

func testImagenet_in(st, end int) {
	// We use full packing: i.e., in_wid**2 element is contained in po2_in_wid**2 sized block <-> half padding of Resnet
	// So ReLU, keep or rot, StoC done on both the 1st & 2nd part of the CtoS ciphertexts
	ker_name := "ker3"
	weight_dir := "weight_imgnet_ker3_h5/"
	logN := 16
	num_blc1 := 3
	num_blc2 := 3
	raw_in_wids := []int{28, 14, 7}    // same as python
	real_batch := []int{128, 256, 512} // same as python
	num_ctxts := []int{2, 1, 1}
	ker_wid := 3
	iter := 2
	fast_pack := false
	in_wids := make([]int, len(raw_in_wids))
	kp_wids := make([]int, len(raw_in_wids))
	for i, elt := range raw_in_wids {
		in_wids[i] = 32 / int(1<<i)
		kp_wids[i] = elt
	}
	cont := newContext(logN, ker_wid, in_wids, kp_wids, true, "Imagenet")

	ker_size := ker_wid * ker_wid
	max_batch := make([]int, len(real_batch)) // the max batch
	for i := range max_batch {
		max_batch[i] = cont.N / (in_wids[i] * in_wids[i])
	}
	alpha := 0.0 // 0.3 => leakyrelu

	for name_iter := st; name_iter < end; name_iter++ {
		norm := 1
		fmt.Println("Start ", name_iter, "-th iter..")
		input := readTxt("./Imagenet/ker3_ct_in_one/input_"+strconv.Itoa(name_iter)+".csv", raw_in_wids[0]*raw_in_wids[0]*real_batch[0])
		input1 := make([]float64, cont.N)
		input2 := make([]float64, cont.N)
		for i := 0; i < in_wids[0]; i++ {
			for j := 0; j < in_wids[0]; j++ {
				for b := 0; b < real_batch[0]/2; b++ {
					if (i < raw_in_wids[0]) && (j < raw_in_wids[0]) {
						input1[i*in_wids[0]*max_batch[0]+j*max_batch[0]+b] = input[i*raw_in_wids[0]*real_batch[0]+j*real_batch[0]+b]
						input2[i*in_wids[0]*max_batch[0]+j*max_batch[0]+b] = input[i*raw_in_wids[0]*real_batch[0]+j*real_batch[0]+b+real_batch[0]/2]
					}
				}
			}
		}

		fmt.Println("Input1: ")
		prt_mat(input1, max_batch[0], 3)
		fmt.Println("Input2: ")
		prt_mat(input2, max_batch[0], 3)

		weight_num := 9 // starting from w9-conv.csv
		ker_in0 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-conv.csv", real_batch[0]*real_batch[1]*ker_size)
		weight_num++
		bn_a0 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-a.csv", real_batch[1])
		bn_b0 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-b.csv", real_batch[1])

		ker_in0_sep := make([][][]float64, 2*num_ctxts[0]) // number of output ctxts
		for out := 0; out < 2*num_ctxts[0]; out++ {
			ker_in0_sep[out] = make([][]float64, num_ctxts[0]) // number of input ctxts
			for in := 0; in < num_ctxts[0]; in++ {
				ker_in0_sep[out][in] = make([]float64, len(ker_in0)/(2*num_ctxts[0]*num_ctxts[0]))
				for k := 0; k < ker_size; k++ {
					for i := 0; i < real_batch[0]/num_ctxts[0]; i++ {
						for j := 0; j < real_batch[1]/(2*num_ctxts[0]); j++ {
							ker_in0_sep[out][in][k*real_batch[0]*real_batch[1]/(2*num_ctxts[0]*num_ctxts[0])+(i*real_batch[1]/(2*num_ctxts[0])+j)] =
								ker_in0[k*real_batch[0]*real_batch[1]+(i+in*real_batch[0]/num_ctxts[0])*real_batch[1]+j+out*real_batch[1]/(2*num_ctxts[0])] // [i][2*j]
						}
					}
				}
			}
		}
		bn_a0_sep := make([][]float64, 2*num_ctxts[0])
		bn_b0_sep := make([][]float64, 2*num_ctxts[0])
		for out := 0; out < 2*num_ctxts[0]; out++ {
			bn_a0_sep[out] = make([]float64, real_batch[1]/4)
			bn_b0_sep[out] = make([]float64, real_batch[1]/4)
			for i := 0; i < real_batch[1]/4; i++ {
				bn_a0_sep[out][i] = bn_a0[i+out*real_batch[1]/4]
				bn_b0_sep[out][i] = bn_b0[i+out*real_batch[1]/4]
			}
		}

		fmt.Println("vec size: ", cont.N)
		fmt.Println("input width: ", raw_in_wids)
		fmt.Println("kernel width: ", ker_wid)
		fmt.Println("num batches: ", real_batch)

		start = time.Now()
		ct_input := make([]*ckks.Ciphertext, 2)
		pl_input := ckks.NewPlaintext(cont.params, cont.ECD_LV, cont.params.Scale()) // contain plaintext values
		cont.encoder.EncodeCoeffs(input1, pl_input)
		ct_input[0] = cont.encryptor.EncryptNew(pl_input)
		cont.encoder.EncodeCoeffs(input2, pl_input)
		ct_input[1] = cont.encryptor.EncryptNew(pl_input)
		fmt.Printf("Encryption done in %s \n", time.Since(start))

		timings := make([]float64, 5)
		begin_start := time.Now()
		new_start := time.Now()
		// input to block 1
		prt_result := false
		var ct_result *ckks.Ciphertext
		for pos := 0; pos < 4; pos++ {
			ct_tmp := cont.evaluator.AddNew(evalConv_BNRelu_new(cont, ct_input[0], ker_in0_sep[pos][0], bn_a0_sep[pos], bn_b0_sep[pos], alpha, in_wids[0], kp_wids[0], ker_wid, max_batch[0], max_batch[0], norm, pos, iter, "StrConv", fast_pack, prt_result),
				evalConv_BNRelu_new(cont, ct_input[1], ker_in0_sep[pos][1], bn_a0_sep[pos], bn_b0_sep[pos], alpha, in_wids[0], kp_wids[0], ker_wid, max_batch[0], max_batch[0], norm, pos, iter, "StrConv", fast_pack, prt_result))
			if pos == 0 {
				ct_result = ct_tmp
			} else {
				ct_result = cont.evaluator.AddNew(ct_result, ct_tmp)
			}
		}
		timings[0] = time.Since(new_start).Seconds()
		fmt.Println("Block0 to 1 done!")
		res_ttmp := cont.encoder.DecodeCoeffs(cont.decryptor.DecryptNew(ct_result))
		prt_mat_norm(res_ttmp, max_batch[1], 1, 0, false)

		new_start = time.Now()

		// Block 1
		ct_layer := ct_result
		prt_result = false
		for i := 1; i <= num_blc1; i++ {
			ker_in1 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-conv.csv", real_batch[1]*real_batch[1]*ker_size)
			weight_num++
			bn_a1 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-a.csv", real_batch[1])
			bn_b1 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-b.csv", real_batch[1])

			if i == num_blc1 {
				prt_result = true
			}
			ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in1, bn_a1, bn_b1, alpha, in_wids[1], kp_wids[1], ker_wid, real_batch[1], real_batch[1], norm, 0, iter, "Conv", fast_pack, prt_result)
			fmt.Println("Block1, Layer ", i, "done!")
		}
		fmt.Println("Block1 done.")
		timings[1] = time.Since(new_start).Seconds()
		new_start = time.Now()

		// block1 to block2
		ker_in12 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-conv.csv", real_batch[1]*real_batch[2]*ker_size)
		weight_num++
		ker_in12_0 := make([]float64, len(ker_in12)/2)
		ker_in12_1 := make([]float64, len(ker_in12)/2)
		for k := 0; k < ker_size; k++ {
			for i := 0; i < real_batch[1]; i++ {
				for j := 0; j < real_batch[2]/2; j++ {
					ker_in12_0[k*real_batch[1]*real_batch[2]/2+(i*real_batch[2]/2+j)] = ker_in12[k*real_batch[1]*real_batch[2]+(i*real_batch[2]+2*j)]   // [i][2*j]
					ker_in12_1[k*real_batch[1]*real_batch[2]/2+(i*real_batch[2]/2+j)] = ker_in12[k*real_batch[1]*real_batch[2]+(i*real_batch[2]+2*j+1)] // [i][2*j+1]
				}
			}
		}
		bn_a12 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-a.csv", real_batch[2])
		bn_b12 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-b.csv", real_batch[2])

		bn_a12_0 := make([]float64, real_batch[2]/2)
		bn_a12_1 := make([]float64, real_batch[2]/2)
		bn_b12_0 := make([]float64, real_batch[2]/2)
		bn_b12_1 := make([]float64, real_batch[2]/2)
		for i := range bn_b12_0 {
			bn_a12_0[i] = bn_a12[2*i]
			bn_a12_1[i] = bn_a12[2*i+1]
			bn_b12_0[i] = bn_b12[2*i]
			bn_b12_1[i] = bn_b12[2*i+1]
		}

		ct_result1 := evalConv_BNRelu_new(cont, ct_layer, ker_in12_0, bn_a12_0, bn_b12_0, alpha, in_wids[1], kp_wids[1], ker_wid, real_batch[1], real_batch[1], norm, 0, iter, "StrConv", true, false)
		ct_result2 := evalConv_BNRelu_new(cont, ct_layer, ker_in12_1, bn_a12_1, bn_b12_1, alpha, in_wids[1], kp_wids[1], ker_wid, real_batch[1], real_batch[1], norm, 2, iter, "StrConv", true, false)
		ct_result = cont.evaluator.AddNew(ct_result1, ct_result2)
		fmt.Println("Block1 to 2 done!")
		res_ttmp = cont.encoder.DecodeCoeffs(cont.decryptor.DecryptNew(ct_result))
		prt_mat_norm(res_ttmp, max_batch[2], 2, 3, false)

		timings[2] = time.Since(new_start).Seconds()
		new_start = time.Now()

		// Block 2
		ct_layer = ct_result
		prt_result = false
		norm = 2
		for i := 1; i <= num_blc2; i++ {
			ker_in2 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-conv.csv", real_batch[2]*real_batch[2]*ker_size)
			weight_num++
			bn_a2 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-a.csv", real_batch[2])
			bn_b2 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-b.csv", real_batch[2])

			if i == num_blc2 {
				prt_result = true
			}
			ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in2, bn_a2, bn_b2, alpha, in_wids[2], kp_wids[2], ker_wid, real_batch[2], real_batch[2], norm, 0, iter, "Conv", fast_pack, prt_result)
			fmt.Println("Block2, Layer ", i, "done!")
		}
		timings[3] = time.Since(new_start).Seconds()
		new_start = time.Now()

		// RMFC
		ker_inf := readTxt(weight_dir+"fc.csv", real_batch[2]*1000)
		bn_af := make([]float64, 1000)
		for i := range bn_af {
			bn_af[i] = 1.0 / (7 * 7) // for reduce mean on 8*8 elements
		}
		bn_bf := make([]float64, 1000)
		for i := range bn_bf {
			bn_bf[i] = 0.0 //10.0 * float64(i)
		}
		ker_inf_ := make([]float64, 7*7*2*real_batch[2]*1000)
		for b := 0; b < 7*7; b++ {
			for i := 0; i < real_batch[2]; i++ {
				for j := 0; j < 1000; j++ {
					ker_inf_[b*2*real_batch[2]*1000+2*i*1000+j] = ker_inf[i*1000+j]
				}
			}
		}

		ct_result = evalConv_BN(cont, ct_layer, ker_inf_, bn_af, bn_bf, in_wids[2], 7, real_batch[2]*2, 1000, 1, float64(1<<30), true, false)

		timings[4] = time.Since(new_start).Seconds()
		new_start = time.Now()
		cont.decryptor.Decrypt(ct_result, pl_input)
		res_tmp := cont.encoder.DecodeCoeffs(pl_input)
		fmt.Printf("Decryption done in %s \n", time.Since(new_start))
		final_result := prt_mat_one_norm(res_tmp, max_batch[2], 1, 4, 4)
		writeTxt("imgnet_class_result_"+ker_name+"/imgnet_enc_class_result_"+ker_name+"_"+strconv.Itoa(name_iter)+".csv", final_result[:1000])

		fmt.Println("Blc0->1: ", timings[0], " sec")
		fmt.Println("Blc1: ", timings[1], " sec")
		fmt.Println("Blc1->2: ", timings[2], " sec")
		fmt.Println("Blc2: ", timings[3], " sec")
		fmt.Println("Final (reduce_mean & FC): ", timings[4], " sec")
		fmt.Printf("Total done in %s \n", time.Since(begin_start))
	}
}

// Eval Conv & Boot
// func testBootFast_Conv(ext_input []int, logN, in_wid, ker_wid int, printResult bool) []float64 {

// 	N := (1 << logN)
// 	in_size := in_wid * in_wid
// 	batch := N / in_size
// 	ker_size := ker_wid * ker_wid
// 	ECD_LV := 3
// 	pos := 0

// 	fmt.Println()
// 	fmt.Println("=========================================")
// 	fmt.Println("    BOOTSTRAPP		then 	Rotation      ")
// 	fmt.Println("=========================================")
// 	fmt.Println()

// 	var btp *ckks.Bootstrapper

// 	// Bootstrapping parameters
// 	// LogSlots is hardcoded to 15 in the parameters, but can be changed from 1 to 15.
// 	// When changing logSlots make sure that the number of levels allocated to CtS and StC is
// 	// smaller or equal to logSlots.
// 	btpParams := ckks.DefaultBootstrapParams[6]
// 	params, err := btpParams.Params()
// 	if err != nil {
// 		panic(err)
// 	}
// 	fmt.Printf("CKKS parameters: logN = %d, logSlots = %d, h = %d, logQP = %d, levels = %d, scale= 2^%f, sigma = %f \n",
// 		params.LogN(), params.LogSlots(), btpParams.H, params.LogQP(), params.QCount(), math.Log2(params.Scale()), params.Sigma())

// 	// Generate rotations for EXT_FULL
// 	r_idx, m_idx := gen_extend_full_nhf(N/2, 2*in_wid, pos, true, true)
// 	var rotations []int
// 	for k := range r_idx {
// 		rotations = append(rotations, k)
// 	}
// 	for k := range m_idx {
// 		rotations = append(rotations, k)
// 	}
// 	// rotations = append(rotations, 1)
// 	fmt.Println("Rotations: ", rotations)

// 	// Scheme context and keys for evaluation (no Boot)
// 	kgen := ckks.NewKeyGenerator(params)
// 	sk, _ := kgen.GenKeyPairSparse(btpParams.H)
// 	rlk := kgen.GenRelinearizationKey(sk, 2)
// 	rotkeys := kgen.GenRotationKeysForRotations(rotations, false, sk)
// 	encoder := ckks.NewEncoder(params)
// 	decryptor := ckks.NewDecryptor(params, sk)
// 	encryptor := ckks.NewEncryptor(params, sk)
// 	evaluator := ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk, Rtks: rotkeys})

// 	plain_idx, pack_evaluator := gen_idxNlogs(ECD_LV, kgen, sk, encoder, params)

// 	input := make([]float64, N)
// 	for i := range input {
// 		input[i] = 1.0 * float64(ext_input[i]) / float64(N)
// 		// input[i] = 1.0 * float64(i) / float64(N)
// 	}

// 	int_tmp := make([]int, N)
// 	for i := range input {
// 		int_tmp[i] = int(float64(N) * input[i])
// 	}
// 	if printResult {
// 		fmt.Print("Input: \n")

// 		for b := 0; b < batch/4; b++ {
// 			print_vec_int("input ("+strconv.Itoa(b)+")", int_tmp, 2*in_wid, b)
// 		}
// 	}

// 	batch_real := batch / 16 // num batches at convolution 		// strided conv -> /(4*4)
// 	in_wid_out := in_wid * 4 // size of in_wid at convolution 	// strided conv -> *4

// 	ker1_in := make([]float64, batch_real*batch_real*ker_size)
// 	for i := range ker1_in {
// 		ker1_in[i] = 1.0 * (float64(len(ker1_in)) - float64(i) - 1) // float64(len(ker1_in))
// 	}

// 	ker1 := reshape_ker(ker1_in, ker_size, batch_real, false)

// 	pl_ker := make([]*ckks.Plaintext, batch_real)
// 	for i := 0; i < batch_real; i++ {
// 		pl_ker[i] = ckks.NewPlaintext(params, ECD_LV, params.Scale())
// 		encoder.EncodeCoeffs(encode_ker(ker1, pos, i, in_wid_out, batch_real, ker_wid, true), pl_ker[i])
// 		encoder.ToNTT(pl_ker[i])
// 	}

// 	fmt.Println("vec size: ", N)
// 	fmt.Println("input width: ", in_wid)
// 	fmt.Println("kernel width: ", ker_wid)
// 	fmt.Println("num batches (input): ", batch)
// 	fmt.Println("num batches (real): ", batch_real)
// 	fmt.Println("Input matrix: ")
// 	prt_vec(input)
// 	fmt.Println("Ker1_in (1st part): ")
// 	prt_vec(ker1[0])

// 	fmt.Println()
// 	fmt.Println("=========================================")
// 	fmt.Println("   PLAINTEXT CREATION & ENCRYPTION       ")
// 	fmt.Println("=========================================")
// 	fmt.Println()

// 	start = time.Now()

// 	cfs_tmp := make([]float64, N)                                             // contain coefficient msgs
// 	plain_tmp := ckks.NewPlaintext(params, params.MaxLevel(), params.Scale()) // contain plaintext values
// 	copy(cfs_tmp, input)

// 	encoder.EncodeCoeffs(cfs_tmp, plain_tmp)
// 	ctxt_input := encryptor.EncryptNew(plain_tmp)
// 	ctxt_out := make([]*ckks.Ciphertext, batch_real)

// 	fmt.Printf("Done in %s \n", time.Since(start))

// 	fmt.Print("Boot in: ")
// 	// Decrypt, print and compare with the plaintext values
// 	fmt.Println()
// 	fmt.Println("Precision of values vs. ciphertext")
// 	values_test := printDebugCfs(params, ctxt_input, cfs_tmp, decryptor, encoder)

// 	// plain_ttmp := ckks.NewPlaintext(params, ctxt_input.Level(), ctxt_input.Scale)
// 	// decryptor.Decrypt(ctxt_input, plain_ttmp)
// 	// tresult := encoder.Decode(plain_ttmp, params.LogSlots())
// 	// prt_res := make([]int, len(tresult))
// 	// fmt.Print(tresult)
// 	// for i := range tresult {
// 	// 	prt_res[i] = int(real(tresult[i]))
// 	// }
// 	// fmt.Println(prt_res)

// 	// ctxt_input = ext_ctxt(evaluator, encoder, ctxt_input, r_idx, m_idx, params)

// 	// fmt.Println("mult and rot")
// 	// decryptor.Decrypt(ctxt_input, plain_ttmp)
// 	// // decryptor.Decrypt(evaluator.RotateNew(ctxt_input, 1), plain_ttmp)
// 	// tresult = encoder.Decode(plain_ttmp, params.LogSlots())
// 	// for i := range tresult {
// 	// 	prt_res[i] = int(real(tresult[i]))
// 	// }
// 	// fmt.Println(prt_res)

// 	fmt.Println()
// 	fmt.Println("=========================================")
// 	fmt.Println("              BOOTSTRAPP                 ")
// 	fmt.Println("=========================================")
// 	fmt.Println()

// 	fmt.Println("Generating bootstrapping keys...")
// 	start = time.Now()
// 	rotations = btpParams.RotationsForBootstrapping(params.LogSlots())
// 	rotkeys = kgen.GenRotationKeysForRotations(rotations, true, sk)
// 	btpKey := ckks.BootstrappingKey{Rlk: rlk, Rtks: rotkeys}
// 	if btp, err = ckks.NewBootstrapper(params, btpParams, btpKey); err != nil {
// 		panic(err)
// 	}
// 	fmt.Printf("Done in %s \n", time.Since(start))

// 	fmt.Println("Bootstrapping... Ours:")
// 	start = time.Now()
// 	// Reason for multpling 1/(2*N) : for higher precision in SineEval & ReLU before StoC (needs to be recovered after/before StoC)
// 	// ciphertext1.SetScalingFactor(ciphertext1.Scale * float64(2*params.N()))

// 	ctxt1, ctxt2, _ := btp.BootstrappConv_CtoS(ctxt_input, float64(pow))
// 	fmt.Printf("Done in %s \n", time.Since(start))
// 	fmt.Println("after Boot: LV = ", ctxt1.Level(), " Scale = ", math.Log2(ctxt1.Scale))

// 	ctxt1 = ext_ctxt(evaluator, encoder, ctxt1, r_idx, params)
// 	ctxt2 = ext_ctxt(evaluator, encoder, ctxt2, r_idx, params)

// 	// evaluator.Rescale(ctxt1, params.Scale(), ctxt1)
// 	// evaluator.Rescale(ctxt2, params.Scale(), ctxt2)
// 	// // evaluator.Rotate(ctxt1, 4, ctxt1)
// 	// evaluator.Rotate(ctxt2, 4, ctxt2)
// 	// evaluator.DropLevel(ctxt1, 2)
// 	// evaluator.DropLevel(ctxt2, 2)

// 	ciphertext := btp.BootstrappConv_StoC(ctxt1, ctxt2)

// 	// evaluator.ScaleUp(ciphertext, params.Scale(), ciphertext)
// 	// evaluator.Rescale(ciphertext, params.Scale(), ciphertext)

// 	// fmt.Println("now level: ", ciphertext.Level())
// 	// fmt.Println("now scale: ", math.Log2(ciphertext.Scale))

// 	fmt.Printf("Boot out: ")
// 	// values_test1 := reverseOrder(values_test[:params.Slots()], params.LogSlots())
// 	// values_test2 := reverseOrder(values_test[params.Slots():], params.LogSlots())
// 	// for i := range values_test1 {
// 	// 	values_test[i] = values_test1[i]
// 	// 	values_test[i+params.Slots()] = values_test2[i]
// 	// }

// 	values_tmp1 := make([]float64, params.Slots())
// 	values_tmp2 := make([]float64, params.Slots())
// 	for i := range values_tmp1 {
// 		values_tmp1[i] = values_test[reverseBits(uint32(i), params.LogSlots())]
// 		values_tmp2[i] = values_test[reverseBits(uint32(i), params.LogSlots())+uint32(params.Slots())]
// 	}

// 	values_tmp11 := extend_full_nhf(values_tmp1, 2*in_wid, pos, true, true)
// 	values_tmp22 := extend_full_nhf(values_tmp2, 2*in_wid, pos, true, true)
// 	for i := range values_tmp1 {
// 		values_tmp1[i] = values_tmp11[reverseBits(uint32(i), params.LogSlots())]
// 		values_tmp2[i] = values_tmp22[reverseBits(uint32(i), params.LogSlots())]
// 	}
// 	values_test = append(values_tmp1, values_tmp2...)

// 	printDebugCfs(params, ciphertext, values_test, decryptor, encoder)

// 	// fmt.Printf("Boot out2: ")
// 	// values_test2 := reverseOrder(values_test[params.Slots():], params.LogSlots())
// 	// for i := range values_testC {
// 	// 	values_testC[i] = complex(values_test2[i], 0)
// 	// }
// 	// printDebug(params, ciphertext3, values_testC, decryptor, encoder)

// 	/// Not Necessary!!! ///
// 	xi_tmp := make([]float64, N)
// 	xi_tmp[(ker_wid-1)*(in_wid_out+1)] = 1.0
// 	xi_plain := ckks.NewPlaintext(params, ECD_LV, 1.0)
// 	encoder.EncodeCoeffs(xi_tmp, xi_plain)
// 	encoder.ToNTT(xi_plain)

// 	evaluator.Mul(ciphertext, xi_plain, ciphertext)

// 	mov_test := false

// 	if mov_test {
// 		fmt.Println()
// 		fmt.Println("=========================================")
// 		fmt.Println("              DECRYPTION                 ")
// 		fmt.Println("=========================================")
// 		fmt.Println()

// 		start = time.Now()

// 		decryptor.Decrypt(ciphertext, plain_tmp)
// 		cfs_tmp = encoder.DecodeCoeffs(plain_tmp)
// 		// cfs_tmp = reshape_conv_out(encoder.DecodeCoeffs(plain_tmp), in_wid, ker_wid, batch)
// 		int_tmp := make([]int, len(cfs_tmp))

// 		for i := range cfs_tmp {
// 			int_tmp[i] = int(float64(N) * cfs_tmp[i])
// 		}

// 		if printResult {
// 			fmt.Print("Result: \n")
// 			// fmt.Println(int_tmp)
// 			for b := 0; b < batch_real; b++ {
// 				print_vec_int("input ("+strconv.Itoa(b)+")", int_tmp, in_wid_out, b)
// 			}

// 			for i := range values_test {
// 				int_tmp[i] = int(float64(N) * values_test[i])
// 			}
// 			for b := 0; b < batch_real; b++ {
// 				print_vec_int("cp_input ("+strconv.Itoa(b)+")", int_tmp, in_wid_out, b)
// 			}
// 		}

// 		fmt.Printf("Done in %s \n", time.Since(start))

// 	} else {
// 		fmt.Println()
// 		fmt.Println("===============================================")
// 		fmt.Println("     			   EVALUATION					")
// 		fmt.Println("===============================================")
// 		fmt.Println()

// 		start = time.Now()

// 		for i := 0; i < batch_real; i++ {
// 			ctxt_out[i] = pack_evaluator.MulNew(ciphertext, pl_ker[i])
// 		}

// 		ctxt_result := pack_ctxts(pack_evaluator, ctxt_out, batch_real, plain_idx, params)
// 		fmt.Println("Result Scale: ", math.Log2(ctxt_result.Scale))
// 		fmt.Println("Result LV: ", ctxt_result.Level())
// 		fmt.Printf("Done in %s \n", time.Since(start))

// 		fmt.Println()
// 		fmt.Println("=========================================")
// 		fmt.Println("              DECRYPTION                 ")
// 		fmt.Println("=========================================")
// 		fmt.Println()

// 		start = time.Now()

// 		decryptor.Decrypt(ctxt_result, plain_tmp)
// 		cfs_tmp = reshape_conv_out(encoder.DecodeCoeffs(plain_tmp), in_wid_out/2, batch_real)

// 		if printResult {
// 			fmt.Print("Result: \n")
// 			prt_mat(cfs_tmp, batch_real, 2*in_wid)
// 		}

// 		fmt.Printf("Done in %s \n", time.Since(start))
// 	}

// 	return cfs_tmp
// }

// func testDCGAN_old() {

// 	alpha := 0.3
// 	print := false
// 	logN := 16
// 	in_wid := [4]int{4, 8, 16, 32}
// 	max_batch := [4]int{1024, 256, 64, 16}
// 	batch := [4]int{512, 128, 64, 1}
// 	// max_batch := [4]int{64, 16, 4, 1}
// 	// batch := [4]int{16, 4, 2, 1}
// 	ker_wid := 5
// 	N := (1 << logN)
// 	// st_batch := N / (2 * in_wid * 2 * in_wid) // We also consider zero-paddings  // must be adjusted when in_wid is not power of 2
// 	// end_batch := 4
// 	ECD_LV := 1

// 	// parameter generation (comment out when do other test)
// 	var btp *ckks.Bootstrapper
// 	btpParams := ckks.DefaultBootstrapParams[6]
// 	params, err := btpParams.Params()
// 	if err != nil {
// 		panic(err)
// 	}
// 	fmt.Printf("CKKS parameters: logN = %d, logSlots = %d, h = %d, logQP = %d, levels = %d, scale= 2^%f, sigma = %f \n",
// 		params.LogN(), params.LogSlots(), btpParams.H, params.LogQP(), params.QCount(), math.Log2(params.Scale()), params.Sigma())

// 	// Generate rotations for EXT_FULL
// 	var rotations []int
// 	r_idx := make([]map[int][]int, 4)
// 	m_idx := make([]map[int][]int, 4)
// 	for pos := 0; pos < 4; pos++ {
// 		r_idx[pos], m_idx[pos] = gen_extend_full_nhf(N/2, in_wid[1], pos, true, true)
// 		for k := range r_idx[pos] {
// 			rotations = append(rotations, k)
// 		}
// 		for k := range m_idx[pos] {
// 			rotations = append(rotations, k)
// 		}
// 	}
// 	r_idx1 := make([]map[int][]int, 4)
// 	m_idx1 := make([]map[int][]int, 4)
// 	for pos := 0; pos < 4; pos++ {
// 		r_idx1[pos], m_idx1[pos] = gen_extend_full_nhf(N/2, in_wid[2], pos, true, true)
// 		for k := range r_idx1[pos] {
// 			rotations = append(rotations, k)
// 		}
// 		for k := range m_idx1[pos] {
// 			rotations = append(rotations, k)
// 		}
// 	}
// 	r_idx2 := make([]map[int][]int, 4)
// 	m_idx2 := make([]map[int][]int, 4)
// 	for pos := 0; pos < 4; pos++ {
// 		r_idx2[pos], m_idx2[pos] = gen_extend_full_nhf(N/2, in_wid[3], pos, true, true)
// 		for k := range r_idx2[pos] {
// 			rotations = append(rotations, k)
// 		}
// 		for k := range m_idx2[pos] {
// 			rotations = append(rotations, k)
// 		}
// 	}
// 	fmt.Println("Rotations: ", rotations)

// 	// Scheme context and keys for evaluation (no Boot)
// 	kgen := ckks.NewKeyGenerator(params)
// 	sk, _ := kgen.GenKeyPairSparse(btpParams.H)
// 	rlk := kgen.GenRelinearizationKey(sk, 2)
// 	rotkeys := kgen.GenRotationKeysForRotations(rotations, false, sk)
// 	encoder := ckks.NewEncoder(params)
// 	decryptor := ckks.NewDecryptor(params, sk)
// 	encryptor := ckks.NewEncryptor(params, sk)
// 	evaluator := ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk, Rtks: rotkeys})

// 	plain_idx, pack_evaluator := gen_idxNlogs(ECD_LV, kgen, sk, encoder, params)

// 	fmt.Println("Generating bootstrapping keys...")
// 	start = time.Now()
// 	rotations = btpParams.RotationsForBootstrapping(params.LogSlots())
// 	rotkeys = kgen.GenRotationKeysForRotations(rotations, true, sk)
// 	btpKey := ckks.BootstrappingKey{Rlk: rlk, Rtks: rotkeys}
// 	if btp, err = ckks.NewBootstrapper_mod(params, btpParams, btpKey); err != nil {
// 		panic(err)
// 	}
// 	fmt.Printf("Done in %s \n", time.Since(start))

// 	for iter := 1; iter < 2; iter++ {
// 		name_iter := fmt.Sprintf("%04d", iter)

// 		input := readTxt("./inputs/input_"+name_iter+".txt", 0)
// 		ext_input := inputExt(input, logN, in_wid[0], false) // Takes arranged input (assume intermediate layers)  // print only outputs first (st_batch) batches

// 		// circ_rows := readTxt("./varaibles/rows.txt")
// 		// circ_mat := make([][]float64, 8)
// 		// plain_mat := make([]*ckks.Plaintext, 8)
// 		// for i := 0; i < len(circ_mat); i++ {
// 		// 	circ_mat[i] = encode_circ(circ_rows[N/2*i:N/2*(i+1)], 16, N)
// 		// 	plain_mat[i] = ckks.NewPlaintext(params, ECD_LV, params.Scale())
// 		// 	encoder.EncodeCoeffs(circ_mat[i], plain_mat[i])
// 		// 	encoder.ToNTT(plain_mat[i])
// 		// }

// 		// test_input := readTxt("./inputs/vec_input_" + name_iter + ".txt")
// 		// enc_test_input := make([]*ckks.Ciphertext, 8)
// 		// test_tmp := ckks.NewPlaintext(params, ECD_LV, params.Scale())
// 		// for i := 0; i < len(enc_test_input); i++ {
// 		// 	encoder.EncodeCoeffs(encode_circ_in(test_input, i, 16, N), test_tmp)
// 		// 	enc_test_input[i] = encryptor.EncryptNew(test_tmp)
// 		// }

// 		// var test_result *ckks.Ciphertext
// 		// for i := 0; i < len(enc_test_input); i++ {
// 		// 	if i == 0 {
// 		// 		test_result = evaluator.MulNew(enc_test_input[i], plain_mat[i])
// 		// 	} else {
// 		// 		evaluator.Add(test_result, evaluator.MulNew(enc_test_input[i], plain_mat[i]), test_result)
// 		// 	}
// 		// }

// 		// input := make([]float64, N)
// 		// for i := range input {
// 		// 	input[i] = 1.0 * float64(ext_input[i]) / float64(in_wid[0]*in_wid[0]*batch[0])
// 		// }

// 		start := time.Now()
// 		plain_in := ckks.NewPlaintext(params, 1, params.Scale()) // contain plaintext values
// 		encoder.EncodeCoeffs(ext_input, plain_in)
// 		ctxt_in := encryptor.EncryptNew(plain_in)

// 		// zeros := make([]complex128, params.Slots())
// 		// plain_in = encoder.EncodeNew(zeros, params.LogSlots())
// 		// ctxt0 := encryptor.EncryptNew(plain_in)

// 		fmt.Printf("Encryption: Done in %s \n", time.Since(start))

// 		if print {
// 			fmt.Println("vec size: ", N)
// 			fmt.Println("input width: ", in_wid)
// 			fmt.Println("kernel width: ", ker_wid)
// 			fmt.Println("num batches (in 1 ctxt with padding): ", max_batch[0])
// 		}

// 		ker1 := readTxt("./variables/conv1.txt", 0)
// 		a1 := readTxt("./variables/a1.txt", 0)
// 		b1 := readTxt("./variables/b1.txt", 0)
// 		b1_coeffs := make([]float64, N)
// 		for i := range b1 {
// 			for j := 0; j < in_wid[1]; j++ {
// 				for k := 0; k < in_wid[1]; k++ {
// 					b1_coeffs[i+(j+k*in_wid[1]*2)*max_batch[1]] = b1[i]
// 				}
// 			}
// 		}

// 		pl_ker := prepKer_in_trans(params, encoder, encryptor, ker1, a1, in_wid[0], ker_wid, max_batch[0], max_batch[1], batch[0], batch[1], ECD_LV)

// 		fmt.Print("Boot in: ")
// 		fmt.Println()
// 		fmt.Println("Precision of values vs. ciphertext")
// 		in_cfs := printDebugCfs(params, ctxt_in, ext_input, decryptor, encoder)

// 		fmt.Println("Bootstrapping... Ours (until CtoS):")
// 		start = time.Now()
// 		ctxt1, ctxt2, _ := btp.BootstrappConv_CtoS(ctxt_in, float64(pow))
// 		fmt.Printf("Done in %s \n", time.Since(start))
// 		fmt.Println("after Boot: LV = ", ctxt1.Level(), " Scale = ", math.Log2(ctxt1.Scale))

// 		// Only for checking the correctness
// 		in_cfs_1_pBoot := make([]float64, params.Slots())
// 		in_cfs_2_pBoot := make([]float64, params.Slots())
// 		in_slots := make([]complex128, params.Slots()) // first part of ceffs
// 		for i := range in_cfs_1_pBoot {
// 			in_cfs_1_pBoot[i] = in_cfs[reverseBits(uint32(i), params.LogSlots())] // first part of coeffs
// 			in_cfs_2_pBoot[i] = in_cfs[reverseBits(uint32(i), params.LogSlots())+uint32(params.Slots())]
// 			in_slots[i] = complex(in_cfs_1_pBoot[i]/math.Pow(2, float64(pow)), 0)
// 		}
// 		ext1_tmp := extend_full_nhf(in_cfs_1_pBoot, in_wid[1], 3, true, true)
// 		ext2_tmp := extend_full_nhf(in_cfs_2_pBoot, in_wid[1], 3, true, true)
// 		for i := range in_cfs_1_pBoot {
// 			in_cfs_1_pBoot[i] = ext1_tmp[reverseBits(uint32(i), params.LogSlots())]
// 			in_cfs_2_pBoot[i] = ext2_tmp[reverseBits(uint32(i), params.LogSlots())]
// 		}
// 		in_cfs_pBoot := append(in_cfs_1_pBoot, in_cfs_2_pBoot...) // After rot(ext) and boot

// 		in_slots = printDebug(params, ctxt1, in_slots, decryptor, encoder) // Compare before & after CtoS

// 		start = time.Now()
// 		// evaluator.MultByConst(ctxt1, 1.000000001, ctxt1)
// 		// evaluator.DropLevel(ctxt1, 10)
// 		evaluator.MulByPow2(ctxt1, pow, ctxt1)
// 		evaluator.DropLevel(ctxt1, ctxt1.Level()-3)

// 		// ctxt1 = evalReLU(params, evaluator, ctxt1, 1.0)
// 		fmt.Printf("NO ReLU Done in %s \n", time.Since(start))

// 		values_ReLU := make([]complex128, len(in_slots))
// 		for i := range values_ReLU {
// 			values_ReLU[i] = complex(math.Pow(2, float64(pow)), 0) * in_slots[i] // complex(math.Max(0, real(in_slots[i])), 0)
// 		}
// 		printDebug(params, ctxt1, values_ReLU, decryptor, encoder)

// 		ext_ctxt1 := make([]*ckks.Ciphertext, 4) // for extend (rotation) of ctxt_in
// 		// ext_ctxt2 := make([]*ckks.Ciphertext, 4)  // do not need if we use po2 inputs dims
// 		ciphertext := make([]*ckks.Ciphertext, 4) // after Bootstrapping

// 		ctxt2 = nil
// 		// evaluator.DropLevel(ctxt2, ctxt2.Level()-2)
// 		start = time.Now()
// 		for pos := 0; pos < 4; pos++ {
// 			ext_ctxt1[pos] = ext_ctxt(evaluator, encoder, ctxt1, r_idx[pos], params)
// 			// fmt.Println(ext_ctxt1[pos].Level(), ctxt2.Level(), ext_ctxt1[pos].Scale, ctxt2.Scale)
// 			// ext_ctxt2[pos] = ext_ctxt(evaluator, encoder, ctxt2, r_idx[pos], m_idx[pos], params)
// 			ciphertext[pos] = btp.BootstrappConv_StoC(ext_ctxt1[pos], ctxt2)
// 			evaluator.Rescale(ciphertext[pos], params.Scale(), ciphertext[pos])
// 		}
// 		fmt.Printf("Boot (StoC) Done in %s \n", time.Since(start))

// 		fmt.Printf("Boot out: ")
// 		// for i := range in_cfs_pBoot {
// 		// 	in_cfs_pBoot[i] = math.Max(0, in_cfs_pBoot[i])
// 		// }
// 		printDebugCfs(params, ciphertext[3], in_cfs_pBoot, decryptor, encoder)

// 		ctxt_result := conv_then_pack_trans(params, pack_evaluator, ciphertext, pl_ker, plain_idx, max_batch[1])

// 		// for Batch Normalization (BN)
// 		plain_in = ckks.NewPlaintext(params, ctxt_result.Level(), ctxt_result.Scale) // contain plaintext values
// 		encoder.EncodeCoeffs(b1_coeffs, plain_in)
// 		encoder.ToNTT(plain_in)
// 		evaluator.Add(ctxt_result, plain_in, ctxt_result)

// 		plain_out := ckks.NewPlaintext(params, ctxt_result.Level(), params.Scale())
// 		start = time.Now()
// 		decryptor.Decrypt(ctxt_result, plain_out)
// 		pre_boot := encoder.DecodeCoeffs(plain_out)
// 		cfs_tmp := reshape_conv_out(encoder.DecodeCoeffs(plain_out), in_wid[1], max_batch[1])

// 		if print {
// 			fmt.Println()
// 			fmt.Println("=========================================")
// 			fmt.Println("              DECRYPTION                 ")
// 			fmt.Println("=========================================")
// 			fmt.Println()

// 			fmt.Print("Result: \n")
// 			prt_mat(cfs_tmp, max_batch[1], in_wid[1])
// 		}
// 		fmt.Printf("(Layer 1) Done in %s \n", time.Since(start))

// 		// // To see each matrix
// 		// cfs_tmp = encoder.DecodeCoeffs(plain_out)
// 		// int_tmpn := make([]int, N)
// 		// for i := range cfs_tmp {
// 		// 	int_tmpn[i] = int(cfs_tmp[i])
// 		// }
// 		// fmt.Print("Output: \n")
// 		// for b := 0; b < batch[1]; b++ {
// 		// 	print_vec("output ("+strconv.Itoa(b)+")", int_tmpn, in_wid[2], b)
// 		// }

// 		// // Layer 1 done

// 		fmt.Println()
// 		fmt.Println("=========================================")
// 		fmt.Println("              LAYER 2	                 ")
// 		fmt.Println("=========================================")
// 		fmt.Println()

// 		ker2 := readTxt("./variables/conv2.txt", 0)
// 		a2 := readTxt("./variables/a2.txt", 0)
// 		b2 := readTxt("./variables/b2.txt", 0)
// 		b2_coeffs := make([]float64, N)
// 		for i := range b2 {
// 			for j := 0; j < in_wid[2]; j++ {
// 				for k := 0; k < in_wid[2]; k++ {
// 					b2_coeffs[i+(j+k*in_wid[2]*2)*max_batch[2]] = b2[i]
// 				}
// 			}
// 		}

// 		pl_ker = prepKer_in_trans(params, encoder, encryptor, ker2, a2, in_wid[1], ker_wid, max_batch[1], max_batch[2], batch[1], batch[2], ECD_LV)

// 		// fmt.Print("Boot in: ")
// 		// fmt.Println()
// 		// fmt.Println("Precision of values vs. ciphertext")
// 		// in_cfs = printDebugCfs(params, ctxt_result, pre_boot, decryptor, encoder)
// 		in_cfs = pre_boot
// 		ctxt_in.Copy(ctxt_result)
// 		// ctxt_in.SetScalingFactor(ctxt_in.Scale * 64)

// 		fmt.Println("Bootstrapping... Ours (until CtoS):")
// 		start = time.Now()
// 		ctxt1, _, _ = btp.BootstrappConv_CtoS(ctxt_in, float64(pow))
// 		fmt.Printf("Done in %s \n", time.Since(start))

// 		// Only for checking the correctness
// 		for i := range in_cfs_1_pBoot {
// 			in_cfs_1_pBoot[i] = in_cfs[reverseBits(uint32(i), params.LogSlots())] // first part of coeffs
// 			in_cfs_2_pBoot[i] = 0                                                 // in_cfs[reverseBits(uint32(i), params.LogSlots())+uint32(params.Slots())]
// 			in_slots[i] = complex(in_cfs_1_pBoot[i]/math.Pow(2, float64(pow)), 0)
// 		}
// 		ext1_tmp = extend_full_nhf(in_cfs_1_pBoot, in_wid[2], 0, true, true)
// 		ext2_tmp = extend_full_nhf(in_cfs_2_pBoot, in_wid[2], 0, true, true)
// 		for i := range in_cfs_1_pBoot {
// 			in_cfs_1_pBoot[i] = ext1_tmp[reverseBits(uint32(i), params.LogSlots())]
// 			in_cfs_2_pBoot[i] = ext2_tmp[reverseBits(uint32(i), params.LogSlots())]
// 		}
// 		in_cfs_pBoot = append(in_cfs_1_pBoot, in_cfs_2_pBoot...) // After rot(ext) and boot
// 		in_slots = printDebug(params, ctxt1, in_slots, decryptor, encoder)

// 		start = time.Now()

// 		plain_ch := ckks.NewPlaintext(params, ctxt1.Level(), params.Scale())
// 		decryptor.Decrypt(ctxt1, plain_ch)
// 		check := encoder.Decode(plain_ch, logN-1)
// 		max := 0.0
// 		avg := 0.0
// 		for _, val := range check {
// 			rval := real(val)
// 			if math.Abs(rval) > math.Abs(max) {
// 				max = rval
// 			}
// 			avg += rval
// 		}
// 		avg = 2 * avg / float64(N)
// 		fmt.Println("max valu: ", max)
// 		fmt.Println("avg valu: ", avg)

// 		// evaluator.MulByPow2(ctxt1, pow, ctxt1)
// 		// evaluator.DropLevel(ctxt1, ctxt1.Level()-3)

// 		ctxt1 = evalReLU(params, evaluator, ctxt1, alpha)
// 		evaluator.MulByPow2(ctxt1, pow, ctxt1)
// 		fmt.Printf("ReLU Done in %s \n", time.Since(start))

// 		for i := range values_ReLU {
// 			values_ReLU[i] = complex(math.Pow(2, float64(pow)), 0) * complex(math.Max(0, real(in_slots[i]))+alpha*math.Min(0, real(in_slots[i])), 0)
// 		}
// 		printDebug(params, ctxt1, values_ReLU, decryptor, encoder)

// 		start = time.Now()
// 		for pos := 0; pos < 4; pos++ {
// 			ext_ctxt1[pos] = ext_ctxt(evaluator, encoder, ctxt1, r_idx1[pos], params)
// 			// ext_ctxt2[pos] = ext_ctxt(evaluator, encoder, ctxt2, r_idx1[pos], m_idx1[pos], params)
// 			ciphertext[pos] = btp.BootstrappConv_StoC(ext_ctxt1[pos], ctxt2)
// 			evaluator.Rescale(ciphertext[pos], params.Scale(), ciphertext[pos])
// 		}
// 		fmt.Printf("Boot (StoC) Done in %s \n", time.Since(start))

// 		fmt.Printf("Boot out: ")
// 		for i := range in_cfs_pBoot {
// 			in_cfs_pBoot[i] = math.Max(0, in_cfs_pBoot[i]) + alpha*math.Min(0, in_cfs_pBoot[i])
// 		}
// 		printDebugCfs(params, ciphertext[0], in_cfs_pBoot, decryptor, encoder)

// 		ctxt_result = conv_then_pack_trans(params, pack_evaluator, ciphertext, pl_ker, plain_idx, max_batch[2])

// 		// for BN
// 		plain_in = ckks.NewPlaintext(params, ctxt_result.Level(), ctxt_result.Scale) // contain plaintext values
// 		encoder.EncodeCoeffs(b2_coeffs, plain_in)
// 		encoder.ToNTT(plain_in)
// 		evaluator.Add(ctxt_result, plain_in, ctxt_result)

// 		start = time.Now()
// 		decryptor.Decrypt(ctxt_result, plain_out)
// 		pre_boot = encoder.DecodeCoeffs(plain_out)
// 		cfs_tmp = reshape_conv_out(encoder.DecodeCoeffs(plain_out), in_wid[2], max_batch[2])

// 		if print {
// 			fmt.Println()
// 			fmt.Println("=========================================")
// 			fmt.Println("              DECRYPTION                 ")
// 			fmt.Println("=========================================")
// 			fmt.Println()
// 			fmt.Print("Result: \n")
// 			prt_mat(cfs_tmp, max_batch[2], in_wid[2])
// 		}
// 		fmt.Printf("(Layer 2) Done in %s \n", time.Since(start))

// 		fmt.Println()
// 		fmt.Println("=========================================")
// 		fmt.Println("              LAYER 3	                 ")
// 		fmt.Println("=========================================")
// 		fmt.Println()

// 		ker3 := readTxt("./variables/conv3.txt", 0)
// 		a3 := make([]float64, batch[3])
// 		for i := range a3 {
// 			a3[i] = 1
// 		}
// 		pl_ker = prepKer_in_trans(params, encoder, encryptor, ker3, a3, in_wid[2], ker_wid, max_batch[2], max_batch[3], batch[2], batch[3], ECD_LV)

// 		// fmt.Print("Boot in: ")
// 		// fmt.Println()
// 		// fmt.Println("Precision of values vs. ciphertext")
// 		// in_cfs = printDebugCfs(params, ctxt_result, pre_boot, decryptor, encoder)
// 		in_cfs = pre_boot
// 		ctxt_in.Copy(ctxt_result)
// 		// ctxt_in.SetScalingFactor(ctxt_in.Scale * 16)

// 		fmt.Println("Bootstrapping... Ours (until CtoS):")
// 		start = time.Now()
// 		ctxt1, _, _ = btp.BootstrappConv_CtoS(ctxt_in, float64(pow))
// 		fmt.Printf("Done in %s \n", time.Since(start))

// 		// Only for checking the correctness
// 		for i := range in_cfs_1_pBoot {
// 			in_cfs_1_pBoot[i] = in_cfs[reverseBits(uint32(i), params.LogSlots())] // first part of coeffs
// 			in_cfs_2_pBoot[i] = 0                                                 // in_cfs[reverseBits(uint32(i), params.LogSlots())+uint32(params.Slots())]
// 			in_slots[i] = complex(in_cfs_1_pBoot[i]/math.Pow(2, float64(pow)), 0)
// 		}
// 		ext1_tmp = extend_full_nhf(in_cfs_1_pBoot, in_wid[3], 0, true, true)
// 		ext2_tmp = extend_full_nhf(in_cfs_2_pBoot, in_wid[3], 0, true, true)
// 		for i := range in_cfs_1_pBoot {
// 			in_cfs_1_pBoot[i] = ext1_tmp[reverseBits(uint32(i), params.LogSlots())]
// 			in_cfs_2_pBoot[i] = ext2_tmp[reverseBits(uint32(i), params.LogSlots())]
// 		}
// 		in_cfs_pBoot = append(in_cfs_1_pBoot, in_cfs_2_pBoot...) // After rot(ext) and boot
// 		in_slots = printDebug(params, ctxt1, in_slots, decryptor, encoder)

// 		start = time.Now()
// 		// evaluator.MultByConst(ctxt1, 1.000000001, ctxt1)
// 		// evaluator.DropLevel(ctxt1, 10)

// 		plain_ch = ckks.NewPlaintext(params, ctxt1.Level(), params.Scale())
// 		decryptor.Decrypt(ctxt1, plain_ch)
// 		check = encoder.Decode(plain_ch, logN-1)
// 		max = 0.0
// 		avg = 0.0
// 		for _, val := range check {
// 			rval := real(val)
// 			if math.Abs(rval) > math.Abs(max) {
// 				max = rval
// 			}
// 			avg += rval
// 		}
// 		avg = 2 * avg / float64(N)
// 		fmt.Println("max valu: ", max)
// 		fmt.Println("avg valu: ", avg)

// 		ctxt1 = evalReLU(params, evaluator, ctxt1, alpha)
// 		evaluator.MulByPow2(ctxt1, pow, ctxt1)
// 		fmt.Printf("ReLU Done in %s \n", time.Since(start))

// 		for i := range values_ReLU {
// 			values_ReLU[i] = complex(math.Pow(2, float64(pow)), 0) * complex(math.Max(0, real(in_slots[i]))+alpha*math.Min(0, real(in_slots[i])), 0)
// 		}
// 		printDebug(params, ctxt1, values_ReLU, decryptor, encoder)

// 		start = time.Now()
// 		for pos := 0; pos < 4; pos++ {
// 			ext_ctxt1[pos] = ext_ctxt(evaluator, encoder, ctxt1, r_idx2[pos], params)
// 			// ext_ctxt2[pos] = ext_ctxt(evaluator, encoder, ctxt2, r_idx2[pos], m_idx2[pos], params)
// 			ciphertext[pos] = btp.BootstrappConv_StoC(ext_ctxt1[pos], ctxt2)
// 			evaluator.Rescale(ciphertext[pos], params.Scale(), ciphertext[pos])
// 		}
// 		fmt.Printf("Boot (StoC) Done in %s \n", time.Since(start))

// 		fmt.Printf("Boot out: ")
// 		for i := range in_cfs_pBoot {
// 			in_cfs_pBoot[i] = math.Max(0, in_cfs_pBoot[i]) + alpha*math.Min(0, in_cfs_pBoot[i])
// 		}
// 		printDebugCfs(params, ciphertext[0], in_cfs_pBoot, decryptor, encoder)

// 		ctxt_result = conv_then_pack_trans(params, pack_evaluator, ciphertext, pl_ker, plain_idx, max_batch[3])
// 		// ctxt_result = conv_then_pack(params, pack_evaluator, ciphertext, pl_ker, plain_idx, batch[2])

// 		start = time.Now()
// 		decryptor.Decrypt(ctxt_result, plain_out)
// 		pre_boot = encoder.DecodeCoeffs(plain_out)
// 		cfs_tmp = reshape_conv_out(encoder.DecodeCoeffs(plain_out), in_wid[3], max_batch[3])

// 		if print {
// 			fmt.Println()
// 			fmt.Println("=========================================")
// 			fmt.Println("              DECRYPTION                 ")
// 			fmt.Println("=========================================")
// 			fmt.Println()
// 			fmt.Print("Result: \n")
// 			prt_mat(cfs_tmp, max_batch[3], in_wid[3])
// 		}
// 		fmt.Printf("(Layer 3) Done in %s \n", time.Since(start))

// 		output := make([]float64, in_wid[3]*in_wid[3])
// 		for i := range output {
// 			output[i] = cfs_tmp[max_batch[3]*i]
// 		}
// 		writeTxt("result_"+name_iter+".txt", output)
// 	}
// }

// Fast Conv without boot, Assume full batch with Po2 in_wid & N
// Normal Conv without output modification (e.g., trimming or expanding)
// Assume that the input is 0 padded according to kernel size: only in_wid - (ker_wid-1)/2 elements in row and columns are nonzero
// func testConv_noBoot_old(logN, in_wid, ker_wid int, printResult, trans bool) []float64 {
// 	N := (1 << logN)
// 	in_size := in_wid * in_wid
// 	batch := N / in_size
// 	ker_size := ker_wid * ker_wid

// 	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{ // Schemes parameters are created from scratch
// 		LogN:     logN,
// 		LogQ:     []int{log_out_scale + log_c_scale, log_in_scale},
// 		LogP:     []int{60},
// 		Sigma:    rlwe.DefaultSigma,
// 		LogSlots: logN - 1,
// 		Scale:    float64(1 << log_in_scale),
// 	})
// 	if err != nil {
// 		panic(err)
// 	}

// 	fmt.Println()
// 	fmt.Println("========================================================")
// 	fmt.Println(" INSTANTIATING SCHEME & PLAINTEXT CREATION & Encryption ")
// 	fmt.Println("========================================================")
// 	fmt.Println()

// 	start = time.Now()
// 	kgen := ckks.NewKeyGenerator(params)
// 	sk := kgen.GenSecretKey()
// 	rlk := kgen.GenRelinearizationKey(sk, 2)
// 	encryptor := ckks.NewEncryptor(params, sk)
// 	decryptor := ckks.NewDecryptor(params, sk)
// 	encoder := ckks.NewEncoder(params)
// 	evaluator := ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk})
// 	plain_idx, pack_evaluator := gen_idxNlogs(params.MaxLevel(), kgen, sk, encoder, params) // for final pack_ciphertexts
// 	fmt.Printf("Keygen, Done in %s \n", time.Since(start))

// 	fmt.Println()
// 	fmt.Printf("CKKS parameters: logN = %d, logSlots = %d, logQP = %d, levels = %d, scale= %f, sigma = %f \n",
// 		params.LogN(), params.LogSlots(), params.LogQP(), params.MaxLevel()+1, params.Scale(), params.Sigma())

// 	// input preparation
// 	input := make([]float64, N)
// 	pad := (ker_wid - 1) / 2 // for batch with largest input possible

// 	if trans {
// 		k := 0.0
// 		for i := 0; i < in_wid/2; i++ {
// 			for j := 0; j < in_wid/2; j++ {
// 				for b := 0; b < batch; b++ {
// 					if ((2*i + 1) < (in_wid - pad)) && ((2*j + 1) < (in_wid - pad)) {
// 						input[(2*i+1)*in_wid*batch+(2*j+1)*batch+b] = k
// 						k += 1.0
// 					}
// 				}
// 			}
// 		}
// 	} else {
// 		k := 1.0
// 		for i := 0; i < in_wid; i++ {
// 			for j := 0; j < in_wid; j++ {
// 				for b := 0; b < batch; b++ {
// 					if (i < in_wid-pad) && (j < in_wid-pad) {
// 						input[i*in_wid*batch+j*batch+b] = k
// 						k += 0.0
// 					}
// 				}
// 			}
// 		}
// 	}

// 	prt_mat(input, batch, 0)

// 	ker_in := make([]float64, batch*batch*ker_size)
// 	for i := range ker_in {
// 		ker_in[i] = 1.0 * float64(i) //* float64(i) / float64(batch*batch*ker_size)
// 	}
// 	bn_a := make([]float64, batch)
// 	bn_b := make([]float64, batch)
// 	b_coeffs := make([]float64, N)
// 	for i := range bn_a {
// 		bn_a[i] = 1.0
// 		bn_b[i] = 0.0
// 		for j := 0; j < in_wid; j++ {
// 			for k := 0; k < in_wid; k++ {
// 				b_coeffs[i+(j+k*in_wid)*batch] = bn_b[i]
// 			}
// 		}
// 	}

// 	start = time.Now()
// 	pl_ker := prepKer_in(params, encoder, ker_in, bn_a, in_wid, ker_wid, batch, batch, params.MaxLevel(), 0, trans)
// 	fmt.Printf("Plaintext (kernel) preparation, Done in %s \n", time.Since(start))

// 	fmt.Println("vec size: ", N)
// 	fmt.Println("input width: ", in_wid)
// 	fmt.Println("kernel width: ", ker_wid)
// 	fmt.Println("num batches: ", batch)
// 	fmt.Println("Input matrix: ")
// 	prt_vec(input)
// 	fmt.Println("Ker1_in (1st part): ")
// 	prt_vec(ker_in)

// 	start = time.Now()
// 	plain_tmp := ckks.NewPlaintext(params, params.MaxLevel(), params.Scale()) // contain plaintext values
// 	encoder.EncodeCoeffs(input, plain_tmp)
// 	ctxt_input := encryptor.EncryptNew(plain_tmp)
// 	fmt.Printf("Encryption done in %s \n", time.Since(start))

// 	fmt.Println()
// 	fmt.Println("===============================================")
// 	fmt.Println("     			   EVALUATION					")
// 	fmt.Println("===============================================")
// 	fmt.Println()

// 	start = time.Now()
// 	ct_result := conv_then_pack(params, pack_evaluator, ctxt_input, pl_ker, plain_idx, batch)
// 	// for Batch Normalization (BN)
// 	pl_bn_b := ckks.NewPlaintext(params, ct_result.Level(), ct_result.Scale) // contain plaintext values
// 	encoder.EncodeCoeffs(b_coeffs, pl_bn_b)
// 	encoder.ToNTT(pl_bn_b)
// 	evaluator.Add(ct_result, pl_bn_b, ct_result)
// 	fmt.Printf("Conv (with BN) Done in %s \n", time.Since(start))

// 	fmt.Println()
// 	fmt.Println("=========================================")
// 	fmt.Println("              DECRYPTION                 ")
// 	fmt.Println("=========================================")
// 	fmt.Println()

// 	start = time.Now()
// 	decryptor.Decrypt(ct_result, plain_tmp)
// 	cfs_tmp := encoder.DecodeCoeffs(plain_tmp)
// 	// cfs_tmp := reshape_conv_out(encoder.DecodeCoeffs(plain_tmp), in_wid, batch)
// 	fmt.Printf("Decryption Done in %s \n", time.Since(start))

// 	if printResult {
// 		fmt.Print("Result: \n")
// 		prt_mat(cfs_tmp, batch, 0)
// 	}

// 	return cfs_tmp
// }
