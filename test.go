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
	pow := 4
	print := false
	logN := 16
	in_wid := [4]int{4, 8, 16, 32}
	max_batch := [4]int{1024, 256, 64, 16}
	batch := [4]int{512, 128, 64, 1}
	// max_batch := [4]int{64, 16, 4, 1}
	// batch := [4]int{16, 4, 2, 1}
	ker_wid := 5
	N := (1 << logN)
	// st_batch := N / (2 * in_wid * 2 * in_wid) // We also consider zero-paddings  // must be adjusted when in_wid is not power of 2
	// end_batch := 4
	ECD_LV := 1

	// parameter generation (comment out when do other test)
	var btp *ckks.Bootstrapper
	btpParams := ckks.DefaultBootstrapParams[6]
	params, err := btpParams.Params()
	if err != nil {
		panic(err)
	}
	fmt.Printf("CKKS parameters: logN = %d, logSlots = %d, h = %d, logQP = %d, levels = %d, scale= 2^%f, sigma = %f \n",
		params.LogN(), params.LogSlots(), btpParams.H, params.LogQP(), params.QCount(), math.Log2(params.Scale()), params.Sigma())

	// Generate rotations for EXT_FULL
	var rotations []int
	r_idx := make([]map[int][]int, 4)
	m_idx := make([]map[int][]int, 4)
	for pos := 0; pos < 4; pos++ {
		r_idx[pos], m_idx[pos] = gen_extend_full(N/2, in_wid[1], pos, true, true)
		for k := range r_idx[pos] {
			rotations = append(rotations, k)
		}
		for k := range m_idx[pos] {
			rotations = append(rotations, k)
		}
	}
	r_idx1 := make([]map[int][]int, 4)
	m_idx1 := make([]map[int][]int, 4)
	for pos := 0; pos < 4; pos++ {
		r_idx1[pos], m_idx1[pos] = gen_extend_full(N/2, in_wid[2], pos, true, true)
		for k := range r_idx1[pos] {
			rotations = append(rotations, k)
		}
		for k := range m_idx1[pos] {
			rotations = append(rotations, k)
		}
	}
	r_idx2 := make([]map[int][]int, 4)
	m_idx2 := make([]map[int][]int, 4)
	for pos := 0; pos < 4; pos++ {
		r_idx2[pos], m_idx2[pos] = gen_extend_full(N/2, in_wid[3], pos, true, true)
		for k := range r_idx2[pos] {
			rotations = append(rotations, k)
		}
		for k := range m_idx2[pos] {
			rotations = append(rotations, k)
		}
	}
	fmt.Println("Rotations: ", rotations)

	// Scheme context and keys for evaluation (no Boot)
	kgen := ckks.NewKeyGenerator(params)
	sk, _ := kgen.GenKeyPairSparse(btpParams.H)
	rlk := kgen.GenRelinearizationKey(sk, 2)
	rotkeys := kgen.GenRotationKeysForRotations(rotations, false, sk)
	encoder := ckks.NewEncoder(params)
	decryptor := ckks.NewDecryptor(params, sk)
	encryptor := ckks.NewEncryptor(params, sk)
	evaluator := ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk, Rtks: rotkeys})

	plain_idx, pack_evaluator := gen_idxNlogs(ECD_LV, kgen, sk, encoder, params)

	fmt.Println("Generating bootstrapping keys...")
	start = time.Now()
	rotations = btpParams.RotationsForBootstrapping(params.LogSlots())
	rotkeys = kgen.GenRotationKeysForRotations(rotations, true, sk)
	btpKey := ckks.BootstrappingKey{Rlk: rlk, Rtks: rotkeys}
	if btp, err = ckks.NewBootstrapper_mod(params, btpParams, btpKey); err != nil {
		panic(err)
	}
	fmt.Printf("Done in %s \n", time.Since(start))

	for iter := 1; iter < 2; iter++ {
		name_iter := fmt.Sprintf("%04d", iter)

		input := readTxt("./inputs/input_" + name_iter + ".txt")
		ext_input := inputExt(input, logN, in_wid[0], false) // Takes arranged input (assume intermediate layers)  // print only outputs first (st_batch) batches

		// circ_rows := readTxt("./varaibles/rows.txt")
		// circ_mat := make([][]float64, 8)
		// plain_mat := make([]*ckks.Plaintext, 8)
		// for i := 0; i < len(circ_mat); i++ {
		// 	circ_mat[i] = encode_circ(circ_rows[N/2*i:N/2*(i+1)], 16, N)
		// 	plain_mat[i] = ckks.NewPlaintext(params, ECD_LV, params.Scale())
		// 	encoder.EncodeCoeffs(circ_mat[i], plain_mat[i])
		// 	encoder.ToNTT(plain_mat[i])
		// }

		// test_input := readTxt("./inputs/vec_input_" + name_iter + ".txt")
		// enc_test_input := make([]*ckks.Ciphertext, 8)
		// test_tmp := ckks.NewPlaintext(params, ECD_LV, params.Scale())
		// for i := 0; i < len(enc_test_input); i++ {
		// 	encoder.EncodeCoeffs(encode_circ_in(test_input, i, 16, N), test_tmp)
		// 	enc_test_input[i] = encryptor.EncryptNew(test_tmp)
		// }

		// var test_result *ckks.Ciphertext
		// for i := 0; i < len(enc_test_input); i++ {
		// 	if i == 0 {
		// 		test_result = evaluator.MulNew(enc_test_input[i], plain_mat[i])
		// 	} else {
		// 		evaluator.Add(test_result, evaluator.MulNew(enc_test_input[i], plain_mat[i]), test_result)
		// 	}
		// }

		// input := make([]float64, N)
		// for i := range input {
		// 	input[i] = 1.0 * float64(ext_input[i]) / float64(in_wid[0]*in_wid[0]*batch[0])
		// }

		start := time.Now()
		plain_in := ckks.NewPlaintext(params, 1, params.Scale()) // contain plaintext values
		encoder.EncodeCoeffs(ext_input, plain_in)
		ctxt_in := encryptor.EncryptNew(plain_in)

		// zeros := make([]complex128, params.Slots())
		// plain_in = encoder.EncodeNew(zeros, params.LogSlots())
		// ctxt0 := encryptor.EncryptNew(plain_in)

		fmt.Printf("Encryption: Done in %s \n", time.Since(start))

		if print {
			fmt.Println("vec size: ", N)
			fmt.Println("input width: ", in_wid)
			fmt.Println("kernel width: ", ker_wid)
			fmt.Println("num batches (in 1 ctxt with padding): ", max_batch[0])
		}

		ker1 := readTxt("./variables/conv1.txt")
		a1 := readTxt("./variables/a1.txt")
		b1 := readTxt("./variables/b1.txt")
		b1_coeffs := make([]float64, N)
		for i := range b1 {
			for j := 0; j < in_wid[1]; j++ {
				for k := 0; k < in_wid[1]; k++ {
					b1_coeffs[i+(j+k*in_wid[1]*2)*max_batch[1]] = b1[i]
				}
			}
		}

		pl_ker := prepKer_in_trans(params, encoder, encryptor, ker1, a1, in_wid[0], ker_wid, max_batch[0], max_batch[1], batch[0], batch[1], ECD_LV)

		fmt.Print("Boot in: ")
		fmt.Println()
		fmt.Println("Precision of values vs. ciphertext")
		in_cfs := printDebugCfs(params, ctxt_in, ext_input, decryptor, encoder)

		fmt.Println("Bootstrapping... Ours (until CtoS):")
		start = time.Now()
		ctxt1, ctxt2, _ := btp.BootstrappConv_CtoS(ctxt_in)
		fmt.Printf("Done in %s \n", time.Since(start))
		fmt.Println("after Boot: LV = ", ctxt1.Level(), " Scale = ", math.Log2(ctxt1.Scale))

		// Only for checking the correctness
		in_cfs_1_pBoot := make([]float64, params.Slots())
		in_cfs_2_pBoot := make([]float64, params.Slots())
		in_slots := make([]complex128, params.Slots()) // first part of ceffs
		for i := range in_cfs_1_pBoot {
			in_cfs_1_pBoot[i] = in_cfs[reverseBits(uint32(i), params.LogSlots())] // first part of coeffs
			in_cfs_2_pBoot[i] = in_cfs[reverseBits(uint32(i), params.LogSlots())+uint32(params.Slots())]
			in_slots[i] = complex(in_cfs_1_pBoot[i]/math.Pow(2, float64(pow)), 0)
		}
		ext1_tmp := extend_full_fl(in_cfs_1_pBoot, in_wid[1], 3, true, true)
		ext2_tmp := extend_full_fl(in_cfs_2_pBoot, in_wid[1], 3, true, true)
		for i := range in_cfs_1_pBoot {
			in_cfs_1_pBoot[i] = ext1_tmp[reverseBits(uint32(i), params.LogSlots())]
			in_cfs_2_pBoot[i] = ext2_tmp[reverseBits(uint32(i), params.LogSlots())]
		}
		in_cfs_pBoot := append(in_cfs_1_pBoot, in_cfs_2_pBoot...) // After rot(ext) and boot

		in_slots = printDebug(params, ctxt1, in_slots, decryptor, encoder) // Compare before & after CtoS

		start = time.Now()
		// evaluator.MultByConst(ctxt1, 1.000000001, ctxt1)
		// evaluator.DropLevel(ctxt1, 10)
		evaluator.MulByPow2(ctxt1, pow, ctxt1)
		evaluator.DropLevel(ctxt1, ctxt1.Level()-3)

		// ctxt1 = evalReLU(params, evaluator, ctxt1, 1.0)
		fmt.Printf("NO ReLU Done in %s \n", time.Since(start))

		values_ReLU := make([]complex128, len(in_slots))
		for i := range values_ReLU {
			values_ReLU[i] = complex(math.Pow(2, float64(pow)), 0) * in_slots[i] // complex(math.Max(0, real(in_slots[i])), 0)
		}
		printDebug(params, ctxt1, values_ReLU, decryptor, encoder)

		ext_ctxt1 := make([]*ckks.Ciphertext, 4) // for extend (rotation) of ctxt_in
		// ext_ctxt2 := make([]*ckks.Ciphertext, 4)  // do not need if we use po2 inputs dims
		ciphertext := make([]*ckks.Ciphertext, 4) // after Bootstrapping

		ctxt2 = nil
		// evaluator.DropLevel(ctxt2, ctxt2.Level()-2)
		start = time.Now()
		for pos := 0; pos < 4; pos++ {
			ext_ctxt1[pos] = ext_ctxt(evaluator, encoder, ctxt1, r_idx[pos], params)
			// fmt.Println(ext_ctxt1[pos].Level(), ctxt2.Level(), ext_ctxt1[pos].Scale, ctxt2.Scale)
			// ext_ctxt2[pos] = ext_ctxt(evaluator, encoder, ctxt2, r_idx[pos], m_idx[pos], params)
			ciphertext[pos] = btp.BootstrappConv_StoC(ext_ctxt1[pos], ctxt2)
			evaluator.Rescale(ciphertext[pos], params.Scale(), ciphertext[pos])
		}
		fmt.Printf("Boot (StoC) Done in %s \n", time.Since(start))

		fmt.Printf("Boot out: ")
		// for i := range in_cfs_pBoot {
		// 	in_cfs_pBoot[i] = math.Max(0, in_cfs_pBoot[i])
		// }
		printDebugCfs(params, ciphertext[3], in_cfs_pBoot, decryptor, encoder)

		ctxt_result := conv_then_pack_trans(params, pack_evaluator, ciphertext, pl_ker, plain_idx, max_batch[1])

		// for Batch Normalization (BN)
		plain_in = ckks.NewPlaintext(params, ctxt_result.Level(), ctxt_result.Scale) // contain plaintext values
		encoder.EncodeCoeffs(b1_coeffs, plain_in)
		encoder.ToNTT(plain_in)
		evaluator.Add(ctxt_result, plain_in, ctxt_result)

		plain_out := ckks.NewPlaintext(params, ctxt_result.Level(), params.Scale())
		start = time.Now()
		decryptor.Decrypt(ctxt_result, plain_out)
		pre_boot := encoder.DecodeCoeffs(plain_out)
		cfs_tmp := reshape_conv_out(encoder.DecodeCoeffs(plain_out), in_wid[1], max_batch[1])

		if print {
			fmt.Println()
			fmt.Println("=========================================")
			fmt.Println("              DECRYPTION                 ")
			fmt.Println("=========================================")
			fmt.Println()

			fmt.Print("Result: \n")
			prt_mat(cfs_tmp, max_batch[1], in_wid[1])
		}
		fmt.Printf("(Layer 1) Done in %s \n", time.Since(start))

		// // To see each matrix
		// cfs_tmp = encoder.DecodeCoeffs(plain_out)
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

		ker2 := readTxt("./variables/conv2.txt")
		a2 := readTxt("./variables/a2.txt")
		b2 := readTxt("./variables/b2.txt")
		b2_coeffs := make([]float64, N)
		for i := range b2 {
			for j := 0; j < in_wid[2]; j++ {
				for k := 0; k < in_wid[2]; k++ {
					b2_coeffs[i+(j+k*in_wid[2]*2)*max_batch[2]] = b2[i]
				}
			}
		}

		pl_ker = prepKer_in_trans(params, encoder, encryptor, ker2, a2, in_wid[1], ker_wid, max_batch[1], max_batch[2], batch[1], batch[2], ECD_LV)

		// fmt.Print("Boot in: ")
		// fmt.Println()
		// fmt.Println("Precision of values vs. ciphertext")
		// in_cfs = printDebugCfs(params, ctxt_result, pre_boot, decryptor, encoder)
		in_cfs = pre_boot
		ctxt_in.Copy(ctxt_result)
		// ctxt_in.SetScalingFactor(ctxt_in.Scale * 64)

		fmt.Println("Bootstrapping... Ours (until CtoS):")
		start = time.Now()
		ctxt1, _, _ = btp.BootstrappConv_CtoS(ctxt_in)
		fmt.Printf("Done in %s \n", time.Since(start))

		// Only for checking the correctness
		for i := range in_cfs_1_pBoot {
			in_cfs_1_pBoot[i] = in_cfs[reverseBits(uint32(i), params.LogSlots())] // first part of coeffs
			in_cfs_2_pBoot[i] = 0                                                 // in_cfs[reverseBits(uint32(i), params.LogSlots())+uint32(params.Slots())]
			in_slots[i] = complex(in_cfs_1_pBoot[i]/math.Pow(2, float64(pow)), 0)
		}
		ext1_tmp = extend_full_fl(in_cfs_1_pBoot, in_wid[2], 0, true, true)
		ext2_tmp = extend_full_fl(in_cfs_2_pBoot, in_wid[2], 0, true, true)
		for i := range in_cfs_1_pBoot {
			in_cfs_1_pBoot[i] = ext1_tmp[reverseBits(uint32(i), params.LogSlots())]
			in_cfs_2_pBoot[i] = ext2_tmp[reverseBits(uint32(i), params.LogSlots())]
		}
		in_cfs_pBoot = append(in_cfs_1_pBoot, in_cfs_2_pBoot...) // After rot(ext) and boot
		in_slots = printDebug(params, ctxt1, in_slots, decryptor, encoder)

		start = time.Now()

		plain_ch := ckks.NewPlaintext(params, ctxt1.Level(), params.Scale())
		decryptor.Decrypt(ctxt1, plain_ch)
		check := encoder.Decode(plain_ch, logN-1)
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

		ctxt1 = evalReLU(params, evaluator, ctxt1, alpha)
		evaluator.MulByPow2(ctxt1, pow, ctxt1)
		fmt.Printf("ReLU Done in %s \n", time.Since(start))

		for i := range values_ReLU {
			values_ReLU[i] = complex(math.Pow(2, float64(pow)), 0) * complex(math.Max(0, real(in_slots[i]))+alpha*math.Min(0, real(in_slots[i])), 0)
		}
		printDebug(params, ctxt1, values_ReLU, decryptor, encoder)

		start = time.Now()
		for pos := 0; pos < 4; pos++ {
			ext_ctxt1[pos] = ext_ctxt(evaluator, encoder, ctxt1, r_idx1[pos], params)
			// ext_ctxt2[pos] = ext_ctxt(evaluator, encoder, ctxt2, r_idx1[pos], m_idx1[pos], params)
			ciphertext[pos] = btp.BootstrappConv_StoC(ext_ctxt1[pos], ctxt2)
			evaluator.Rescale(ciphertext[pos], params.Scale(), ciphertext[pos])
		}
		fmt.Printf("Boot (StoC) Done in %s \n", time.Since(start))

		fmt.Printf("Boot out: ")
		for i := range in_cfs_pBoot {
			in_cfs_pBoot[i] = math.Max(0, in_cfs_pBoot[i]) + alpha*math.Min(0, in_cfs_pBoot[i])
		}
		printDebugCfs(params, ciphertext[0], in_cfs_pBoot, decryptor, encoder)

		ctxt_result = conv_then_pack_trans(params, pack_evaluator, ciphertext, pl_ker, plain_idx, max_batch[2])

		// for BN
		plain_in = ckks.NewPlaintext(params, ctxt_result.Level(), ctxt_result.Scale) // contain plaintext values
		encoder.EncodeCoeffs(b2_coeffs, plain_in)
		encoder.ToNTT(plain_in)
		evaluator.Add(ctxt_result, plain_in, ctxt_result)

		start = time.Now()
		decryptor.Decrypt(ctxt_result, plain_out)
		pre_boot = encoder.DecodeCoeffs(plain_out)
		cfs_tmp = reshape_conv_out(encoder.DecodeCoeffs(plain_out), in_wid[2], max_batch[2])

		if print {
			fmt.Println()
			fmt.Println("=========================================")
			fmt.Println("              DECRYPTION                 ")
			fmt.Println("=========================================")
			fmt.Println()
			fmt.Print("Result: \n")
			prt_mat(cfs_tmp, max_batch[2], in_wid[2])
		}
		fmt.Printf("(Layer 2) Done in %s \n", time.Since(start))

		fmt.Println()
		fmt.Println("=========================================")
		fmt.Println("              LAYER 3	                 ")
		fmt.Println("=========================================")
		fmt.Println()

		ker3 := readTxt("./variables/conv3.txt")
		a3 := make([]float64, batch[3])
		for i := range a3 {
			a3[i] = 1
		}
		pl_ker = prepKer_in_trans(params, encoder, encryptor, ker3, a3, in_wid[2], ker_wid, max_batch[2], max_batch[3], batch[2], batch[3], ECD_LV)

		// fmt.Print("Boot in: ")
		// fmt.Println()
		// fmt.Println("Precision of values vs. ciphertext")
		// in_cfs = printDebugCfs(params, ctxt_result, pre_boot, decryptor, encoder)
		in_cfs = pre_boot
		ctxt_in.Copy(ctxt_result)
		// ctxt_in.SetScalingFactor(ctxt_in.Scale * 16)

		fmt.Println("Bootstrapping... Ours (until CtoS):")
		start = time.Now()
		ctxt1, _, _ = btp.BootstrappConv_CtoS(ctxt_in)
		fmt.Printf("Done in %s \n", time.Since(start))

		// Only for checking the correctness
		for i := range in_cfs_1_pBoot {
			in_cfs_1_pBoot[i] = in_cfs[reverseBits(uint32(i), params.LogSlots())] // first part of coeffs
			in_cfs_2_pBoot[i] = 0                                                 // in_cfs[reverseBits(uint32(i), params.LogSlots())+uint32(params.Slots())]
			in_slots[i] = complex(in_cfs_1_pBoot[i]/math.Pow(2, float64(pow)), 0)
		}
		ext1_tmp = extend_full_fl(in_cfs_1_pBoot, in_wid[3], 0, true, true)
		ext2_tmp = extend_full_fl(in_cfs_2_pBoot, in_wid[3], 0, true, true)
		for i := range in_cfs_1_pBoot {
			in_cfs_1_pBoot[i] = ext1_tmp[reverseBits(uint32(i), params.LogSlots())]
			in_cfs_2_pBoot[i] = ext2_tmp[reverseBits(uint32(i), params.LogSlots())]
		}
		in_cfs_pBoot = append(in_cfs_1_pBoot, in_cfs_2_pBoot...) // After rot(ext) and boot
		in_slots = printDebug(params, ctxt1, in_slots, decryptor, encoder)

		start = time.Now()
		// evaluator.MultByConst(ctxt1, 1.000000001, ctxt1)
		// evaluator.DropLevel(ctxt1, 10)

		plain_ch = ckks.NewPlaintext(params, ctxt1.Level(), params.Scale())
		decryptor.Decrypt(ctxt1, plain_ch)
		check = encoder.Decode(plain_ch, logN-1)
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

		ctxt1 = evalReLU(params, evaluator, ctxt1, alpha)
		evaluator.MulByPow2(ctxt1, pow, ctxt1)
		fmt.Printf("ReLU Done in %s \n", time.Since(start))

		for i := range values_ReLU {
			values_ReLU[i] = complex(math.Pow(2, float64(pow)), 0) * complex(math.Max(0, real(in_slots[i]))+alpha*math.Min(0, real(in_slots[i])), 0)
		}
		printDebug(params, ctxt1, values_ReLU, decryptor, encoder)

		start = time.Now()
		for pos := 0; pos < 4; pos++ {
			ext_ctxt1[pos] = ext_ctxt(evaluator, encoder, ctxt1, r_idx2[pos], params)
			// ext_ctxt2[pos] = ext_ctxt(evaluator, encoder, ctxt2, r_idx2[pos], m_idx2[pos], params)
			ciphertext[pos] = btp.BootstrappConv_StoC(ext_ctxt1[pos], ctxt2)
			evaluator.Rescale(ciphertext[pos], params.Scale(), ciphertext[pos])
		}
		fmt.Printf("Boot (StoC) Done in %s \n", time.Since(start))

		fmt.Printf("Boot out: ")
		for i := range in_cfs_pBoot {
			in_cfs_pBoot[i] = math.Max(0, in_cfs_pBoot[i]) + alpha*math.Min(0, in_cfs_pBoot[i])
		}
		printDebugCfs(params, ciphertext[0], in_cfs_pBoot, decryptor, encoder)

		ctxt_result = conv_then_pack_trans(params, pack_evaluator, ciphertext, pl_ker, plain_idx, max_batch[3])
		// ctxt_result = conv_then_pack(params, pack_evaluator, ciphertext, pl_ker, plain_idx, batch[2])

		start = time.Now()
		decryptor.Decrypt(ctxt_result, plain_out)
		pre_boot = encoder.DecodeCoeffs(plain_out)
		cfs_tmp = reshape_conv_out(encoder.DecodeCoeffs(plain_out), in_wid[3], max_batch[3])

		if print {
			fmt.Println()
			fmt.Println("=========================================")
			fmt.Println("              DECRYPTION                 ")
			fmt.Println("=========================================")
			fmt.Println()
			fmt.Print("Result: \n")
			prt_mat(cfs_tmp, max_batch[3], in_wid[3])
		}
		fmt.Printf("(Layer 3) Done in %s \n", time.Since(start))

		output := make([]float64, in_wid[3]*in_wid[3])
		for i := range output {
			output[i] = cfs_tmp[max_batch[3]*i]
		}
		writeTxt("result_"+name_iter+".txt", output)
	}
}

// BaseLine Conv without boot, Assume full batch with Po2 in_wid & N
// Normal Conv without output modification (e.g., trimming or expanding)
// Input does not need padding
func testConv_noBoot_BL(log_slots, in_wid, ker_wid int, printResult bool) {
	slots := (1 << log_slots)
	in_size := in_wid * in_wid
	batch := slots / in_size
	ker_size := ker_wid * ker_wid

	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{ // Schemes parameters are created from scratch
		LogN:     log_slots + 1,
		LogQ:     []int{log_out_scale + log_c_scale, log_in_scale},
		LogP:     []int{60},
		Sigma:    rlwe.DefaultSigma,
		LogSlots: log_slots,
		Scale:    float64(1 << log_in_scale),
	})
	if err != nil {
		panic(err)
	}

	fmt.Println("Q ", params.Q())

	fmt.Println()
	fmt.Println("========================================================")
	fmt.Println(" INSTANTIATING SCHEME & PLAINTEXT CREATION & Encryption ")
	fmt.Println("========================================================")
	fmt.Println()

	var rotations []int
	for k := -(ker_wid / 2); k <= ker_wid/2; k++ {
		for k2 := -(ker_wid / 2); k2 <= ker_wid/2; k2++ {
			rotations = append(rotations, k*in_wid+k2)
		}
	}
	for k := 1; k <= batch; k++ {
		rotations = append(rotations, k*in_size)
	}
	// fmt.Println("Rotations: ", rotations)

	start := time.Now()
	kgen := ckks.NewKeyGenerator(params)
	sk := kgen.GenSecretKey()
	rotkeys := kgen.GenRotationKeysForRotations(rotations, false, sk)
	encryptor := ckks.NewEncryptor(params, sk)
	decryptor := ckks.NewDecryptor(params, sk)
	encoder := ckks.NewEncoder(params)
	evaluator := ckks.NewEvaluator(params, rlwe.EvaluationKey{Rtks: rotkeys})

	fmt.Printf("Keygen, Done in %s \n", time.Since(start))

	fmt.Println()
	fmt.Printf("CKKS parameters: logN = %d, logSlots = %d, logQP = %d, levels = %d, scale= %f, sigma = %f \n",
		params.LogN(), params.LogSlots(), params.LogQP(), params.MaxLevel()+1, params.Scale(), params.Sigma())

	input := make([]float64, slots)
	for i := range input {
		input[i] = 1.0 * float64(i) / float64(len(input))
	}
	input_rs := reshape_input_BL(input, in_wid)
	if printResult {
		prt_mat_BL(input_rs, batch, 0)
	}

	ker_in := make([]float64, batch*batch*ker_size)
	for i := range ker_in {
		ker_in[i] = 1.0 * float64(i) / float64(len(ker_in)) //0.1 * float64(i) / float64(batch*batch*ker_size)
	}
	bn_a := make([]float64, batch)
	for i := range bn_a {
		bn_a[i] = 1.0
	}
	start = time.Now()
	max_ker_rs := reshape_ker_BL(ker_in, bn_a, ker_wid, batch, batch, batch)
	fmt.Printf("Plaintext (kernel) preparation, Done in %s \n", time.Since(start))

	fmt.Println("vec size: ", slots)
	fmt.Println("input width: ", in_wid)
	fmt.Println("kernel width: ", ker_wid)
	fmt.Println("num batches: ", batch)
	fmt.Println("Input matrix: ")
	real_input_rs := make([]float64, len(input_rs))
	for i, elt := range input_rs {
		real_input_rs[i] = real(elt)
	}
	prt_vec(real_input_rs)
	fmt.Println("Ker1_in (1st to 1st part): ")
	for i := 0; i < ker_wid; i++ {
		for j := 0; j < ker_wid; j++ {
			fmt.Print(max_ker_rs[i][j][0][0], ", ")
		}
	}
	fmt.Print("\n\n")

	start = time.Now()
	plain_tmp := ckks.NewPlaintext(params, params.MaxLevel(), params.Scale()) // contain plaintext values
	encoder.Encode(plain_tmp, input_rs, log_slots)
	ctxt_input := encryptor.EncryptNew(plain_tmp)
	fmt.Printf("Encryption done in %s \n", time.Since(start))

	fmt.Println()
	fmt.Println("===============================================")
	fmt.Println("     			   EVALUATION					")
	fmt.Println("===============================================")
	fmt.Println()

	start = time.Now()
	ct_inputs_rots := preConv_BL(evaluator, ctxt_input, in_wid, ker_wid)
	fmt.Printf("preConv done in %s \n", time.Since(start))

	var ct_result *ckks.Ciphertext
	for i := 0; i < batch; i++ {
		ct_tmp := postConv_BL(params, encoder, evaluator, ct_inputs_rots, in_wid, ker_wid, i, max_ker_rs)
		if i == 0 {
			ct_result = ct_tmp
		} else {
			evaluator.Add(ct_result, evaluator.RotateNew(ct_tmp, i*in_size), ct_result)
		}
	}
	fmt.Printf("Eval Done in %s \n", time.Since(start))

	fmt.Println()
	fmt.Println("=========================================")
	fmt.Println("              DECRYPTION                 ")
	fmt.Println("=========================================")
	fmt.Println()

	start = time.Now()
	decryptor.Decrypt(ct_result, plain_tmp)
	vals_tmp := encoder.Decode(plain_tmp, log_slots)
	real_vals_tmp := make([]float64, len(vals_tmp))
	for i, elt := range vals_tmp {
		real_vals_tmp[i] = real(elt)
	}
	fmt.Printf("Decryption Done in %s \n", time.Since(start))

	if printResult {
		fmt.Print("Result: \n")
		prt_mat_BL(vals_tmp, batch, 0)
	}
}

// BaseLine Conv without boot, Assume full batch with Po2 in_wid & N
// Normal Conv without output modification (e.g., trimming or expanding)
// Input does not need padding
func testConv_BNRelu_BL(log_slots, in_wid, ker_wid int, printResult bool) {
	slots := (1 << log_slots)
	in_size := in_wid * in_wid
	batch := slots / in_size
	ker_size := ker_wid * ker_wid
	ECD_LV := 1
	alpha := 0.0

	var btp *ckks.Bootstrapper
	btpParams := ckks.DefaultBootstrapParams[7]
	params, err := btpParams.Params()
	if err != nil {
		panic(err)
	}
	fmt.Printf("CKKS parameters: logN = %d, logSlots = %d, h = %d, logQP = %d, levels = %d, scale= 2^%f, sigma = %f \n",
		params.LogN(), params.LogSlots(), btpParams.H, params.LogQP(), params.QCount(), math.Log2(params.Scale()), params.Sigma())

	var rotations []int
	for k := -(ker_wid / 2); k <= ker_wid/2; k++ {
		for k2 := -(ker_wid / 2); k2 <= ker_wid/2; k2++ {
			rotations = append(rotations, k*in_wid+k2)
		}
	}
	for k := 1; k <= batch; k++ {
		rotations = append(rotations, k*in_size)
	}
	// fmt.Println("Rotations: ", rotations)

	start := time.Now()
	kgen := ckks.NewKeyGenerator(params)
	sk, _ := kgen.GenKeyPairSparse(btpParams.H)
	rlk := kgen.GenRelinearizationKey(sk, 2)
	rotkeys := kgen.GenRotationKeysForRotations(rotations, false, sk)
	encryptor := ckks.NewEncryptor(params, sk)
	decryptor := ckks.NewDecryptor(params, sk)
	encoder := ckks.NewEncoder(params)
	evaluator := ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk, Rtks: rotkeys})

	fmt.Println("Generating bootstrapping keys...")
	start = time.Now()
	rotations = btpParams.RotationsForBootstrapping(params.LogSlots())
	rotkeys = kgen.GenRotationKeysForRotations(rotations, true, sk)
	btpKey := ckks.BootstrappingKey{Rlk: rlk, Rtks: rotkeys}
	if btp, err = ckks.NewBootstrapper(params, btpParams, btpKey); err != nil {
		panic(err)
	}
	fmt.Printf("Done in %s \n", time.Since(start))

	input := make([]float64, slots)
	for i := range input {
		input[i] = 1.0 * float64(i) / float64(len(input))
	}
	input_rs := reshape_input_BL(input, in_wid)
	if printResult {
		prt_mat_BL(input_rs, batch, 0)
	}

	ker_in := make([]float64, batch*batch*ker_size)
	for i := range ker_in {
		ker_in[i] = 1.0 * float64(i) / float64(len(ker_in)) //0.1 * float64(i) / float64(batch*batch*ker_size)
	}
	bn_a := make([]float64, batch)
	for i := range bn_a {
		bn_a[i] = 0.001
	}
	start = time.Now()
	max_ker_rs := reshape_ker_BL(ker_in, bn_a, ker_wid, batch, batch, batch)
	fmt.Printf("Plaintext (kernel) preparation, Done in %s \n", time.Since(start))

	fmt.Println("vec size: ", slots)
	fmt.Println("input width: ", in_wid)
	fmt.Println("kernel width: ", ker_wid)
	fmt.Println("num batches: ", batch)
	fmt.Println("Input matrix: ")
	real_input_rs := make([]float64, len(input_rs))
	for i, elt := range input_rs {
		real_input_rs[i] = real(elt)
	}
	prt_vec(real_input_rs)
	fmt.Println("Ker1_in (1st to 1st part): ")
	for i := 0; i < ker_wid; i++ {
		for j := 0; j < ker_wid; j++ {
			fmt.Print(max_ker_rs[i][j][0][0], ", ")
		}
	}
	fmt.Print("\n\n")

	start = time.Now()
	plain_tmp := ckks.NewPlaintext(params, ECD_LV, params.Scale()) // contain plaintext values
	encoder.Encode(plain_tmp, input_rs, log_slots)
	ct_input := encryptor.EncryptNew(plain_tmp)
	fmt.Printf("Encryption done in %s \n", time.Since(start))

	fmt.Println()
	fmt.Println("===============================================")
	fmt.Println("     			   EVALUATION					")
	fmt.Println("===============================================")
	fmt.Println()

	start = time.Now()
	ct_inputs_rots := preConv_BL(evaluator, ct_input, in_wid, ker_wid)
	fmt.Printf("preConv done in %s \n", time.Since(start))

	var ct_conv *ckks.Ciphertext
	for i := 0; i < batch; i++ {
		ct_tmp := postConv_BL(params, encoder, evaluator, ct_inputs_rots, in_wid, ker_wid, i, max_ker_rs)
		if i == 0 {
			ct_conv = ct_tmp
		} else {
			evaluator.Add(ct_conv, evaluator.RotateNew(ct_tmp, i*in_size), ct_conv)
		}
	}
	fmt.Printf("Eval (Conv) Done in %s \n", time.Since(start))

	fmt.Println()
	fmt.Println("=========================================")
	fmt.Println("              DECRYPTION                 ")
	fmt.Println("=========================================")
	fmt.Println()

	start = time.Now()
	decryptor.Decrypt(ct_conv, plain_tmp)
	vals_tmp := encoder.Decode(plain_tmp, log_slots)
	real_vals_tmp := make([]float64, len(vals_tmp))
	for i, elt := range vals_tmp {
		real_vals_tmp[i] = real(elt)
	}
	fmt.Printf("Decryption (Conv) Done in %s \n", time.Since(start))

	if printResult {
		fmt.Print("Result: \n")
		prt_mat_BL(vals_tmp, batch, 0)
	}

	fmt.Println("Boot in: ")
	fmt.Println("inputs: ")
	in_vals := printDebug(params, ct_conv, vals_tmp, decryptor, encoder)

	fmt.Println("Bootstrapping... (original):")
	start_boot := time.Now()
	ct_boot := btp.Bootstrapp(ct_conv)
	fmt.Printf("Done in %s \n", time.Since(start_boot))
	fmt.Println("after Boot: LV = ", ct_boot.Level(), " Scale = ", math.Log2(ct_boot.Scale))
	fmt.Println()
	fmt.Println("Precision of ciphertext vs. Bootstrapp(ciphertext)")
	in_relu := printDebug(params, ct_boot, in_vals, decryptor, encoder)
	for i, elt := range in_relu {
		in_relu[i] = complex(math.Max(0, real(elt)), 0)
	}

	start = time.Now()
	in_relu = printDebug(params, ct_boot, in_relu, decryptor, encoder)
	evaluator.Rescale(ct_boot, params.Scale(), ct_boot)
	evaluator.ScaleUp(ct_boot, params.Scale()/ct_boot.Scale, ct_boot)
	fmt.Println("after Rescale: LV = ", ct_boot.Level(), " Scale = 2^", math.Log2(ct_boot.Scale))
	in_relu = printDebug(params, ct_boot, in_relu, decryptor, encoder)

	ct_relu := evalReLU(params, evaluator, ct_boot, alpha)
	evaluator.SetScale(ct_relu, params.Scale())
	fmt.Printf("Relu Done in %s \n", time.Since(start))
	fmt.Println("Precision of relu")
	printDebug(params, ct_relu, in_relu, decryptor, encoder)

	fmt.Println()
	fmt.Println("=========================================")
	fmt.Println("              DECRYPTION                 ")
	fmt.Println("=========================================")
	fmt.Println()

	start = time.Now()
	decryptor.Decrypt(ct_relu, plain_tmp)
	vals_tmp = encoder.Decode(plain_tmp, log_slots)
	fmt.Printf("Decryption (Relu) Done in %s \n", time.Since(start))
	fmt.Println("after relu: LV = ", ct_relu.Level(), " Scale = 2^", math.Log2(ct_relu.Scale))
	if printResult {
		fmt.Print("Result: \n")
		prt_mat_BL(vals_tmp, batch, 0)
	}
}

// Fast Conv without boot, Assume full batch with Po2 in_wid & N
// Normal Conv without output modification (e.g., trimming or expanding)
// Assume that the input is 0 padded according to kernel size: only in_wid - (ker_wid-1)/2 elements in row and columns are nonzero
func testConv_noBoot(logN, in_wid, ker_wid int, printResult bool) []float64 {
	N := (1 << logN)
	in_size := in_wid * in_wid
	batch := N / in_size
	ker_size := ker_wid * ker_wid

	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{ // Schemes parameters are created from scratch
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
	fmt.Println("========================================================")
	fmt.Println(" INSTANTIATING SCHEME & PLAINTEXT CREATION & Encryption ")
	fmt.Println("========================================================")
	fmt.Println()

	start = time.Now()
	kgen := ckks.NewKeyGenerator(params)
	sk := kgen.GenSecretKey()
	encryptor := ckks.NewEncryptor(params, sk)
	decryptor := ckks.NewDecryptor(params, sk)
	encoder := ckks.NewEncoder(params)
	plain_idx, pack_evaluator := gen_idxNlogs(params.MaxLevel(), kgen, sk, encoder, params) // for final pack_ciphertexts
	fmt.Printf("Keygen, Done in %s \n", time.Since(start))

	fmt.Println()
	fmt.Printf("CKKS parameters: logN = %d, logSlots = %d, logQP = %d, levels = %d, scale= %f, sigma = %f \n",
		params.LogN(), params.LogSlots(), params.LogQP(), params.MaxLevel()+1, params.Scale(), params.Sigma())

	input := make([]float64, N)
	pad := (ker_wid - 1) / 2
	k := 0.0
	for i := 0; i < in_wid; i++ {
		for j := 0; j < in_wid; j++ {
			for b := 0; b < batch; b++ {
				if (i < in_wid-pad) && (j < in_wid-pad) {
					input[i*in_wid*batch+j*batch+b] = k
					k += 1.0
				}
			}
		}
	}
	// for i := range input {
	// 	input[i] = 1.0 * float64(i) //0.1 * float64(i) / float64(N)
	// }
	ker_in := make([]float64, batch*batch*ker_size)
	for i := range ker_in {
		ker_in[i] = 1.0 * float64(i) //0.1 * float64(i) / float64(batch*batch*ker_size)
	}
	bn_a := make([]float64, batch)
	for i := range bn_a {
		bn_a[i] = 1.0
	}
	start = time.Now()
	pl_ker := prepKer_in(params, encoder, ker_in, bn_a, in_wid, ker_wid, batch, batch, batch, batch, params.MaxLevel())
	fmt.Printf("Plaintext (kernel) preparation, Done in %s \n", time.Since(start))

	fmt.Println("vec size: ", N)
	fmt.Println("input width: ", in_wid)
	fmt.Println("kernel width: ", ker_wid)
	fmt.Println("num batches: ", batch)
	fmt.Println("Input matrix: ")
	prt_vec(input)
	fmt.Println("Ker1_in (1st part): ")
	prt_vec(ker_in)

	start = time.Now()
	plain_tmp := ckks.NewPlaintext(params, params.MaxLevel(), params.Scale()) // contain plaintext values
	encoder.EncodeCoeffs(input, plain_tmp)
	ctxt_input := encryptor.EncryptNew(plain_tmp)
	fmt.Printf("Encryption done in %s \n", time.Since(start))

	fmt.Println()
	fmt.Println("===============================================")
	fmt.Println("     			   EVALUATION					")
	fmt.Println("===============================================")
	fmt.Println()

	start = time.Now()
	ct_result := conv_then_pack(params, pack_evaluator, ctxt_input, pl_ker, plain_idx, batch)
	fmt.Printf("Done in %s \n", time.Since(start))

	fmt.Println()
	fmt.Println("=========================================")
	fmt.Println("              DECRYPTION                 ")
	fmt.Println("=========================================")
	fmt.Println()

	start = time.Now()
	decryptor.Decrypt(ct_result, plain_tmp)
	cfs_tmp := encoder.DecodeCoeffs(plain_tmp)
	// cfs_tmp := reshape_conv_out(encoder.DecodeCoeffs(plain_tmp), in_wid, batch)
	fmt.Printf("Decryption Done in %s \n", time.Since(start))

	if printResult {
		fmt.Print("Result: \n")
		prt_mat(cfs_tmp, batch, 0)
	}

	return cfs_tmp
}

// Eval Conv, BN, relu with Boot
// in_wid must be Po2,
func testConv_BNRelu(in_wid, ker_wid, pad int, printResult bool) {

	logN := 16
	alpha := 0.0 // 0.3 => leakyrelu
	pow := 4
	N := (1 << logN)
	in_size := in_wid * in_wid
	batch := N / in_size
	ker_size := ker_wid * ker_wid
	ECD_LV := 1

	var btp *ckks.Bootstrapper
	btpParams := ckks.DefaultBootstrapParams[6]
	params, err := btpParams.Params()
	if err != nil {
		panic(err)
	}
	fmt.Printf("CKKS parameters: logN = %d, logSlots = %d, h = %d, logQP = %d, levels = %d, scale= 2^%f, sigma = %f \n",
		params.LogN(), params.LogSlots(), btpParams.H, params.LogQP(), params.QCount(), math.Log2(params.Scale()), params.Sigma())

	// Generate ext_idx for extracting valid values from conv with "same" padding
	ext_idx := make([][]int, 2)
	for i := 0; i < 2; i++ {
		ext_idx[i] = gen_keep_vec(logN, in_wid, pad, i)
	}

	// Scheme context and keys for evaluation (no Boot)
	kgen := ckks.NewKeyGenerator(params)
	sk, _ := kgen.GenKeyPairSparse(btpParams.H)
	rlk := kgen.GenRelinearizationKey(sk, 2)
	encoder := ckks.NewEncoder(params)
	decryptor := ckks.NewDecryptor(params, sk)
	encryptor := ckks.NewEncryptor(params, sk)
	evaluator := ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk})

	pl_idx, pack_evaluator := gen_idxNlogs(ECD_LV, kgen, sk, encoder, params)

	fmt.Println("Generating bootstrapping keys...")
	start = time.Now()
	rotations := btpParams.RotationsForBootstrapping(params.LogSlots())
	rotkeys := kgen.GenRotationKeysForRotations(rotations, true, sk)
	btpKey := ckks.BootstrappingKey{Rlk: rlk, Rtks: rotkeys}
	if btp, err = ckks.NewBootstrapper_mod(params, btpParams, btpKey); err != nil {
		panic(err)
	}
	fmt.Printf("Done in %s \n", time.Since(start))

	input := make([]float64, N)
	input_pad := (ker_wid - 1) / 2
	k := 0.0
	for i := 0; i < in_wid; i++ {
		for j := 0; j < in_wid; j++ {
			for b := 0; b < batch; b++ {
				if (i < in_wid-input_pad) && (j < in_wid-input_pad) {
					input[i*in_wid*batch+j*batch+b] = k
					k += (1.0 / float64(batch*(in_wid-input_pad)*(in_wid-input_pad)))
				}
			}
		}
	}
	// input := make([]float64, N)
	// for i := range input {
	// 	input[i] = 1.0 * float64(i) / float64(N)
	// }
	ker_in := make([]float64, batch*batch*ker_size)
	for i := range ker_in {
		ker_in[i] = 1.0 * float64(i) / float64(batch*batch*ker_size)
	}

	start = time.Now()
	bn_a := make([]float64, batch)
	bn_b := make([]float64, batch)
	b_coeffs := make([]float64, N)
	for i := range bn_a {
		bn_a[i] = 1.0 / 10000 // * float64(i) / float64(batch)
		bn_b[i] = 0.0 * float64(i) / float64(batch)
	}
	for i := range bn_b {
		for j := 0; j < in_wid; j++ {
			for k := 0; k < in_wid; k++ {
				b_coeffs[i+(j+k*in_wid)*batch] = bn_b[i]
			}
		}
	}
	pl_ker := prepKer_in(params, encoder, ker_in, bn_a, in_wid, ker_wid, batch, batch, batch, batch, ECD_LV)
	fmt.Printf("Plaintext (kernel) preparation, Done in %s \n", time.Since(start))

	fmt.Println("vec size: ", N)
	fmt.Println("input width: ", in_wid)
	fmt.Println("kernel width: ", ker_wid)
	fmt.Println("num batches: ", batch)
	fmt.Println("Input matrix: ")
	prt_vec(input)
	fmt.Println("Ker1_in (1st part): ")
	prt_vec(ker_in)

	start = time.Now()
	pl_input := ckks.NewPlaintext(params, ECD_LV, params.Scale()) // contain plaintext values
	encoder.EncodeCoeffs(input, pl_input)
	ct_input := encryptor.EncryptNew(pl_input)
	fmt.Printf("Encryption done in %s \n", time.Since(start))

	fmt.Println()
	fmt.Println("===============================================")
	fmt.Println("     			   EVALUATION					")
	fmt.Println("===============================================")
	fmt.Println()

	start = time.Now()
	ct_conv := conv_then_pack(params, pack_evaluator, ct_input, pl_ker, pl_idx, batch)

	// for Batch Normalization (BN)
	pl_bn_b := ckks.NewPlaintext(params, ct_conv.Level(), ct_conv.Scale) // contain plaintext values
	encoder.EncodeCoeffs(b_coeffs, pl_bn_b)
	encoder.ToNTT(pl_bn_b)
	evaluator.Add(ct_conv, pl_bn_b, ct_conv)
	fmt.Printf("Conv (with BN) Done in %s \n", time.Since(start))

	pl_conv := ckks.NewPlaintext(params, ct_conv.Level(), params.Scale())
	decryptor.Decrypt(ct_conv, pl_conv)
	val_preB := encoder.DecodeCoeffs(pl_conv)

	fmt.Println("Boot in: ")
	fmt.Println("Precision of values vs. ciphertext")
	in_cfs := printDebugCfs(params, ct_conv, val_preB, decryptor, encoder)

	fmt.Println("Bootstrapping... Ours (until CtoS):")
	start = time.Now()
	ct_boots := make([]*ckks.Ciphertext, 2)
	ct_boots[0], ct_boots[1], _ = btp.BootstrappConv_CtoS(ct_conv)
	fmt.Printf("Done in %s \n", time.Since(start))
	fmt.Println("after Boot (CtoS): LV = ", ct_boots[0].Level(), " Scale = ", math.Log2(ct_boots[0].Scale))

	// Only for checking the correctness
	in_cfs1_preB := make([]float64, params.Slots())
	in_cfs2_preB := make([]float64, params.Slots())
	in_slots1 := make([]complex128, params.Slots()) // first part of ceffs
	in_slots2 := make([]complex128, params.Slots()) // second part of ceffs
	for i := range in_cfs1_preB {
		in_cfs1_preB[i] = in_cfs[reverseBits(uint32(i), params.LogSlots())] // first part of coeffs
		in_cfs2_preB[i] = in_cfs[reverseBits(uint32(i), params.LogSlots())+uint32(params.Slots())]
		in_slots1[i] = complex(in_cfs1_preB[i]/math.Pow(2, float64(pow)), 0)
		in_slots2[i] = complex(in_cfs2_preB[i]/math.Pow(2, float64(pow)), 0)
	}
	ext1_tmp := keep_vec_fl(in_cfs1_preB, in_wid, pad, 0)
	ext2_tmp := keep_vec_fl(in_cfs2_preB, in_wid, pad, 1)
	for i := range in_cfs1_preB {
		in_cfs1_preB[i] = ext1_tmp[reverseBits(uint32(i), params.LogSlots())]
		in_cfs2_preB[i] = ext2_tmp[reverseBits(uint32(i), params.LogSlots())]
	}
	in_cfs_pBoot := append(in_cfs1_preB, in_cfs2_preB...)                      // After rot(ext) and boot
	in_slots1 = printDebug(params, ct_boots[0], in_slots1, decryptor, encoder) // Compare before & after CtoS
	in_slots2 = printDebug(params, ct_boots[1], in_slots2, decryptor, encoder) // Compare before & after CtoS

	start = time.Now()
	for pos := 0; pos < 2; pos++ {
		ct_boots[pos] = evalReLU(params, evaluator, ct_boots[pos], alpha)
		evaluator.MulByPow2(ct_boots[pos], pow, ct_boots[pos])
	}
	fmt.Printf("ReLU Done in %s \n", time.Since(start))

	values_ReLU := make([]complex128, len(in_slots1))
	for i := range values_ReLU {
		values_ReLU[i] = complex(math.Max(0, real(in_slots1[i])*math.Pow(2, float64(pow))), 0)
	}
	printDebug(params, ct_boots[0], values_ReLU, decryptor, encoder)
	for i := range values_ReLU {
		values_ReLU[i] = complex(math.Max(0, real(in_slots2[i])*math.Pow(2, float64(pow))), 0)
	}
	printDebug(params, ct_boots[1], values_ReLU, decryptor, encoder)

	ct_keep := make([]*ckks.Ciphertext, 2) // for extend (rotation) of ctxt_in

	start = time.Now()
	for pos := 0; pos < 2; pos++ {
		ct_keep[pos] = keep_ctxt(params, evaluator, encoder, ct_boots[pos], ext_idx[pos])
	}
	ct_res := btp.BootstrappConv_StoC(ct_keep[0], ct_keep[1])
	evaluator.Rescale(ct_res, params.Scale(), ct_res)

	fmt.Printf("Boot (StoC) Done in %s \n", time.Since(start))

	fmt.Printf("Boot out: ")
	printDebugCfs(params, ct_res, in_cfs_pBoot, decryptor, encoder)

	start = time.Now()
	decryptor.Decrypt(ct_res, pl_conv)
	cfs_tmp := encoder.DecodeCoeffs(pl_conv)
	fmt.Printf("Decryption Done in %s \n", time.Since(start))

	if printResult {
		fmt.Println()
		fmt.Println("=========================================")
		fmt.Println("              DECRYPTION                 ")
		fmt.Println("=========================================")
		fmt.Println()

		fmt.Print("Result: \n")
		prt_mat(cfs_tmp, batch, 0)
	}
}

func testResNet() {
	// For ResNet, we use padding: i.e., in_wid**2 element is contained in (2*in_wid)**2 sized block
	// So ReLU, keep or rot, StoC done only on the 1st part of the CtoS ciphertexts
	logN := 16
	raw_in_wids := []int{32, 16, 8} // same as python
	real_batch := []int{16, 32, 64} // same as python
	ker_wid := 3
	ECD_LV := 1
	padding := true
	in_wids := make([]int, len(raw_in_wids))
	for i, elt := range raw_in_wids {
		if padding {
			in_wids[i] = 2 * elt
		} else {
			in_wids[i] = elt
		}
	}
	cont := newContext(logN, ECD_LV, in_wids, padding)

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
				if (i < raw_in_wids[0]) && (j < raw_in_wids[0]) {
					input[i*in_wids[0]*max_batch[0]+j*max_batch[0]+b] = k
					k += (1.0 / float64(max_batch[0]*(raw_in_wids[0])*(raw_in_wids[0])))
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
		bn_a[i] = 0.02 // * float64(i) / float64(batch)
		bn_b[i] = 0.0  //0.1 * float64(i) // float64(real_batch[0])
	}
	bn_a2 := make([]float64, real_batch[1])
	bn_b2 := make([]float64, real_batch[1])
	for i := range bn_a2 {
		bn_a2[i] = 0.02 // * float64(i) / float64(batch)
		bn_b2[i] = 0.0  //0.1 * float64(i)
	}
	bn_a3 := make([]float64, real_batch[2])
	bn_b3 := make([]float64, real_batch[2])
	for i := range bn_a3 {
		bn_a3[i] = 0.02 // * float64(i) / float64(batch)
		bn_b3[i] = 0.0  //0.1 * float64(i)
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

	timings := make([]float64, 5)
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
		ct_layer[i] = evalConv_BNRelu(cont, ct_layer[i-1], ker_in, bn_a, bn_b, alpha, in_wids[0], ker_wid, max_batch[0], max_batch[0], 0, padding, false, prt_result)
		fmt.Println("Block1, Layer ", i, "done!")
	}
	fmt.Println("done.")
	timings[0] = time.Since(new_start).Seconds()
	new_start = time.Now()

	ct_result1 := evalConv_BNRelu(cont, ct_layer[num_blc1], ker_in12_0, bn_a, bn_b, alpha, in_wids[0], ker_wid, real_batch[0], real_batch[1]/2, 0, padding, true, prt_result)
	ct_result2 := evalConv_BNRelu(cont, ct_layer[num_blc1], ker_in12_1, bn_a, bn_b, alpha, in_wids[0], ker_wid, real_batch[0], real_batch[1]/2, 1, padding, true, prt_result)
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
		ct_layer2[i] = evalConv_BNRelu(cont, ct_layer2[i-1], ker_in2, bn_a2, bn_b2, alpha, in_wids[1], ker_wid, real_batch[1], real_batch[1], 0, padding, false, prt_result)
		fmt.Println("Block2, Layer ", i, "done!")
	}
	timings[2] = time.Since(new_start).Seconds()
	new_start = time.Now()

	ct_result = evalConv_BNRelu(cont, ct_layer2[num_blc2], ker_in23, bn_a3, bn_b3, alpha, in_wids[1], ker_wid, real_batch[1], real_batch[2], 0, padding, true, prt_result)
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
		ct_layer3[i] = evalConv_BNRelu(cont, ct_layer3[i-1], ker_in3, bn_a3, bn_b3, alpha, in_wids[2], ker_wid, real_batch[2], real_batch[2], 0, padding, false, prt_result)
		fmt.Println("Block3, Layer ", i, "done!")
	}
	timings[4] = time.Since(new_start).Seconds()
	new_start = time.Now()

	fmt.Println("Blc1: ", timings[0], " sec")
	fmt.Println("Blc1->2: ", timings[1], " sec")
	fmt.Println("Blc2: ", timings[2], " sec")
	fmt.Println("Blc2->3: ", timings[3], " sec")
	fmt.Println("Blc3: ", timings[4], " sec")
	fmt.Printf("Total done in %s \n", time.Since(begin_start))
}

// Eval Conv & Boot
func testBootFast_Conv(ext_input []int, logN, in_wid, ker_wid int, printResult bool) []float64 {

	N := (1 << logN)
	in_size := in_wid * in_wid
	batch := N / in_size
	ker_size := ker_wid * ker_wid
	ECD_LV := 3
	pos := 0

	fmt.Println()
	fmt.Println("=========================================")
	fmt.Println("    BOOTSTRAPP		then 	Rotation      ")
	fmt.Println("=========================================")
	fmt.Println()

	var btp *ckks.Bootstrapper

	// Bootstrapping parameters
	// LogSlots is hardcoded to 15 in the parameters, but can be changed from 1 to 15.
	// When changing logSlots make sure that the number of levels allocated to CtS and StC is
	// smaller or equal to logSlots.
	btpParams := ckks.DefaultBootstrapParams[6]
	params, err := btpParams.Params()
	if err != nil {
		panic(err)
	}
	fmt.Printf("CKKS parameters: logN = %d, logSlots = %d, h = %d, logQP = %d, levels = %d, scale= 2^%f, sigma = %f \n",
		params.LogN(), params.LogSlots(), btpParams.H, params.LogQP(), params.QCount(), math.Log2(params.Scale()), params.Sigma())

	// Generate rotations for EXT_FULL
	r_idx, m_idx := gen_extend_full(N/2, 2*in_wid, pos, true, true)
	var rotations []int
	for k := range r_idx {
		rotations = append(rotations, k)
	}
	for k := range m_idx {
		rotations = append(rotations, k)
	}
	// rotations = append(rotations, 1)
	fmt.Println("Rotations: ", rotations)

	// Scheme context and keys for evaluation (no Boot)
	kgen := ckks.NewKeyGenerator(params)
	sk, _ := kgen.GenKeyPairSparse(btpParams.H)
	rlk := kgen.GenRelinearizationKey(sk, 2)
	rotkeys := kgen.GenRotationKeysForRotations(rotations, false, sk)
	encoder := ckks.NewEncoder(params)
	decryptor := ckks.NewDecryptor(params, sk)
	encryptor := ckks.NewEncryptor(params, sk)
	evaluator := ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk, Rtks: rotkeys})

	plain_idx, pack_evaluator := gen_idxNlogs(ECD_LV, kgen, sk, encoder, params)

	input := make([]float64, N)
	for i := range input {
		input[i] = 1.0 * float64(ext_input[i]) / float64(N)
		// input[i] = 1.0 * float64(i) / float64(N)
	}

	int_tmp := make([]int, N)
	for i := range input {
		int_tmp[i] = int(float64(N) * input[i])
	}
	if printResult {
		fmt.Print("Input: \n")

		for b := 0; b < batch/4; b++ {
			print_vec("input ("+strconv.Itoa(b)+")", int_tmp, 2*in_wid, b)
		}
	}

	batch_real := batch / 16 // num batches at convolution 		// strided conv -> /(4*4)
	in_wid_out := in_wid * 4 // size of in_wid at convolution 	// strided conv -> *4

	ker1_in := make([]float64, batch_real*batch_real*ker_size)
	for i := range ker1_in {
		ker1_in[i] = 1.0 * (float64(len(ker1_in)) - float64(i) - 1) // float64(len(ker1_in))
	}

	ker1 := reshape_ker(ker1_in, ker_size, batch_real, false)

	pl_ker := make([]*ckks.Plaintext, batch_real)
	for i := 0; i < batch_real; i++ {
		pl_ker[i] = ckks.NewPlaintext(params, ECD_LV, params.Scale())
		encoder.EncodeCoeffs(encode_ker(ker1, pos, i, in_wid_out, batch_real, ker_wid, true), pl_ker[i])
		encoder.ToNTT(pl_ker[i])
	}

	fmt.Println("vec size: ", N)
	fmt.Println("input width: ", in_wid)
	fmt.Println("kernel width: ", ker_wid)
	fmt.Println("num batches (input): ", batch)
	fmt.Println("num batches (real): ", batch_real)
	fmt.Println("Input matrix: ")
	prt_vec(input)
	fmt.Println("Ker1_in (1st part): ")
	prt_vec(ker1[0])

	fmt.Println()
	fmt.Println("=========================================")
	fmt.Println("   PLAINTEXT CREATION & ENCRYPTION       ")
	fmt.Println("=========================================")
	fmt.Println()

	start = time.Now()

	cfs_tmp := make([]float64, N)                                             // contain coefficient msgs
	plain_tmp := ckks.NewPlaintext(params, params.MaxLevel(), params.Scale()) // contain plaintext values
	copy(cfs_tmp, input)

	encoder.EncodeCoeffs(cfs_tmp, plain_tmp)
	ctxt_input := encryptor.EncryptNew(plain_tmp)
	ctxt_out := make([]*ckks.Ciphertext, batch_real)

	fmt.Printf("Done in %s \n", time.Since(start))

	fmt.Print("Boot in: ")
	// Decrypt, print and compare with the plaintext values
	fmt.Println()
	fmt.Println("Precision of values vs. ciphertext")
	values_test := printDebugCfs(params, ctxt_input, cfs_tmp, decryptor, encoder)

	// plain_ttmp := ckks.NewPlaintext(params, ctxt_input.Level(), ctxt_input.Scale)
	// decryptor.Decrypt(ctxt_input, plain_ttmp)
	// tresult := encoder.Decode(plain_ttmp, params.LogSlots())
	// prt_res := make([]int, len(tresult))
	// fmt.Print(tresult)
	// for i := range tresult {
	// 	prt_res[i] = int(real(tresult[i]))
	// }
	// fmt.Println(prt_res)

	// ctxt_input = ext_ctxt(evaluator, encoder, ctxt_input, r_idx, m_idx, params)

	// fmt.Println("mult and rot")
	// decryptor.Decrypt(ctxt_input, plain_ttmp)
	// // decryptor.Decrypt(evaluator.RotateNew(ctxt_input, 1), plain_ttmp)
	// tresult = encoder.Decode(plain_ttmp, params.LogSlots())
	// for i := range tresult {
	// 	prt_res[i] = int(real(tresult[i]))
	// }
	// fmt.Println(prt_res)

	fmt.Println()
	fmt.Println("=========================================")
	fmt.Println("              BOOTSTRAPP                 ")
	fmt.Println("=========================================")
	fmt.Println()

	fmt.Println("Generating bootstrapping keys...")
	start = time.Now()
	rotations = btpParams.RotationsForBootstrapping(params.LogSlots())
	rotkeys = kgen.GenRotationKeysForRotations(rotations, true, sk)
	btpKey := ckks.BootstrappingKey{Rlk: rlk, Rtks: rotkeys}
	if btp, err = ckks.NewBootstrapper(params, btpParams, btpKey); err != nil {
		panic(err)
	}
	fmt.Printf("Done in %s \n", time.Since(start))

	fmt.Println("Bootstrapping... Ours:")
	start = time.Now()
	// Reason for multpling 1/(2*N) : for higher precision in SineEval & ReLU before StoC (needs to be recovered after/before StoC)
	// ciphertext1.SetScalingFactor(ciphertext1.Scale * float64(2*params.N()))

	ctxt1, ctxt2, _ := btp.BootstrappConv_CtoS(ctxt_input)
	fmt.Printf("Done in %s \n", time.Since(start))
	fmt.Println("after Boot: LV = ", ctxt1.Level(), " Scale = ", math.Log2(ctxt1.Scale))

	ctxt1 = ext_ctxt(evaluator, encoder, ctxt1, r_idx, params)
	ctxt2 = ext_ctxt(evaluator, encoder, ctxt2, r_idx, params)

	// evaluator.Rescale(ctxt1, params.Scale(), ctxt1)
	// evaluator.Rescale(ctxt2, params.Scale(), ctxt2)
	// // evaluator.Rotate(ctxt1, 4, ctxt1)
	// evaluator.Rotate(ctxt2, 4, ctxt2)
	// evaluator.DropLevel(ctxt1, 2)
	// evaluator.DropLevel(ctxt2, 2)

	ciphertext := btp.BootstrappConv_StoC(ctxt1, ctxt2)

	// evaluator.ScaleUp(ciphertext, params.Scale(), ciphertext)
	// evaluator.Rescale(ciphertext, params.Scale(), ciphertext)

	// fmt.Println("now level: ", ciphertext.Level())
	// fmt.Println("now scale: ", math.Log2(ciphertext.Scale))

	fmt.Printf("Boot out: ")
	// values_test1 := reverseOrder(values_test[:params.Slots()], params.LogSlots())
	// values_test2 := reverseOrder(values_test[params.Slots():], params.LogSlots())
	// for i := range values_test1 {
	// 	values_test[i] = values_test1[i]
	// 	values_test[i+params.Slots()] = values_test2[i]
	// }

	values_tmp1 := make([]float64, params.Slots())
	values_tmp2 := make([]float64, params.Slots())
	for i := range values_tmp1 {
		values_tmp1[i] = values_test[reverseBits(uint32(i), params.LogSlots())]
		values_tmp2[i] = values_test[reverseBits(uint32(i), params.LogSlots())+uint32(params.Slots())]
	}

	values_tmp11 := extend_full_fl(values_tmp1, 2*in_wid, pos, true, true)
	values_tmp22 := extend_full_fl(values_tmp2, 2*in_wid, pos, true, true)
	for i := range values_tmp1 {
		values_tmp1[i] = values_tmp11[reverseBits(uint32(i), params.LogSlots())]
		values_tmp2[i] = values_tmp22[reverseBits(uint32(i), params.LogSlots())]
	}
	values_test = append(values_tmp1, values_tmp2...)

	printDebugCfs(params, ciphertext, values_test, decryptor, encoder)

	// fmt.Printf("Boot out2: ")
	// values_test2 := reverseOrder(values_test[params.Slots():], params.LogSlots())
	// for i := range values_testC {
	// 	values_testC[i] = complex(values_test2[i], 0)
	// }
	// printDebug(params, ciphertext3, values_testC, decryptor, encoder)

	/// Not Necessary!!! ///
	xi_tmp := make([]float64, N)
	xi_tmp[(ker_wid-1)*(in_wid_out+1)] = 1.0
	xi_plain := ckks.NewPlaintext(params, ECD_LV, 1.0)
	encoder.EncodeCoeffs(xi_tmp, xi_plain)
	encoder.ToNTT(xi_plain)

	evaluator.Mul(ciphertext, xi_plain, ciphertext)

	mov_test := false

	if mov_test {
		fmt.Println()
		fmt.Println("=========================================")
		fmt.Println("              DECRYPTION                 ")
		fmt.Println("=========================================")
		fmt.Println()

		start = time.Now()

		decryptor.Decrypt(ciphertext, plain_tmp)
		cfs_tmp = encoder.DecodeCoeffs(plain_tmp)
		// cfs_tmp = reshape_conv_out(encoder.DecodeCoeffs(plain_tmp), in_wid, ker_wid, batch)
		int_tmp := make([]int, len(cfs_tmp))

		for i := range cfs_tmp {
			int_tmp[i] = int(float64(N) * cfs_tmp[i])
		}

		if printResult {
			fmt.Print("Result: \n")
			// fmt.Println(int_tmp)
			for b := 0; b < batch_real; b++ {
				print_vec("input ("+strconv.Itoa(b)+")", int_tmp, in_wid_out, b)
			}

			for i := range values_test {
				int_tmp[i] = int(float64(N) * values_test[i])
			}
			for b := 0; b < batch_real; b++ {
				print_vec("cp_input ("+strconv.Itoa(b)+")", int_tmp, in_wid_out, b)
			}
		}

		fmt.Printf("Done in %s \n", time.Since(start))

	} else {
		fmt.Println()
		fmt.Println("===============================================")
		fmt.Println("     			   EVALUATION					")
		fmt.Println("===============================================")
		fmt.Println()

		start = time.Now()

		for i := 0; i < batch_real; i++ {
			ctxt_out[i] = pack_evaluator.MulNew(ciphertext, pl_ker[i])
		}

		ctxt_result := pack_ctxts(pack_evaluator, ctxt_out, batch_real, plain_idx, params)
		fmt.Println("Result Scale: ", math.Log2(ctxt_result.Scale))
		fmt.Println("Result LV: ", ctxt_result.Level())
		fmt.Printf("Done in %s \n", time.Since(start))

		fmt.Println()
		fmt.Println("=========================================")
		fmt.Println("              DECRYPTION                 ")
		fmt.Println("=========================================")
		fmt.Println()

		start = time.Now()

		decryptor.Decrypt(ctxt_result, plain_tmp)
		cfs_tmp = reshape_conv_out(encoder.DecodeCoeffs(plain_tmp), in_wid_out/2, batch_real)

		if printResult {
			fmt.Print("Result: \n")
			prt_mat(cfs_tmp, batch_real, 2*in_wid)
		}

		fmt.Printf("Done in %s \n", time.Since(start))
	}

	return cfs_tmp
}

// set input as in_wid * in_wid * batch, then zero padding other values.
func testBRrot(logN, in_wid int, print bool) {

	N := (1 << logN)
	in_size := in_wid * in_wid
	batch := N / in_size
	pos := 0
	padding := true

	sm_input := make([]int, in_size) // each will be packed to input vector
	input := make([]int, N)
	input_rev := make([]int, N/2) // for upper or lowerpart
	output := make([]int, N)

	// set input and desired output
	for b := 0; b < batch; b++ {
		for i := range sm_input {
			sm_input[i] = i + b*in_size
		}
		arrgvec(sm_input, input, b)
	}

	for b := 0; b < batch; b++ {
		print_vec("input ("+strconv.Itoa(b)+")", input, in_wid, b)
	}

	input_up := input[0 : N/2]
	input_lw := input[N/2 : N]
	for i, elt := range input_up {
		input_rev[reverseBits(uint32(i), logN-1)] = elt
	}
	output_rev := comprs_full_hf(input_rev, in_wid, pos, padding)
	for i, elt := range output_rev {
		output[reverseBits(uint32(i), logN-1)] = elt
	}
	for i, elt := range input_lw {
		input_rev[reverseBits(uint32(i), logN-1)] = elt
	}
	output_rev = comprs_full_hf(input_rev, in_wid, pos, padding)
	for i, elt := range output_rev {
		output[uint32(N/2)+reverseBits(uint32(i), logN-1)] = elt
		if padding {
			output[uint32(N/2)+reverseBits(uint32(i), logN-1)] = 0
		}
	}
	for b := 0; b < batch*4; b++ {
		print_vec("output ("+strconv.Itoa(b)+")", output, in_wid/2, b)
	}
	fmt.Println(output)
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

	circ_rows := readTxt("./variables/rows.txt")
	circ_mat := make([][]float64, 8)
	plain_mat := make([]*ckks.Plaintext, 8)
	for i := 0; i < len(circ_mat); i++ {
		circ_mat[i] = encode_circ(circ_rows[N/2*i:N/2*(i+1)], 16, N)
		plain_mat[i] = ckks.NewPlaintext(params, ECD_LV, params.Scale())
		encoder.EncodeCoeffs(circ_mat[i], plain_mat[i])
		encoder.ToNTT(plain_mat[i])
	}

	test_input := readTxt("./inputs/vec_input_" + name_iter + ".txt")
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

// func testPoly() {

// 	fmt.Println()
// 	fmt.Println("=========================================")
// 	fmt.Println("              Eval Poly                  ")
// 	fmt.Println("=========================================")
// 	fmt.Println()

// 	var plaintext *ckks.Plaintext

// 	// Using Bootstrapping parameters
// 	// btpParams := ckks.DefaultBootstrapParams[5]
// 	// params, err := btpParams.Params()
// 	// if err != nil {
// 	// 	panic(err)
// 	// }

// 	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
// 		LogN: 12,
// 		LogQ: []int{log_out_scale, log_in_scale, log_in_scale, log_in_scale,
// 			log_in_scale, log_in_scale, log_in_scale, log_in_scale,
// 			log_in_scale, log_in_scale, log_in_scale, log_in_scale,
// 			log_in_scale, log_in_scale, log_in_scale, log_in_scale,
// 			log_in_scale, log_in_scale, log_in_scale, log_in_scale,
// 		},
// 		LogP:     []int{60},
// 		Sigma:    rlwe.DefaultSigma,
// 		LogSlots: 12 - 1,
// 		Scale:    float64(1 << log_in_scale),
// 	})
// 	if err != nil {
// 		panic(err)
// 	}

// 	// for i := 1; i < params.QCount(); i++ {
// 	// 	fmt.Printf("%x \n", params.Q()[i])
// 	// }

// 	fmt.Println()
// 	fmt.Printf("CKKS parameters: logN = %d, logSlots = %d, logQP = %d, levels = %d, scale= 2^%f, sigma = %f \n",
// 		params.LogN(), params.LogSlots(), params.LogQP(), params.QCount(), math.Log2(params.Scale()), params.Sigma())

// 	fmt.Println()
// 	fmt.Println("Generating keys...")
// 	start = time.Now()

// 	// Scheme context and keys
// 	kgen := ckks.NewKeyGenerator(params)
// 	sk := kgen.GenSecretKey()
// 	// sk, _ := kgen.GenKeyPairSparse(btpParams.H)
// 	rlk := kgen.GenRelinearizationKey(sk, 2)
// 	encoder := ckks.NewEncoder(params)
// 	decryptor := ckks.NewDecryptor(params, sk)
// 	encryptor := ckks.NewEncryptor(params, sk)
// 	evaluator := ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk})

// 	fmt.Printf("Done in %s \n", time.Since(start))

// 	// // Gen Boot Keys
// 	// fmt.Println()
// 	// fmt.Println("Generating bootstrapping keys...")
// 	// start = time.Now()
// 	// rotations := btpParams.RotationsForBootstrapping(params.LogSlots())
// 	// rotkeys := kgen.GenRotationKeysForRotations(rotations, true, sk)
// 	// btpKey := ckks.BootstrappingKey{Rlk: rlk, Rtks: rotkeys}
// 	// var btp *ckks.Bootstrapper
// 	// if btp, err = ckks.NewBootstrapper(params, btpParams, btpKey); err != nil {
// 	// 	panic(err)
// 	// }
// 	// fmt.Printf("Done in %s \n", time.Since(start))

// 	// Generate a random plaintext
// 	valuesWant := make([]complex128, params.Slots())
// 	for i := range valuesWant {
// 		if i < 13 {
// 			valuesWant[i] = complex(1.0/(math.Pow(2, float64(i))), 0.0)
// 		} else {
// 			// valuesWant[i] = complex(utils.RandFloat64(-1, 1), 0.0)
// 			tmp := rand.NormFloat64() * 0.16
// 			// for tmp > 1 || tmp < -1 {
// 			// 	tmp = rand.NormFloat64()
// 			// }
// 			// fmt.Print(tmp)
// 			valuesWant[i] = complex(tmp, 0.0)
// 		}
// 	}

// 	fmt.Print("Input: ")
// 	plaintext = ckks.NewPlaintext(params, 14, params.Scale()) // contain plaintext values
// 	encoder.Encode(plaintext, valuesWant, params.LogSlots())

// 	// Encrypt
// 	ciphertext := encryptor.EncryptNew(plaintext)

// 	// Decrypt, print and compare with the plaintext values
// 	fmt.Println()
// 	fmt.Println("Precision of values vs. ciphertext")
// 	values_test := printDebug(params, ciphertext, valuesWant, decryptor, encoder)

// 	// fmt.Println()
// 	// fmt.Println("Bootstrapping... Original:")

// 	// start = time.Now()
// 	// ciphertext0.SetScalingFactor(ciphertext0.Scale * float64(256.0))
// 	ciphertext0 := ciphertext
// 	// ciphertext0 := btp.Bootstrapp(ciphertext)
// 	// ciphertext4.SetScalingFactor(ciphertext4.Scale / float64(256.0))
// 	evaluator.Rescale(ciphertext0, params.Scale(), ciphertext0)
// 	fmt.Printf("level %d, scale %f \n", ciphertext0.Level(), math.Log2(ciphertext0.Scale))
// 	fmt.Printf("Done in %s \n", time.Since(start))
// 	values_test = printDebug(params, ciphertext0, values_test, decryptor, encoder)

// 	// // alpha 12
// 	// coeffs_tmp := []complex128{0.0, 11.5523042357223, 0.0, -67.7794513440968, 0.0, 125.283740404562, 0.0, -69.0142908232934}
// 	// coeffsReLU := ckks.NewPoly(coeffs_tmp)

// 	// coeffs_tmp2 := []complex128{0.0, 9.65167636181626, 0.0, -61.6939174538469, 0.0, 155.170351652298, 0.0, -182.697582383214, 0.0, 112.910726525406, 0.0, -37.7752411770263, 0.0, 6.47503909732344, 0.0, -0.445613365723361}
// 	// coeffsReLU2 := ckks.NewPoly(coeffs_tmp2)

// 	// coeffs_tmp3 := []complex128{0.0, 5.25888355571745, 0.0, -33.7233593794284, 0.0, 164.983085013457, 0.0, -541.408891406992, 0.0, 1222.96207997963, 0.0, -1952.01910566479, 0.0, 2240.84021378300, 0.0, -1866.34916983170, 0.0,
// 	// 	1127.22117843121, 0.0, -488.070474638380, 0.0, 147.497846308920, 0.0, -29.5171048879526, 0.0, 3.51269520930994, 0.0, -0.188101836557879}
// 	// coeffsReLU3 := ckks.NewPoly(coeffs_tmp3)

// 	// // F & G from CKK
// 	// coeffs_tmpF := []complex128{0.0, 315.0 / 128, 0.0, -420.0 / 128, 0.0, 378.0 / 128, 0.0, -180.0 / 128, 0.0, 35.0 / 128}
// 	// coeffsReLUF := ckks.NewPoly(coeffs_tmpF)

// 	// coeffs_tmpG := []complex128{0.0, 5850.0 / 1024, 0.0, -34974.0 / 1024, 0.0, 97015.0 / 1024, 0.0, -113492.0 / 1024, 0.0, 46623.0 / 1024}
// 	// coeffsReLUG := ckks.NewPoly(coeffs_tmpG)

// 	coeffs_tmpFD := []complex128{0.0, 35.0 / 16, 0.0, -35.0 / 16, 0.0, 21.0 / 16, 0.0, -5.0 / 16}
// 	coeffsReLUFD := ckks.NewPoly(coeffs_tmpFD)

// 	coeffs_tmpFD2 := []complex128{0.0, 35.0 / 32, 0.0, -35.0 / 32, 0.0, 21.0 / 32, 0.0, -5.0 / 32}
// 	coeffsReLUFD2 := ckks.NewPoly(coeffs_tmpFD2)

// 	coeffs_tmpGD := []complex128{0.0, 4.4824, 0.0, -16.1915, 0.0, 25.0202, 0.0, -12.5611}
// 	coeffsReLUGD := ckks.NewPoly(coeffs_tmpGD)

// 	// coeffs_tmp := []complex128{0.0, 7.30445164958251, 0.0, -34.6825871108659, 0.0, 59.8596518298826, 0.0, -31.8755225906466}
// 	// coeffsReLU := ckks.NewPoly(coeffs_tmp)
// 	// coeffs_tmp2 := []complex128{0.0, 2.40085652217597, 0.0, -2.63125454261783, 0.0, 1.54912674773593, 0.0, -0.331172956504304}
// 	// coeffsReLU2 := ckks.NewPoly(coeffs_tmp2)

// 	fmt.Printf("Eval(Ours): ")

// 	start = time.Now()
// 	ciphertext1, _ := evaluator.EvaluatePoly(ciphertext0, coeffsReLUGD, params.Scale())
// 	ciphertext1, _ = evaluator.EvaluatePoly(ciphertext1, coeffsReLUGD, params.Scale())
// 	ciphertext1, _ = evaluator.EvaluatePoly(ciphertext1, coeffsReLUFD, params.Scale())
// 	ciphertext1, _ = evaluator.EvaluatePoly(ciphertext1, coeffsReLUFD2, params.Scale())

// 	ciphertext2 := evaluator.AddConstNew(ciphertext1, 0.5)
// 	ciphertext3 := ciphertext0.CopyNew()
// 	evaluator.Rescale(ciphertext3, ciphertext2.Scale, ciphertext3)
// 	evaluator.DropLevel(ciphertext3, ciphertext3.Level()-ciphertext2.Level())
// 	evaluator.Mul(ciphertext3, ciphertext2, ciphertext2)
// 	// ciphertext2.SetScalingFactor(ciphertext2.Scale * 2)

// 	fmt.Printf("Done in %s \n", time.Since(start))
// 	fmt.Print("Sign: ")
// 	values_test1 := make([]complex128, len(values_test))
// 	for i := range values_test {
// 		values_test1[i] = complex(real(values_test[i])/math.Abs(real(values_test[i])), 0)
// 	}
// 	printDebug(params, ciphertext1, values_test1, decryptor, encoder)

// 	fmt.Print("ReLU: ")
// 	values_test2 := make([]complex128, len(values_test))
// 	for i := range values_test {
// 		values_test2[i] = complex(math.Max(0, real(values_test[i])), 0)
// 	}
// 	printDebug(params, ciphertext2, values_test2, decryptor, encoder)

// 	// ciphertext0 = encryptor.EncryptNew(plaintext)
// 	fmt.Printf("Eval: ")
// 	start = time.Now()

// 	ciphertext2 = evalReLU(params, evaluator, ciphertext0, 1.0)

// 	// ciphertext1, _ = evaluator.EvaluatePoly(ciphertext0, coeffsReLU, params.Scale())
// 	// ciphertext1, _ = evaluator.EvaluatePoly(ciphertext1, coeffsReLU2, params.Scale())
// 	// ciphertext1, _ = evaluator.EvaluatePoly(ciphertext1, coeffsReLU3, params.Scale())

// 	// ciphertext2 = evaluator.AddConstNew(ciphertext1, 0.5)
// 	// ciphertext3 = ciphertext0.CopyNew()
// 	// fmt.Println("c3 scale: ", math.Log2(ciphertext3.Scale))

// 	// // Modify c3 scale so that the mult result after rescale has desired scale
// 	// // constPlain := ckks.NewPlaintext(params, ciphertext3.Level(), float64(params.Q()[ciphertext1.Level()])/(ciphertext3.Scale))
// 	// // valuesPlain := make([]float64, params.N())
// 	// // valuesPlain[0] = 1.0
// 	// // encoder.EncodeCoeffs(valuesPlain, constPlain)
// 	// // encoder.ToNTT(constPlain)
// 	// // evaluator.Mul(ciphertext3, constPlain, ciphertext3)

// 	// evaluator.DropLevel(ciphertext3, ciphertext3.Level()-ciphertext2.Level())
// 	// evaluator.Mul(ciphertext3, ciphertext2, ciphertext2)
// 	// // evaluator.Rescale(ciphertext2, ciphertext0.Scale, ciphertext2)

// 	fmt.Printf("Done in %s \n", time.Since(start))
// 	// fmt.Print("Sign: ")
// 	// printDebug(params, ciphertext1, values_test1, decryptor, encoder)

// 	fmt.Print("ReLU: ")
// 	printDebug(params, ciphertext2, values_test2, decryptor, encoder)
// }

// // Eval Conv, then Pack
// // The ciphertexts must be packed into full (without vacant position)
// // For multiple output ciphertexts
// func conv_then_pack_mult(params ckks.Parameters, pack_evaluator ckks.Evaluator, ctxt_in []*ckks.Ciphertext, pl_ker [][]*ckks.Plaintext, plain_idx []*ckks.Plaintext, cnum_in, cnum_out, batch_out int) *ckks.Ciphertext {

// 	start := time.Now()
// 	ctxt_out := make([]*ckks.Ciphertext, batch_out)

// 	for in := 0; in < cnum_in; in++ {

// 		for i := 0; i < batch_out; i++ {
// 			ctxt_out[i] = pack_evaluator.MulNew(ctxt_in[0], pl_ker[0][i])
// 			for pos := 1; pos < 4; pos++ {
// 				pack_evaluator.Add(ctxt_out[i], pack_evaluator.MulNew(ctxt_in[pos], pl_ker[pos][i]), ctxt_out[i])
// 				// ctxt_out[i] += pack_evaluator.MulNew(ctxt_in[pos], pl_ker[pos][i]) // also need to modify prepKer's Encode_ker
// 			}
// 		}

// 	}

// 	ctxt_result := pack_ctxts(pack_evaluator, ctxt_out, batch_out, plain_idx, params)
// 	fmt.Println("Result Scale: ", math.Log2(ctxt_result.Scale))
// 	fmt.Println("Result LV: ", ctxt_result.Level())
// 	fmt.Printf("Done in %s \n", time.Since(start))

// 	return ctxt_result
// }
