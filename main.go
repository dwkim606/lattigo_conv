package main

import (
	"fmt"
	"math"
	"time"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/rlwe"
)

var start time.Time
var err error

const log_c_scale = 30
const log_in_scale = 30
const log_out_scale = 30

func main() {
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

		circ_rows := readTxt("./varaibles/rows.txt")
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

		var test_result *ckks.Ciphertext
		for i := 0; i < len(enc_test_input); i++ {
			if i == 0 {
				test_result = evaluator.MulNew(enc_test_input[i], plain_mat[i])
			} else {
				evaluator.Add(test_result, evaluator.MulNew(enc_test_input[i], plain_mat[i]), test_result)
			}
		}

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

		pl_ker := prepKer_in(params, encoder, encryptor, ker1, a1, in_wid[0], ker_wid, max_batch[0], max_batch[1], batch[0], batch[1], ECD_LV)

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
			ext_ctxt1[pos] = ext_ctxt(evaluator, encoder, ctxt1, r_idx[pos], m_idx[pos], params)
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

		ctxt_result := conv_then_pack(params, pack_evaluator, ciphertext, pl_ker, plain_idx, max_batch[1])

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

		pl_ker = prepKer_in(params, encoder, encryptor, ker2, a2, in_wid[1], ker_wid, max_batch[1], max_batch[2], batch[1], batch[2], ECD_LV)

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
			ext_ctxt1[pos] = ext_ctxt(evaluator, encoder, ctxt1, r_idx1[pos], m_idx1[pos], params)
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

		ctxt_result = conv_then_pack(params, pack_evaluator, ciphertext, pl_ker, plain_idx, max_batch[2])

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
		pl_ker = prepKer_in(params, encoder, encryptor, ker3, a3, in_wid[2], ker_wid, max_batch[2], max_batch[3], batch[2], batch[3], ECD_LV)

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
			ext_ctxt1[pos] = ext_ctxt(evaluator, encoder, ctxt1, r_idx2[pos], m_idx2[pos], params)
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

		ctxt_result = conv_then_pack(params, pack_evaluator, ciphertext, pl_ker, plain_idx, max_batch[3])
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

	// // To see each matrix
	// cfs_tmp = encoder.DecodeCoeffs(plain_out)
	// int_tmpn := make([]int, N)
	// for i := range cfs_tmp {
	// 	int_tmpn[i] = int(cfs_tmp[i])
	// }
	// fmt.Print("Output: \n")
	// for b := 0; b < batch[2]; b++ {
	// 	print_vec("output ("+strconv.Itoa(b)+")", int_tmpn, in_wid[3], b)
	// }

	// // again boot to see the correctness
	// ctxt_result.SetScalingFactor(ctxt_result.Scale * 32)
	// ctxt_boot1, ctxt_boot2, _ := btp.BootstrappConv_CtoS(ctxt_result)

	// evaluator.DropLevel(ctxt_boot1, ctxt_boot1.Level()-2)
	// evaluator.DropLevel(ctxt_boot2, ctxt_boot2.Level()-2)

	// ctxt_boot := btp.BootstrappConv_StoC(ctxt_boot1, ctxt_boot2)
	// fmt.Println("After boot scale? LV?", math.Log2(ctxt_boot.Scale), ctxt_boot.Level())
	// evaluator.Rescale(ctxt_boot, params.Scale(), ctxt_boot)

	// ctxt_boot.SetScalingFactor(ctxt_boot.Scale / 32)

	// printDebugCfs(params, ctxt_boot, pre_boot, decryptor, encoder)

	// input := testBRrot(logN, in_wid)
	// testPoly()
	// testBoot()

	// testBootFast_Conv(input, logN, in_wid, ker_wid, print)

	// valuesTest := testBootFast(logN, in_wid, ker_wid, print)
	// valuesWant := testConv(logN, in_wid, ker_wid, print)
	// printDebugCfsPlain(valuesTest, valuesWant)
}

func printDebugCfs(params ckks.Parameters, ciphertext *ckks.Ciphertext, valuesWant []float64, decryptor ckks.Decryptor, encoder ckks.Encoder) (valuesTest []float64) {
	total_size := make([]int, 15)
	valuesTest = encoder.DecodeCoeffs(decryptor.DecryptNew(ciphertext))

	fmt.Println()
	fmt.Printf("Level: %d (logQ = %d)\n", ciphertext.Level(), params.LogQLvl(ciphertext.Level()))
	fmt.Printf("Scale: 2^%f\n", math.Log2(ciphertext.Scale))
	fmt.Printf("ValuesTest:")
	for i := range total_size {
		fmt.Printf("%6.10f, ", valuesTest[i])
	}
	fmt.Printf("... \n")
	fmt.Printf("ValuesWant:")
	for i := range total_size {
		fmt.Printf("%6.10f, ", valuesWant[i])
	}
	fmt.Printf("... \n")

	valuesWantC := make([]complex128, len(valuesWant))
	valuesTestC := make([]complex128, len(valuesTest))
	for i := range valuesWantC {
		valuesWantC[i] = complex(valuesWant[i], 0)
		valuesTestC[i] = complex(valuesTest[i], 0)
	}
	precStats := ckks.GetPrecisionStats(params, encoder, nil, valuesWantC[:params.Slots()], valuesTestC[:params.Slots()], params.LogSlots(), 0)

	fmt.Println(precStats.String())

	precStats = ckks.GetPrecisionStats(params, encoder, nil, valuesWantC[params.Slots():], valuesTestC[params.Slots():], params.LogSlots(), 0)

	fmt.Println(precStats.String())
	fmt.Println()

	return
}

func printDebugCfsPlain(valuesTest, valuesWant []float64) {
	total_size := make([]int, 15)

	fmt.Printf("ValuesTest:")
	for i := range total_size {
		fmt.Printf("%6.10f, ", valuesTest[i])
	}
	fmt.Printf("... \n")
	fmt.Printf("ValuesWant:")
	for i := range total_size {
		fmt.Printf("%6.10f, ", valuesWant[i])
	}
	fmt.Printf("... \n")

	valuesWantC := make([]complex128, len(valuesWant))
	valuesTestC := make([]complex128, len(valuesTest))
	for i := range valuesWantC {
		valuesWantC[i] = complex(valuesWant[i], 0)
		valuesTestC[i] = complex(valuesTest[i], 0)
	}
	precStats := ckks.GetPrecisionStatsPlain(valuesWantC, valuesTestC, len(valuesWantC), 0)
	fmt.Println(precStats.String())
	fmt.Println()
}

func printDebug(params ckks.Parameters, ciphertext *ckks.Ciphertext, valuesWant []complex128, decryptor ckks.Decryptor, encoder ckks.Encoder) (valuesTest []complex128) {
	total_size := make([]int, 15)
	valuesTest = encoder.Decode(decryptor.DecryptNew(ciphertext), params.LogSlots())

	fmt.Println()
	fmt.Printf("Level: %d (logQ = %d)\n", ciphertext.Level(), params.LogQLvl(ciphertext.Level()))
	fmt.Printf("Scale: 2^%f\n", math.Log2(ciphertext.Scale))
	fmt.Printf("ValuesTest:")
	for i := range total_size {
		fmt.Printf("%6.5f, ", real(valuesTest[i]))
	}
	fmt.Printf("... \n")
	fmt.Printf("ValuesWant:")
	for i := range total_size {
		fmt.Printf("%6.5f, ", real(valuesWant[i]))
	}
	fmt.Printf("... \n")

	precStats := ckks.GetPrecisionStats(params, encoder, nil, valuesWant, valuesTest, params.LogSlots(), 0)

	fmt.Println(precStats.String())
	fmt.Println()

	return
}

// print slice (back and forth prt_size elements)
// scaled by 2N
func prt_vecc(vec []complex128) {
	prt_size := 5
	total_size := len(vec)

	if total_size <= 2*prt_size {
		fmt.Print("    [")
		for i := 0; i < total_size; i++ {
			fmt.Printf("  %4.5f + %1.2f i, ", real(vec[i]), imag(vec[i]))
		}
		fmt.Print(" ]\n")
	} else {
		fmt.Print("    [")
		for i := 0; i < prt_size; i++ {
			fmt.Printf(" %4.5f + %1.2f i, ", real(vec[i]), imag(vec[i]))
		}
		fmt.Printf(" ...,")
		for i := total_size - prt_size; i < total_size; i++ {
			fmt.Printf(" %4.5f + %1.2f i, ", real(vec[i]), imag(vec[i]))
		}
		fmt.Print(" ]\n")
	}
	fmt.Println()
}

// print slice (back and forth prt_size elements)
func prt_vec(vec []float64) {
	prt_size := 5
	total_size := len(vec)

	if total_size <= 2*prt_size {
		fmt.Print("    [")
		for i := 0; i < total_size; i++ {
			fmt.Printf("  %4.4f, ", vec[i])
		}
		fmt.Print(" ]\n")
	} else {
		fmt.Print("    [")
		for i := 0; i < prt_size; i++ {
			fmt.Printf(" %4.4f, ", vec[i])
		}
		fmt.Printf(" ...,")
		for i := total_size - prt_size; i < total_size; i++ {
			fmt.Printf(" %4.4f", vec[i])
		}
		fmt.Print(" ]\n")
	}
	fmt.Println()
}

// vec = arrgvec with batch batches, each batch is sqr-sized
// print (i,j)-th position in [batches], only shows (show, show) entries show = 0 : print all
func prt_mat(vec []float64, batch, show int) {
	mat_size := len(vec) / batch
	j, k := 1, 1
	for i := 0; i < len(vec); i += batch {
		if (show == 0) || (((j == 1) || (j == show)) && ((k <= 3) || (k >= show-3))) {
			fmt.Printf("(%d, %d): ", j, k)
			prt_vec(vec[i : i+batch])
		}
		k++
		if k*k > mat_size {
			k = 1
			j++
		}
	}
}

// // all layers have relu layer
// func main() {

// 	print := true
// 	logN := 12
// 	in_wid := [4]int{4, 8, 16, 32}
// 	max_batch := [4]int{64, 16, 4, 1}
// 	batch := [4]int{16, 4, 2, 1}
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
// 		r_idx[pos], m_idx[pos] = gen_extend_full(N/2, in_wid[1], pos, true, true)
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
// 		r_idx1[pos], m_idx1[pos] = gen_extend_full(N/2, in_wid[2], pos, true, true)
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
// 		r_idx2[pos], m_idx2[pos] = gen_extend_full(N/2, in_wid[3], pos, true, true)
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

// 	input := readTxt("./inputs/input_0000.txt")
// 	ext_input := inputExt(input, logN, in_wid[0], false) // Takes arranged input (assume intermediate layers)  // print only outputs first (st_batch) batches

// 	// input := make([]float64, N)
// 	// for i := range input {
// 	// 	input[i] = 1.0 * float64(ext_input[i]) / float64(in_wid[0]*in_wid[0]*batch[0])
// 	// }

// 	start := time.Now()
// 	plain_in := ckks.NewPlaintext(params, 1, params.Scale()) // contain plaintext values
// 	encoder.EncodeCoeffs(ext_input, plain_in)
// 	ctxt_in := encryptor.EncryptNew(plain_in)
// 	fmt.Printf("Encryption: Done in %s \n", time.Since(start))

// 	if print {
// 		fmt.Println("vec size: ", N)
// 		fmt.Println("input width: ", in_wid)
// 		fmt.Println("kernel width: ", ker_wid)
// 		fmt.Println("num batches (in 1 ctxt with padding): ", max_batch[0])
// 	}

// 	ker1 := readTxt("./variables/conv1.txt")
// 	pl_ker := prepKer_in(params, encoder, encryptor, ker1, in_wid[0], ker_wid, max_batch[0], max_batch[1], batch[0], batch[1], ECD_LV)

// 	fmt.Print("Boot in: ")
// 	fmt.Println()
// 	fmt.Println("Precision of values vs. ciphertext")
// 	in_cfs := printDebugCfs(params, ctxt_in, ext_input, decryptor, encoder)

// 	fmt.Println("Bootstrapping... Ours (until CtoS):")
// 	start = time.Now()
// 	ctxt1, ctxt2, _ := btp.BootstrappConv_CtoS(ctxt_in)
// 	fmt.Printf("Done in %s \n", time.Since(start))
// 	fmt.Println("after Boot: LV = ", ctxt1.Level(), " Scale = ", math.Log2(ctxt1.Scale))

// 	// Only for checking the correctness
// 	in_cfs_1_pBoot := make([]float64, params.Slots())
// 	in_cfs_2_pBoot := make([]float64, params.Slots())
// 	in_slots := make([]complex128, params.Slots()) // first part of ceffs
// 	for i := range in_cfs_1_pBoot {
// 		in_cfs_1_pBoot[i] = in_cfs[reverseBits(uint32(i), params.LogSlots())] // first part of coeffs
// 		in_cfs_2_pBoot[i] = in_cfs[reverseBits(uint32(i), params.LogSlots())+uint32(params.Slots())]
// 		in_slots[i] = complex(in_cfs_1_pBoot[i], 0)
// 	}
// 	ext1_tmp := extend_full_fl(in_cfs_1_pBoot, in_wid[1], 3, true, true)
// 	ext2_tmp := extend_full_fl(in_cfs_2_pBoot, in_wid[1], 3, true, true)
// 	for i := range in_cfs_1_pBoot {
// 		in_cfs_1_pBoot[i] = ext1_tmp[reverseBits(uint32(i), params.LogSlots())]
// 		in_cfs_2_pBoot[i] = ext2_tmp[reverseBits(uint32(i), params.LogSlots())]
// 	}
// 	in_cfs_pBoot := append(in_cfs_1_pBoot, in_cfs_2_pBoot...) // After rot(ext) and boot

// 	in_slots = printDebug(params, ctxt1, in_slots, decryptor, encoder)

// 	start = time.Now()
// 	// evaluator.MultByConst(ctxt1, 1.000000001, ctxt1)
// 	// evaluator.DropLevel(ctxt1, 10)

// 	ctxt1 = evalReLU(params, evaluator, ctxt1, 1.0)
// 	fmt.Printf("ReLU Done in %s \n", time.Since(start))

// 	values_ReLU := make([]complex128, len(in_slots))
// 	for i := range values_ReLU {
// 		values_ReLU[i] = complex(math.Max(0, real(in_slots[i])), 0)
// 	}
// 	printDebug(params, ctxt1, values_ReLU, decryptor, encoder)

// 	ext_ctxt1 := make([]*ckks.Ciphertext, 4) // for extend (rotation) of ctxt_in
// 	// ext_ctxt2 := make([]*ckks.Ciphertext, 4)  // do not need if we use po2 inputs dims
// 	ciphertext := make([]*ckks.Ciphertext, 4) // after Bootstrapping

// 	evaluator.DropLevel(ctxt2, ctxt2.Level()-2)
// 	start = time.Now()
// 	for pos := 0; pos < 4; pos++ {
// 		ext_ctxt1[pos] = ext_ctxt(evaluator, encoder, ctxt1, r_idx[pos], m_idx[pos], params)
// 		// fmt.Println(ext_ctxt1[pos].Level(), ctxt2.Level(), ext_ctxt1[pos].Scale, ctxt2.Scale)
// 		// ext_ctxt2[pos] = ext_ctxt(evaluator, encoder, ctxt2, r_idx[pos], m_idx[pos], params)
// 		ciphertext[pos] = btp.BootstrappConv_StoC(ext_ctxt1[pos], ctxt2)
// 		evaluator.Rescale(ciphertext[pos], params.Scale(), ciphertext[pos])
// 	}
// 	fmt.Printf("Boot (StoC) Done in %s \n", time.Since(start))

// 	fmt.Printf("Boot out: ")
// 	for i := range in_cfs_pBoot {
// 		in_cfs_pBoot[i] = math.Max(0, in_cfs_pBoot[i])
// 	}
// 	printDebugCfs(params, ciphertext[3], in_cfs_pBoot, decryptor, encoder)

// 	ctxt_result := conv_then_pack(params, pack_evaluator, ciphertext, pl_ker, plain_idx, max_batch[1])

// 	fmt.Println()
// 	fmt.Println("=========================================")
// 	fmt.Println("              DECRYPTION                 ")
// 	fmt.Println("=========================================")
// 	fmt.Println()

// 	plain_out := ckks.NewPlaintext(params, ctxt_result.Level(), params.Scale())
// 	start = time.Now()
// 	decryptor.Decrypt(ctxt_result, plain_out)
// 	pre_boot := encoder.DecodeCoeffs(plain_out)
// 	cfs_tmp := reshape_conv_out(encoder.DecodeCoeffs(plain_out), in_wid[1], max_batch[1])

// 	if print {
// 		fmt.Print("Result: \n")
// 		prt_mat(cfs_tmp, max_batch[1], in_wid[1])
// 	}
// 	fmt.Printf("(Layer 1) Done in %s \n", time.Since(start))

// 	// // To see each matrix
// 	// cfs_tmp = encoder.DecodeCoeffs(plain_out)
// 	// int_tmpn := make([]int, N)
// 	// for i := range cfs_tmp {
// 	// 	int_tmpn[i] = int(cfs_tmp[i])
// 	// }
// 	// fmt.Print("Output: \n")
// 	// for b := 0; b < batch[1]; b++ {
// 	// 	print_vec("output ("+strconv.Itoa(b)+")", int_tmpn, in_wid[2], b)
// 	// }

// 	// // Layer 1 done

// 	fmt.Println()
// 	fmt.Println("=========================================")
// 	fmt.Println("              LAYER 2	                 ")
// 	fmt.Println("=========================================")
// 	fmt.Println()

// 	ker2 := readTxt("./variables/conv2.txt")
// 	pl_ker = prepKer_in(params, encoder, encryptor, ker2, in_wid[1], ker_wid, max_batch[1], max_batch[2], batch[1], batch[2], ECD_LV)

// 	// fmt.Print("Boot in: ")
// 	// fmt.Println()
// 	// fmt.Println("Precision of values vs. ciphertext")
// 	// in_cfs = printDebugCfs(params, ctxt_result, pre_boot, decryptor, encoder)
// 	in_cfs = pre_boot
// 	for i := range in_cfs {
// 		in_cfs[i] = in_cfs[i] / 64
// 	}
// 	ctxt_in.Copy(ctxt_result)
// 	ctxt_in.SetScalingFactor(ctxt_in.Scale * 64)

// 	fmt.Println("Bootstrapping... Ours (until CtoS):")
// 	start = time.Now()
// 	ctxt1, _, _ = btp.BootstrappConv_CtoS(ctxt_in)
// 	fmt.Printf("Done in %s \n", time.Since(start))

// 	// Only for checking the correctness
// 	for i := range in_cfs_1_pBoot {
// 		in_cfs_1_pBoot[i] = in_cfs[reverseBits(uint32(i), params.LogSlots())] // first part of coeffs
// 		in_cfs_2_pBoot[i] = 0                                                 // in_cfs[reverseBits(uint32(i), params.LogSlots())+uint32(params.Slots())]
// 		in_slots[i] = complex(in_cfs_1_pBoot[i], 0)
// 	}
// 	ext1_tmp = extend_full_fl(in_cfs_1_pBoot, in_wid[2], 3, true, true)
// 	ext2_tmp = extend_full_fl(in_cfs_2_pBoot, in_wid[2], 3, true, true)
// 	for i := range in_cfs_1_pBoot {
// 		in_cfs_1_pBoot[i] = ext1_tmp[reverseBits(uint32(i), params.LogSlots())]
// 		in_cfs_2_pBoot[i] = ext2_tmp[reverseBits(uint32(i), params.LogSlots())]
// 	}
// 	in_cfs_pBoot = append(in_cfs_1_pBoot, in_cfs_2_pBoot...) // After rot(ext) and boot
// 	in_slots = printDebug(params, ctxt1, in_slots, decryptor, encoder)

// 	start = time.Now()
// 	// evaluator.MultByConst(ctxt1, 1.000000001, ctxt1)
// 	// evaluator.DropLevel(ctxt1, 10)
// 	ctxt1 = evalReLU(params, evaluator, ctxt1, 1.0)
// 	fmt.Printf("ReLU Done in %s \n", time.Since(start))

// 	for i := range values_ReLU {
// 		values_ReLU[i] = complex(math.Max(0, real(in_slots[i])), 0)
// 	}
// 	printDebug(params, ctxt1, values_ReLU, decryptor, encoder)

// 	start = time.Now()
// 	for pos := 0; pos < 4; pos++ {
// 		ext_ctxt1[pos] = ext_ctxt(evaluator, encoder, ctxt1, r_idx1[pos], m_idx1[pos], params)
// 		// ext_ctxt2[pos] = ext_ctxt(evaluator, encoder, ctxt2, r_idx1[pos], m_idx1[pos], params)
// 		ciphertext[pos] = btp.BootstrappConv_StoC(ext_ctxt1[pos], ctxt2)
// 		evaluator.Rescale(ciphertext[pos], params.Scale(), ciphertext[pos])
// 	}
// 	fmt.Printf("Boot (StoC) Done in %s \n", time.Since(start))

// 	fmt.Printf("Boot out: ")
// 	for i := range in_cfs_pBoot {
// 		in_cfs_pBoot[i] = math.Max(0, in_cfs_pBoot[i])
// 	}
// 	printDebugCfs(params, ciphertext[3], in_cfs_pBoot, decryptor, encoder)

// 	ctxt_result = conv_then_pack(params, pack_evaluator, ciphertext, pl_ker, plain_idx, max_batch[2])
// 	// ctxt_result = conv_then_pack(params, pack_evaluator, ciphertext, pl_ker, plain_idx, batch[2])

// 	fmt.Println()
// 	fmt.Println("=========================================")
// 	fmt.Println("              DECRYPTION                 ")
// 	fmt.Println("=========================================")
// 	fmt.Println()

// 	start = time.Now()
// 	decryptor.Decrypt(ctxt_result, plain_out)
// 	pre_boot = encoder.DecodeCoeffs(plain_out)
// 	cfs_tmp = reshape_conv_out(encoder.DecodeCoeffs(plain_out), in_wid[2], max_batch[2])

// 	if print {
// 		fmt.Print("Result: \n")
// 		prt_mat(cfs_tmp, max_batch[2], in_wid[2])
// 	}
// 	fmt.Printf("(Layer 2) Done in %s \n", time.Since(start))

// 	fmt.Println()
// 	fmt.Println("=========================================")
// 	fmt.Println("              LAYER 3	                 ")
// 	fmt.Println("=========================================")
// 	fmt.Println()

// 	ker3 := readTxt("./variables/conv3.txt")
// 	pl_ker = prepKer_in(params, encoder, encryptor, ker3, in_wid[2], ker_wid, max_batch[2], max_batch[3], batch[2], batch[3], ECD_LV)

// 	// fmt.Print("Boot in: ")
// 	// fmt.Println()
// 	// fmt.Println("Precision of values vs. ciphertext")
// 	// in_cfs = printDebugCfs(params, ctxt_result, pre_boot, decryptor, encoder)
// 	in_cfs = pre_boot
// 	for i := range in_cfs {
// 		in_cfs[i] = in_cfs[i] / 16
// 	}
// 	ctxt_in.Copy(ctxt_result)
// 	ctxt_in.SetScalingFactor(ctxt_in.Scale * 16)

// 	fmt.Println("Bootstrapping... Ours (until CtoS):")
// 	start = time.Now()
// 	ctxt1, _, _ = btp.BootstrappConv_CtoS(ctxt_in)
// 	fmt.Printf("Done in %s \n", time.Since(start))

// 	// Only for checking the correctness
// 	for i := range in_cfs_1_pBoot {
// 		in_cfs_1_pBoot[i] = in_cfs[reverseBits(uint32(i), params.LogSlots())] // first part of coeffs
// 		in_cfs_2_pBoot[i] = 0                                                 // in_cfs[reverseBits(uint32(i), params.LogSlots())+uint32(params.Slots())]
// 		in_slots[i] = complex(in_cfs_1_pBoot[i], 0)
// 	}
// 	ext1_tmp = extend_full_fl(in_cfs_1_pBoot, in_wid[3], 3, true, true)
// 	ext2_tmp = extend_full_fl(in_cfs_2_pBoot, in_wid[3], 3, true, true)
// 	for i := range in_cfs_1_pBoot {
// 		in_cfs_1_pBoot[i] = ext1_tmp[reverseBits(uint32(i), params.LogSlots())]
// 		in_cfs_2_pBoot[i] = ext2_tmp[reverseBits(uint32(i), params.LogSlots())]
// 	}
// 	in_cfs_pBoot = append(in_cfs_1_pBoot, in_cfs_2_pBoot...) // After rot(ext) and boot
// 	in_slots = printDebug(params, ctxt1, in_slots, decryptor, encoder)

// 	start = time.Now()
// 	// evaluator.MultByConst(ctxt1, 1.000000001, ctxt1)
// 	// evaluator.DropLevel(ctxt1, 10)

// 	ctxt1 = evalReLU(params, evaluator, ctxt1, 1.0)
// 	fmt.Printf("ReLU Done in %s \n", time.Since(start))

// 	for i := range values_ReLU {
// 		values_ReLU[i] = complex(math.Max(0, real(in_slots[i])), 0)
// 	}
// 	printDebug(params, ctxt1, values_ReLU, decryptor, encoder)

// 	start = time.Now()
// 	for pos := 0; pos < 4; pos++ {
// 		ext_ctxt1[pos] = ext_ctxt(evaluator, encoder, ctxt1, r_idx2[pos], m_idx2[pos], params)
// 		// ext_ctxt2[pos] = ext_ctxt(evaluator, encoder, ctxt2, r_idx2[pos], m_idx2[pos], params)
// 		ciphertext[pos] = btp.BootstrappConv_StoC(ext_ctxt1[pos], ctxt2)
// 		evaluator.Rescale(ciphertext[pos], params.Scale(), ciphertext[pos])
// 	}
// 	fmt.Printf("Boot (StoC) Done in %s \n", time.Since(start))

// 	fmt.Printf("Boot out: ")
// 	for i := range in_cfs_pBoot {
// 		in_cfs_pBoot[i] = math.Max(0, in_cfs_pBoot[i])
// 	}
// 	printDebugCfs(params, ciphertext[3], in_cfs_pBoot, decryptor, encoder)

// 	ctxt_result = conv_then_pack(params, pack_evaluator, ciphertext, pl_ker, plain_idx, max_batch[3])
// 	// ctxt_result = conv_then_pack(params, pack_evaluator, ciphertext, pl_ker, plain_idx, batch[2])

// 	fmt.Println()
// 	fmt.Println("=========================================")
// 	fmt.Println("              DECRYPTION                 ")
// 	fmt.Println("=========================================")
// 	fmt.Println()

// 	start = time.Now()
// 	decryptor.Decrypt(ctxt_result, plain_out)
// 	pre_boot = encoder.DecodeCoeffs(plain_out)
// 	cfs_tmp = reshape_conv_out(encoder.DecodeCoeffs(plain_out), in_wid[3], max_batch[3])

// 	if print {
// 		fmt.Print("Result: \n")
// 		prt_mat(cfs_tmp, max_batch[3], in_wid[3])
// 	}
// 	fmt.Printf("(Layer 3) Done in %s \n", time.Since(start))

// 	// // To see each matrix
// 	// cfs_tmp = encoder.DecodeCoeffs(plain_out)
// 	// int_tmpn := make([]int, N)
// 	// for i := range cfs_tmp {
// 	// 	int_tmpn[i] = int(cfs_tmp[i])
// 	// }
// 	// fmt.Print("Output: \n")
// 	// for b := 0; b < batch[2]; b++ {
// 	// 	print_vec("output ("+strconv.Itoa(b)+")", int_tmpn, in_wid[3], b)
// 	// }

// 	// // again boot to see the correctness
// 	// ctxt_result.SetScalingFactor(ctxt_result.Scale * 32)
// 	// ctxt_boot1, ctxt_boot2, _ := btp.BootstrappConv_CtoS(ctxt_result)

// 	// evaluator.DropLevel(ctxt_boot1, ctxt_boot1.Level()-2)
// 	// evaluator.DropLevel(ctxt_boot2, ctxt_boot2.Level()-2)

// 	// ctxt_boot := btp.BootstrappConv_StoC(ctxt_boot1, ctxt_boot2)
// 	// fmt.Println("After boot scale? LV?", math.Log2(ctxt_boot.Scale), ctxt_boot.Level())
// 	// evaluator.Rescale(ctxt_boot, params.Scale(), ctxt_boot)

// 	// ctxt_boot.SetScalingFactor(ctxt_boot.Scale / 32)

// 	// printDebugCfs(params, ctxt_boot, pre_boot, decryptor, encoder)

// 	// input := testBRrot(logN, in_wid)
// 	// testPoly()
// 	// testBoot()

// 	// testBootFast_Conv(input, logN, in_wid, ker_wid, print)

// 	// valuesTest := testBootFast(logN, in_wid, ker_wid, print)
// 	// valuesWant := testConv(logN, in_wid, ker_wid, print)
// 	// printDebugCfsPlain(valuesTest, valuesWant)
// }
