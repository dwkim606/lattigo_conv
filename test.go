package main

import (
	"fmt"
	"math"
	"strconv"
	"time"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/rlwe"
	"github.com/ldsec/lattigo/v2/utils"
)

func testConv(logN, in_wid, ker_wid int, printResult bool) []float64 {

	N := (1 << logN)
	in_size := in_wid * in_wid
	batch := N / in_size
	ker_size := ker_wid * ker_wid

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

	// evaluator := ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk})
	plain_idx, pack_evaluator := gen_idxNlogs(params.MaxLevel(), kgen, sk, encoder, params)

	fmt.Printf("Done in %s \n", time.Since(start))

	fmt.Println()
	fmt.Printf("CKKS parameters: logN = %d, logSlots = %d, logQP = %d, levels = %d, scale= %f, sigma = %f \n",
		params.LogN(), params.LogSlots(), params.LogQP(), params.MaxLevel()+1, params.Scale(), params.Sigma())

	input := make([]float64, N)
	for i := range input {
		input[i] = 1.0 * float64(i) / float64(N)
	}

	ker1_in := make([]float64, batch*batch*ker_size)
	for i := range ker1_in {
		ker1_in[i] = 1.0 * float64(i) / float64(batch*batch*ker_size)
	}
	ker1 := make([][]float64, batch)
	reshape_ker(ker1_in, ker1, ker_size)

	pl_ker := make([]*ckks.Plaintext, batch)
	for i := 0; i < batch; i++ {
		pl_ker[i] = ckks.NewPlaintext(params, params.MaxLevel(), params.Scale())
		encoder.EncodeCoeffs(encode_ker(ker1, i, in_wid, batch, ker_wid), pl_ker[i])
		encoder.ToNTT(pl_ker[i])
	}

	fmt.Println("vec size: ", N)
	fmt.Println("input width: ", in_wid)
	fmt.Println("kernel width: ", ker_wid)
	fmt.Println("num batches: ", batch)
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
	ctxt_out := make([]*ckks.Ciphertext, batch)

	fmt.Printf("Done in %s \n", time.Since(start))

	fmt.Println()
	fmt.Println("===============================================")
	fmt.Println("     			   EVALUATION					")
	fmt.Println("===============================================")
	fmt.Println()

	start = time.Now()

	for i := 0; i < batch; i++ {
		ctxt_out[i] = pack_evaluator.MulNew(ctxt_input, pl_ker[i])
	}

	result := pack_ctxts(pack_evaluator, ctxt_out, batch, plain_idx, params)

	fmt.Printf("Done in %s \n", time.Since(start))

	fmt.Println()
	fmt.Println("=========================================")
	fmt.Println("              DECRYPTION                 ")
	fmt.Println("=========================================")
	fmt.Println()

	start = time.Now()

	decryptor.Decrypt(result, plain_tmp)
	cfs_tmp = reshape_conv_out(encoder.DecodeCoeffs(plain_tmp), in_wid, ker_wid, batch)

	if printResult {
		fmt.Print("Result: \n")
		prt_mat(cfs_tmp, batch, 0)
	}

	fmt.Printf("Done in %s \n", time.Since(start))

	return cfs_tmp
}

func testPoly() {

	fmt.Println()
	fmt.Println("=========================================")
	fmt.Println("              Eval Poly                  ")
	fmt.Println("=========================================")
	fmt.Println()

	var plaintext *ckks.Plaintext

	// Using Bootstrapping parameters
	btpParams := ckks.DefaultBootstrapParams[5]
	params, err := btpParams.Params()
	if err != nil {
		panic(err)
	}

	// params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
	// 	LogN: logN,
	// 	LogQ: []int{log_out_scale, log_in_scale,
	// 		log_in_scale, log_in_scale, log_in_scale, log_in_scale,
	// 		log_in_scale, log_in_scale, log_in_scale, log_in_scale,
	// 		log_in_scale, log_in_scale, log_in_scale, log_in_scale,
	// 		log_in_scale, log_in_scale, log_in_scale, log_in_scale,
	// 		log_in_scale, log_in_scale, log_in_scale, log_in_scale,
	// 		log_in_scale, log_in_scale, log_in_scale, log_in_scale,
	// 		log_in_scale, log_in_scale, log_in_scale, log_in_scale,
	// 	},
	// 	LogP:     []int{60},
	// 	Sigma:    rlwe.DefaultSigma,
	// 	LogSlots: logN - 1,
	// 	Scale:    float64(1 << log_in_scale),
	// })
	// if err != nil {
	// 	panic(err)
	// }

	// for i := 1; i < params.QCount(); i++ {
	// 	fmt.Printf("%x \n", params.Q()[i])
	// }

	fmt.Println()
	fmt.Printf("CKKS parameters: logN = %d, logSlots = %d, logQP = %d, levels = %d, scale= 2^%f, sigma = %f \n",
		params.LogN(), params.LogSlots(), params.LogQP(), params.QCount(), math.Log2(params.Scale()), params.Sigma())

	fmt.Println()
	fmt.Println("Generating keys...")
	start = time.Now()

	// Scheme context and keys
	kgen := ckks.NewKeyGenerator(params)
	sk, _ := kgen.GenKeyPairSparse(btpParams.H)
	rlk := kgen.GenRelinearizationKey(sk, 2)
	encoder := ckks.NewEncoder(params)
	decryptor := ckks.NewDecryptor(params, sk)
	encryptor := ckks.NewEncryptor(params, sk)
	evaluator := ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk})

	fmt.Printf("Done in %s \n", time.Since(start))

	// Gen Boot Keys
	fmt.Println()
	fmt.Println("Generating bootstrapping keys...")
	start = time.Now()
	rotations := btpParams.RotationsForBootstrapping(params.LogSlots())
	rotkeys := kgen.GenRotationKeysForRotations(rotations, true, sk)
	btpKey := ckks.BootstrappingKey{Rlk: rlk, Rtks: rotkeys}
	var btp *ckks.Bootstrapper
	if btp, err = ckks.NewBootstrapper(params, btpParams, btpKey); err != nil {
		panic(err)
	}
	fmt.Printf("Done in %s \n", time.Since(start))

	// Generate a random plaintext
	valuesWant := make([]complex128, params.Slots())
	for i := range valuesWant {
		if i < 13 {
			valuesWant[i] = complex(1.0/(math.Pow(2, float64(i))), 0.0)
		} else {
			valuesWant[i] = complex(utils.RandFloat64(-1, 1), 0.0)
		}
	}

	fmt.Print("Input: ")
	plaintext = ckks.NewPlaintext(params, 9, params.Scale()) // contain plaintext values
	encoder.Encode(plaintext, valuesWant, params.LogSlots())

	// Encrypt
	ciphertext := encryptor.EncryptNew(plaintext)

	// Decrypt, print and compare with the plaintext values
	fmt.Println()
	fmt.Println("Precision of values vs. ciphertext")
	values_test := printDebug(params, ciphertext, valuesWant, decryptor, encoder)

	fmt.Println()
	fmt.Println("Bootstrapping... Original:")

	start = time.Now()
	// ciphertext0.SetScalingFactor(ciphertext0.Scale * float64(256.0))
	ciphertext0 := btp.Bootstrapp(ciphertext)
	// ciphertext4.SetScalingFactor(ciphertext4.Scale / float64(256.0))
	evaluator.Rescale(ciphertext0, params.Scale(), ciphertext0)
	fmt.Printf("level %d, scale %f \n", ciphertext0.Level(), math.Log2(ciphertext0.Scale))
	fmt.Printf("Done in %s \n", time.Since(start))
	values_test = printDebug(params, ciphertext0, values_test, decryptor, encoder)

	// alpha 10
	coeffs_tmp := []complex128{0.0, 10.8541842577442, 0.0, -62.2833925211098, 0.0, 114.369227820443, 0.0, -62.8023496973074}
	coeffsReLU := ckks.NewPoly(coeffs_tmp)

	coeffs_tmp2 := []complex128{0.0, 4.13976170985111, 0.0, -5.84997640211679, 0.0, 2.94376255659280, 0.0, -0.454530437460152}
	coeffsReLU2 := ckks.NewPoly(coeffs_tmp2)

	coeffs_tmp3 := []complex128{0.0, 3.29956739043733, 0.0, -7.84227260291355, 0.0, 12.8907764115564, 0.0, -12.4917112584486, 0.0, 6.94167991428074, 0.0, -2.04298067399942, 0.0, 0.246407138926031}
	coeffsReLU3 := ckks.NewPoly(coeffs_tmp3)

	// // alpha 12
	// coeffs_tmp := []complex128{0.0, 11.5523042357223, 0.0, -67.7794513440968, 0.0, 125.283740404562, 0.0, -69.0142908232934}
	// coeffsReLU := ckks.NewPoly(coeffs_tmp)

	// coeffs_tmp2 := []complex128{0.0, 9.65167636181626, 0.0, -61.6939174538469, 0.0, 155.170351652298, 0.0, -182.697582383214, 0.0, 112.910726525406, 0.0, -37.7752411770263, 0.0, 6.47503909732344, 0.0, -0.445613365723361}
	// coeffsReLU2 := ckks.NewPoly(coeffs_tmp2)

	// coeffs_tmp3 := []complex128{0.0, 5.25888355571745, 0.0, -33.7233593794284, 0.0, 164.983085013457, 0.0, -541.408891406992, 0.0, 1222.96207997963, 0.0, -1952.01910566479, 0.0, 2240.84021378300, 0.0, -1866.34916983170, 0.0,
	// 	1127.22117843121, 0.0, -488.070474638380, 0.0, 147.497846308920, 0.0, -29.5171048879526, 0.0, 3.51269520930994, 0.0, -0.188101836557879}
	// coeffsReLU3 := ckks.NewPoly(coeffs_tmp3)

	coeffs_tmpF := []complex128{0.0, 315.0 / 128, 0.0, -420.0 / 128, 0.0, 378.0 / 128, 0.0, -180.0 / 128, 0.0, 35.0 / 128}
	coeffsReLUF := ckks.NewPoly(coeffs_tmpF)
	_ = coeffsReLUF

	coeffs_tmpG := []complex128{0.0, 5850.0 / 1024, 0.0, -34974.0 / 1024, 0.0, 97015.0 / 1024, 0.0, -113492.0 / 1024, 0.0, 46623.0 / 1024}
	coeffsReLUG := ckks.NewPoly(coeffs_tmpG)
	_ = coeffsReLUG

	coeffs_tmpFD := []complex128{0.0, 35.0 / 16, 0.0, -35.0 / 16, 0.0, 21.0 / 16, 0.0, -5.0 / 16}
	coeffsReLUFD := ckks.NewPoly(coeffs_tmpFD)

	coeffs_tmpGD := []complex128{0.0, 4589.0 / 1024, 0.0, -16577.0 / 1024, 0.0, 25614.0 / 1024, 0.0, -12860.0 / 1024}
	coeffsReLUGD := ckks.NewPoly(coeffs_tmpGD)

	// coeffs_tmp := []complex128{0.0, 7.30445164958251, 0.0, -34.6825871108659, 0.0, 59.8596518298826, 0.0, -31.8755225906466}
	// coeffsReLU := ckks.NewPoly(coeffs_tmp)
	// coeffs_tmp2 := []complex128{0.0, 2.40085652217597, 0.0, -2.63125454261783, 0.0, 1.54912674773593, 0.0, -0.331172956504304}
	// coeffsReLU2 := ckks.NewPoly(coeffs_tmp2)

	fmt.Printf("Eval(Ours): ")

	start = time.Now()
	ciphertext1, _ := evaluator.EvaluatePoly(ciphertext0, coeffsReLUGD, params.Scale())
	ciphertext1, _ = evaluator.EvaluatePoly(ciphertext1, coeffsReLUGD, params.Scale())
	ciphertext1, _ = evaluator.EvaluatePoly(ciphertext1, coeffsReLUFD, params.Scale())
	ciphertext1, _ = evaluator.EvaluatePoly(ciphertext1, coeffsReLUFD, params.Scale())

	ciphertext2 := evaluator.AddConstNew(ciphertext1, 1.0)
	ciphertext3 := ciphertext0.CopyNew()

	evaluator.Rescale(ciphertext3, ciphertext2.Scale, ciphertext3)
	evaluator.DropLevel(ciphertext3, ciphertext3.Level()-ciphertext2.Level())
	evaluator.Mul(ciphertext3, ciphertext2, ciphertext2)
	ciphertext2.SetScalingFactor(ciphertext2.Scale * 2)

	fmt.Printf("Done in %s \n", time.Since(start))
	fmt.Print("Sign: ")
	values_test1 := make([]complex128, len(values_test))
	for i := range values_test {
		values_test1[i] = complex(real(values_test[i])/math.Abs(real(values_test[i])), 0)
	}
	printDebug(params, ciphertext1, values_test1, decryptor, encoder)

	fmt.Print("ReLU: ")
	values_test2 := make([]complex128, len(values_test))
	for i := range values_test {
		values_test2[i] = complex(math.Max(0, real(values_test[i])), 0)
		// values_test[i] = complex(real(values_test[i])/math.Abs(real(values_test[i])), 0)
	}
	printDebug(params, ciphertext2, values_test2, decryptor, encoder)

	// ciphertext0 = encryptor.EncryptNew(plaintext)
	fmt.Printf("Eval: ")
	start = time.Now()
	ciphertext1, _ = evaluator.EvaluatePoly(ciphertext0, coeffsReLU, params.Scale())
	ciphertext1, _ = evaluator.EvaluatePoly(ciphertext1, coeffsReLU2, params.Scale())
	ciphertext1, _ = evaluator.EvaluatePoly(ciphertext1, coeffsReLU3, params.Scale())

	ciphertext2 = evaluator.AddConstNew(ciphertext1, 1.0)
	ciphertext3 = ciphertext0.CopyNew()
	evaluator.Rescale(ciphertext3, ciphertext2.Scale, ciphertext3)
	evaluator.DropLevel(ciphertext3, ciphertext3.Level()-ciphertext2.Level())
	evaluator.Mul(ciphertext3, ciphertext2, ciphertext2)
	ciphertext2.SetScalingFactor(ciphertext2.Scale * 2)

	fmt.Printf("Done in %s \n", time.Since(start))
	fmt.Print("Sign: ")
	printDebug(params, ciphertext1, values_test1, decryptor, encoder)

	fmt.Print("ReLU: ")
	printDebug(params, ciphertext2, values_test2, decryptor, encoder)
}

func testBoot() {

	fmt.Println()
	fmt.Println("=========================================")
	fmt.Println("              BOOTSTRAPP                 ")
	fmt.Println("=========================================")
	fmt.Println()

	var btp *ckks.Bootstrapper
	var plaintext *ckks.Plaintext

	// Bootstrapping parameters
	// Four sets of parameters (index 0 to 3) ensuring 128 bit of security
	// are available in github.com/ldsec/lattigo/v2/ckks/bootstrap_params
	// LogSlots is hardcoded to 15 in the parameters, but can be changed from 1 to 15.
	// When changing logSlots make sure that the number of levels allocated to CtS and StC is
	// smaller or equal to logSlots.
	btpParams := ckks.DefaultBootstrapParams[5]
	params, err := btpParams.Params()
	if err != nil {
		panic(err)
	}

	fmt.Println()
	fmt.Printf("CKKS parameters: logN = %d, logSlots = %d, h = %d, logQP = %d, levels = %d, scale= 2^%f, sigma = %f \n",
		params.LogN(), params.LogSlots(), btpParams.H, params.LogQP(), params.QCount(), math.Log2(params.Scale()), params.Sigma())

	// Scheme context and keys
	kgen := ckks.NewKeyGenerator(params)
	sk, _ := kgen.GenKeyPairSparse(btpParams.H)
	rlk := kgen.GenRelinearizationKey(sk, 2)
	encoder := ckks.NewEncoder(params)
	decryptor := ckks.NewDecryptor(params, sk)
	encryptor := ckks.NewEncryptor(params, sk)
	// evaluator := ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk})

	fmt.Println()
	fmt.Println("Generating bootstrapping keys...")
	start = time.Now()
	rotations := btpParams.RotationsForBootstrapping(params.LogSlots())
	rotkeys := kgen.GenRotationKeysForRotations(rotations, true, sk)
	btpKey := ckks.BootstrappingKey{Rlk: rlk, Rtks: rotkeys}
	if btp, err = ckks.NewBootstrapper(params, btpParams, btpKey); err != nil {
		panic(err)
	}
	fmt.Printf("Done in %s \n", time.Since(start))

	// Generate a random plaintext
	// valuesWant := make([]complex128, params.Slots())
	// for i := range valuesWant {
	// 	valuesWant[i] = utils.RandComplex128(-1, 1)
	// }

	// plaintext = encoder.EncodeNew(valuesWant, params.LogSlots())
	cfs_tmp := make([]float64, params.N())
	for i := range cfs_tmp {
		// cfs_tmp[i] = math.Mod(math.Pow(2, float64(i)), float64(params.N()))
		cfs_tmp[i] = utils.RandFloat64(-1, 1)
		// cfs_tmp[i] = 1.0 * float64(i) / float64(len(cfs_tmp))
	}
	fmt.Print("Boot in: ")
	prt_vec(cfs_tmp)
	plaintext = ckks.NewPlaintext(params, params.MaxLevel(), params.Scale()) // contain plaintext values
	encoder.EncodeCoeffs(cfs_tmp, plaintext)

	// Encrypt
	ciphertext0 := encryptor.EncryptNew(plaintext)
	ciphertext1 := ciphertext0.CopyNew()

	// Decrypt, print and compare with the plaintext values
	fmt.Println()
	fmt.Println("Precision of values vs. ciphertext")
	values_test := printDebugCfs(params, ciphertext0, cfs_tmp, decryptor, encoder)

	// Bootstrap the ciphertext (homomorphic re-encryption)
	// It takes a ciphertext at level 0 (if not at level 0, then it will reduce it to level 0)
	// and returns a ciphertext at level MaxLevel - k, where k is the depth of the bootstrapping circuit.
	// CAUTION: the scale of the ciphertext MUST be equal (or very close) to params.Scale
	// To equalize the scale, the function evaluator.SetScale(ciphertext, parameters.Scale) can be used at the expense of one level.
	fmt.Println()
	fmt.Println("Bootstrapping... Original:")

	start = time.Now()
	// ciphertext0.SetScalingFactor(ciphertext0.Scale * float64(256.0))
	ciphertext4 := btp.Bootstrapp(ciphertext0)
	// ciphertext4.SetScalingFactor(ciphertext4.Scale / float64(256.0))
	fmt.Printf("Done in %s \n", time.Since(start))
	printDebugCfs(params, ciphertext4, values_test, decryptor, encoder)

	fmt.Println("Bootstrapping... Ours:")
	start = time.Now()
	// Reason for multpling 1/(2*N) : for higher precision in SineEval & ReLU before StoC (needs to be recovered after/before StoC)
	// ciphertext1.SetScalingFactor(ciphertext1.Scale * float64(2*params.N()))

	ciphertext2, ciphertext3 := btp.BootstrappConv_PreStoC(ciphertext1)
	fmt.Printf("Done in %s \n", time.Since(start))

	values_testC := make([]complex128, params.Slots())
	fmt.Printf("Boot out1: ")
	values_test1 := reverseOrder(values_test[:params.Slots()], params.LogSlots())
	for i := range values_testC {
		values_testC[i] = complex(values_test1[i], 0)
	}
	printDebug(params, ciphertext2, values_testC, decryptor, encoder)

	fmt.Printf("Boot out2: ")
	values_test2 := reverseOrder(values_test[params.Slots():], params.LogSlots())
	for i := range values_testC {
		values_testC[i] = complex(values_test2[i], 0)
	}
	printDebug(params, ciphertext3, values_testC, decryptor, encoder)

}

func testBRrot(logN, in_wid int, print bool) []int {

	N := (1 << logN)
	in_size := in_wid * in_wid
	batch := N / in_size

	if print {
		fmt.Print("Batch: ", batch, "\n\n")
	}

	sm_input := make([]int, in_size) // each will be packed to input vector
	input := make([]int, N)
	input_rev := make([]int, N)
	// input_hf_rev := make([]int, N/2)

	// out_dsr := make([]int, N)     // desired output
	// out_dsr_rev := make([]int, N) // desired output, bitrev
	// sm_out_final := make([]int, 4*4*in_size) // to print out the result (ext & ext_sp)
	test_out := make([]int, N)
	// test_out_hf := make([]int, N/2)

	// set input and desired output

	for b := 0; b < batch; b++ {
		for i := range sm_input {
			sm_input[i] = batch*i + b
		}
		arrgvec(sm_input, input, b)
	}

	// row := 4 * in_wid

	if print {
		for b := 0; b < 4; b++ {
			print_vec("input ("+strconv.Itoa(b)+")", input, in_wid, b)
		}
	}
	for i, elt := range input {
		input_rev[reverseBits(uint32(i), logN)] = elt
	}
	// fmt.Println("inputRev:")
	// for i := 0; i < len(input_rev); i += row {
	// 	fmt.Println(input_rev[i : i+row])
	// }
	// fmt.Println()

	pos := 0
	// test_out_rev := extend_vec(input_rev, in_wid, pos)
	test_out_rev := extend_sp(input_rev, in_wid, pos)
	// test_out_rev := extend_full(input_rev, in_wid, pos, false)

	// fmt.Println("extend sp testRev: pos(" + strconv.Itoa(pos) + ")")
	// for i := 0; i < len(test_out_rev); i += row {
	// 	fmt.Println(test_out_rev[i : i+row])
	// }
	// fmt.Println()

	for i, elt := range test_out_rev {
		test_out[reverseBits(uint32(i), logN)] = elt
	}
	if print {
		for b := 0; b < 1; b++ {
			print_vec("output ("+strconv.Itoa(b)+")", test_out, 2*in_wid, b)
		}
	}

	return test_out

	// for i, elt := range test_out[:N/2] {
	// 	input_hf_rev[reverseBits(uint32(i), logN-1)] = elt
	// }

	// test_out_hf = extend_full(input_hf_rev, 2*in_wid, 1, true, true)
	// // test_out_rev = extend_full(test_out_rev, 2*in_wid, 0, true, false)

	// fmt.Println("extend full testRev: pos(" + strconv.Itoa(pos) + ")")
	// for i := 0; i < len(test_out_hf); i += row {
	// 	fmt.Println(test_out_hf[i : i+row])
	// }
	// fmt.Println()

	// for i, elt := range test_out_hf {
	// 	test_out[reverseBits(uint32(i), logN-1)] = elt
	// }

	// for b := 0; b < batch/16; b++ {
	// 	print_vec_hf("output ("+strconv.Itoa(b)+")", test_out[:N/2], 4*in_wid, b)
	// }

	// for i, elt := range test_out_rev {
	// 	test_out[reverseBits(uint32(i), logN)] = elt
	// }
	// for b := 0; b < batch/16; b++ {
	// 	print_vec("output ("+strconv.Itoa(b)+")", test_out, 4*in_wid, b)
	// }

	// test_out_rev = comprs_full(test_out_rev, 4*in_wid, 15, false)

	// fmt.Println("after compressing")

	// fmt.Println("testRev: pos(" + strconv.Itoa(pos) + ")")
	// for i := 0; i < len(test_out_rev); i += row {
	// 	fmt.Println(test_out_rev[i : i+row])
	// }
	// fmt.Println()

	// for i, elt := range test_out_rev {
	// 	test_out[reverseBits(uint32(i), logN)] = elt
	// }
	// for b := 0; b < batch; b++ {
	// 	print_vec("output ("+strconv.Itoa(b)+")", test_out, in_wid, b)
	// }
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
	ker1 := make([][]float64, batch_real)
	reshape_ker(ker1_in, ker1, ker_size)

	pl_ker := make([]*ckks.Plaintext, batch_real)
	for i := 0; i < batch_real; i++ {
		pl_ker[i] = ckks.NewPlaintext(params, ECD_LV, params.Scale())
		encoder.EncodeCoeffs(encode_ker(ker1, i, in_wid_out, batch_real, ker_wid), pl_ker[i])
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

	ctxt1, ctxt2 := btp.BootstrappConv_PreStoC(ctxt_input)
	fmt.Printf("Done in %s \n", time.Since(start))
	fmt.Println("after Boot: LV = ", ctxt1.Level(), " Scale = ", math.Log2(ctxt1.Scale))

	ctxt1 = ext_ctxt(evaluator, encoder, ctxt1, r_idx, m_idx, params)
	ctxt2 = ext_ctxt(evaluator, encoder, ctxt2, r_idx, m_idx, params)

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
		cfs_tmp = reshape_conv_out(encoder.DecodeCoeffs(plain_tmp), in_wid_out, ker_wid, batch_real)

		if printResult {
			fmt.Print("Result: \n")
			prt_mat(cfs_tmp, batch_real, 2*in_wid)
		}

		fmt.Printf("Done in %s \n", time.Since(start))
	}

	return cfs_tmp
}

// Encode Kernel and outputs Plain(ker)
// in_wid : width of input (except padding)
// in_batch / out_batch: batches in 1 ctxt (input / output)
func prepKer(params ckks.Parameters, encoder ckks.Encoder, encryptor ckks.Encryptor, in_wid, ker_wid, in_batch, out_batch, ECD_LV int) []*ckks.Plaintext {
	ker_size := ker_wid * ker_wid
	in_batch_conv := in_batch / 16 // num batches at convolution 		// strided conv -> /(4*4)
	in_wid_conv := in_wid * 4      // size of in_wid at convolution 	// strided conv -> *4

	ker_in := make([]float64, in_batch*out_batch*ker_size)
	for i := range ker_in {
		ker_in[i] = 1.0 * (float64(len(ker_in)) - float64(i) - 1) // float64(len(ker1_in))
	}
	ker1 := make([][]float64, out_batch) // ker1[i][j] = j-th kernel for i-th output
	reshape_ker(ker_in, ker1, ker_size)

	pl_ker := make([]*ckks.Plaintext, out_batch)
	for i := 0; i < out_batch; i++ {
		pl_ker[i] = ckks.NewPlaintext(params, ECD_LV, params.Scale())
		encoder.EncodeCoeffs(encode_ker(ker1, i, in_wid_conv, in_batch_conv, ker_wid), pl_ker[i])
		encoder.ToNTT(pl_ker[i])
	}

	return pl_ker
}

// Eval Conv, then Pack
func conv_then_pack(params ckks.Parameters, pack_evaluator ckks.Evaluator, ctxt_in *ckks.Ciphertext, pl_ker, plain_idx []*ckks.Plaintext, batch_out int) *ckks.Ciphertext {

	start := time.Now()
	ctxt_out := make([]*ckks.Ciphertext, batch_out)
	for i := 0; i < batch_out; i++ {
		ctxt_out[i] = pack_evaluator.MulNew(ctxt_in, pl_ker[i])
	}

	ctxt_result := pack_ctxts(pack_evaluator, ctxt_out, batch_out, plain_idx, params)
	fmt.Println("Result Scale: ", math.Log2(ctxt_result.Scale))
	fmt.Println("Result LV: ", ctxt_result.Level())
	fmt.Printf("Done in %s \n", time.Since(start))

	return ctxt_result
}
