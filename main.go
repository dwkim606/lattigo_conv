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

var start time.Time
var err error

const log_c_scale = 30
const log_in_scale = 30
const log_out_scale = 45
const logN = 16

func main() {

	testBRrot()
	// testPoly()
	// testBoot()

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

func testConv() {
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
	plain_idx, pack_evaluator := gen_idxNlogs(kgen, sk, encoder, params)

	fmt.Printf("Done in %s \n", time.Since(start))

	fmt.Println()
	fmt.Printf("CKKS parameters: logN = %d, logSlots = %d, logQP = %d, levels = %d, scale= %f, sigma = %f \n",
		params.LogN(), params.LogSlots(), params.LogQP(), params.MaxLevel()+1, params.Scale(), params.Sigma())

	const N = (1 << logN)
	const in_wid = 8
	const in_size = in_wid * in_wid
	const batch = N / in_size
	const ker_wid = 5
	const ker_size = ker_wid * ker_wid

	input := make([]float64, N)
	for i := range input {
		input[i] = 1.0 * float64(i) / float64(N)
	}

	ker1_in := make([]float64, batch*batch*ker_size)
	for i := range ker1_in {
		ker1_in[i] = 1.0 * float64(i) / float64(batch*batch*ker_size)
	}
	ker1 := make([][]float64, batch)
	reshape_ker(ker1_in, ker1)

	pl_ker := make([]*ckks.Plaintext, batch)
	for i := 0; i < batch; i++ {
		pl_ker[i] = ckks.NewPlaintext(params, params.MaxLevel(), params.Scale())
		encoder.EncodeCoeffs(encode_ker(ker1, i, in_wid, ker_wid), pl_ker[i])
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

	fmt.Print("Result: \n")
	prt_mat(cfs_tmp, batch)

	fmt.Printf("Done in %s \n", time.Since(start))
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

	ciphertext2, ciphertext3 := btp.BootstrappConv(ciphertext1)
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

func testBRrot() {

	const logN = 10
	const N = (1 << logN)
	const in_wid = (1 << 2)
	const in_size = in_wid * in_wid
	const batch = N / in_size

	fmt.Print("Batch: ", batch, "\n\n")

	sm_input := make([]int, in_size) // each will be packed to input vector
	input := make([]int, N)
	input_rev := make([]int, N)

	// out_dsr := make([]int, N)     // desired output
	// out_dsr_rev := make([]int, N) // desired output, bitrev
	// sm_out_final := make([]int, 4*4*in_size) // to print out the result (ext & ext_sp)
	test_out := make([]int, N)

	// set input and desired output

	for b := 0; b < batch; b++ {
		for i := range sm_input {
			sm_input[i] = batch*i + b
		}
		arrgvec(sm_input, input, b)
	}

	row := 4 * in_wid

	for b := 0; b < batch; b++ {
		print_vec("input ("+strconv.Itoa(b)+")", input, in_wid, b)
	}
	for i, elt := range input {
		input_rev[reverseBits(uint32(i), logN)] = elt
	}
	fmt.Println("inputRev:")
	for i := 0; i < len(input_rev); i += row {
		fmt.Println(input_rev[i : i+row])
	}
	fmt.Println()

	pos := 0
	// test_out_rev := extend_vec(input_rev, in_wid, pos)
	test_out_rev := extend_sp(input_rev, in_wid, pos)
	// test_out_rev := extend_full(input_rev, in_wid, pos, false)

	fmt.Println("testRev: pos(" + strconv.Itoa(pos) + ")")
	for i := 0; i < len(test_out_rev); i += row {
		fmt.Println(test_out_rev[i : i+row])
	}
	fmt.Println()

	for i, elt := range test_out_rev {
		test_out[reverseBits(uint32(i), logN)] = elt
	}
	for b := 0; b < batch/4; b++ {
		print_vec("output ("+strconv.Itoa(b)+")", test_out, 2*in_wid, b)
	}

	test_out_rev = extend_full(test_out_rev, in_wid, 0, true)

	fmt.Println("testRev: pos(" + strconv.Itoa(pos) + ")")
	for i := 0; i < len(test_out_rev); i += row {
		fmt.Println(test_out_rev[i : i+row])
	}
	fmt.Println()

	for i, elt := range test_out_rev {
		test_out[reverseBits(uint32(i), logN)] = elt
	}
	for b := 0; b < batch/16; b++ {
		print_vec("output ("+strconv.Itoa(b)+")", test_out, 4*in_wid, b)
	}

	test_out_rev2 := comprs_full(test_out_rev, 4*in_wid, 0, true)

	fmt.Println("after compressing")

	fmt.Println("testRev: pos(" + strconv.Itoa(pos) + ")")
	for i := 0; i < len(test_out_rev2); i += row {
		fmt.Println(test_out_rev2[i : i+row])
	}
	fmt.Println()

	for i, elt := range test_out_rev2 {
		test_out[reverseBits(uint32(i), logN)] = elt
	}
	for b := 0; b < batch/4; b++ {
		print_vec("output ("+strconv.Itoa(b)+")", test_out, 2*in_wid, b)
	}

	// pos = 1
	// input_rev = comprs_vec(test_out_rev, 2*in_wid, pos)

	// for i, elt := range input_rev {
	// 	input[reverseBits(uint32(i), logN)] = elt
	// }
	// for b := 0; b < batch; b++ {
	// 	print_vec("input after compress ("+strconv.Itoa(b)+")", input, in_wid, b)
	// }

	// print_vec("test (1)", test_out_rev, 2*in_wid, 1)

	// // fmt.Println("test: ")
	// // for i := 0; i < len(test_out_rev); i += row {
	// // 	fmt.Println(test_out_rev[i : i+row])
	// // }

	// fmt.Println("outDesiredRev: ")
	// for i := 0; i < len(out_dsr_rev); i += row {
	// 	fmt.Println(out_dsr_rev[i : i+row])
	// }

	// fmt.Println("OutDesired: ")
	// for i := 0; i < len(out_dsr); i += row {
	// 	fmt.Println(out_dsr[i : i+row])
	// }

}

// distribute input to the output starting from pos position
func arrgvec(input []int, output []int, pos int) {
	batch := len(output) / len(input)

	for i, elt := range input {
		output[pos+i*batch] = elt
	}
}

// print out arrg vec input from pos position
func print_vec(title string, input []int, in_wid int, pos int) {
	row := make([]int, in_wid)
	step := len(input) / (in_wid * in_wid)
	fmt.Println(title, ": ")
	for j := 0; j < in_wid; j++ {
		for i := range row {
			row[i] = input[(j*in_wid+i)*step+pos]

		}
		fmt.Println(row)
	}
	fmt.Println()

}

func lRot(a []int, rotation int) []int {
	size := len(a)
	var newArray []int
	for i := 0; i < rotation; i++ {
		newArray = a[1:size]
		newArray = append(newArray, a[0])
		a = newArray
	}
	return a
}

func rRot(a []int, rotation int) []int {
	return lRot(a, len(a)-rotation)
}

func addSlice(a []int, b []int) []int {
	c := make([]int, len(a))
	for i := range a {
		c[i] = a[i] + b[i]

	}
	return c
}

// extend a vector (with in_wid * in_wid elt) inside a full-sized input vector to a strided vector (both are bitreversed)
// assume that the full vector is filled with sm vectors
// e.g., 1234 -> 1020 // 0000 // 3040 // 0000 (before bitreversed)
// 0 <= pos < 4 determines which part of input is extended to output
func extend_vec(input []int, in_wid int, pos int) []int {
	pos = int(reverseBits(uint32(pos), 2))
	output := make([]int, len(input))
	batch := len(input) / (in_wid * in_wid)
	min_batch := batch / 4
	if batch%4 != 0 {
		panic("batch size not divisible by 4")
	}

	for j := 0; j < in_wid; j++ { // kinds of mov depends on j
		tmp := make([]int, len(input))
		for b := 0; b < min_batch; b++ {
			for l := 0; l < in_wid; l++ {
				idx := 4*in_wid*in_wid*b + pos*in_wid*in_wid + in_wid*j + l
				tmp[idx] = input[idx]
			}
		}
		rot := j*in_wid - pos*in_wid*in_wid
		output = addSlice(output, rRot(tmp, rot))
	}

	return output
}

// embed a vector into larger space (both are bitreversed)
// assume that the full vector is filled with sm vectors
// e.g., 1234 -> 12 00 // 34 00 // 0000 // 0000 (before bitreversed)
// 0 <= pos < 4 determines which part of input is extended to output
func extend_sp(input []int, in_wid int, pos int) []int {
	mid_out := make([]int, len(input))
	output := make([]int, len(input))
	batch := len(input) / (in_wid * in_wid)
	pos = int(reverseBits(uint32(pos), 2))
	min_batch := batch / 4
	if batch%4 != 0 {
		panic("batch size not divisible by 4")
	}

	for j := 0; j < in_wid; j++ { // kinds of mov depends on j
		tmp := make([]int, len(input))
		for b := 0; b < min_batch; b++ {
			for i := 0; i < in_wid; i++ {
				idx := 4*in_wid*in_wid*b + pos*in_wid*in_wid + in_wid*j + i
				tmp[idx] = input[idx]
			}
		}
		rot := 3*j*in_wid - pos*in_wid*in_wid
		mid_out = addSlice(mid_out, rRot(tmp, rot))
	}

	for i := 0; i < in_wid; i++ { // kinds of mov depends on i
		tmp := make([]int, len(input))
		for b := 0; b < min_batch; b++ {
			for j := 0; j < in_wid; j++ {
				idx := 4*in_wid*in_wid*b + 4*in_wid*j + i
				tmp[idx] = mid_out[idx]
			}
		}
		rot := i
		output = addSlice(output, rRot(tmp, rot))
	}

	return output
}

// extend_vec then extend_sp (with in_wid * in_wid elt) inside a full-sized input vector to a strided vector (both are bitreversed)
// assume that the full vector is filled with sm vectors
// padding = true: sm vector is already inside the 4*len(sm_vector) size vector with zeros
// e.g., 12 00 // 34 00 // 00 00 // 00 00
// 0 <= pos < 4 determines which part of input is extended to output
// padding = false: then, 0 <= pos < 16
func extend_full(input []int, in_wid int, pos int, padding bool) []int {
	mid_out := make([]int, len(input))
	output := make([]int, len(input))

	if padding == false {
		batch := len(input) / (in_wid * in_wid)
		pos = int(reverseBits(uint32(pos), 4))
		min_batch := batch / 16
		if batch%16 != 0 {
			panic("batch size not divisible by 16")
		}

		for j := 0; j < in_wid; j++ { // kinds of mov depends on j
			tmp := make([]int, len(input))
			for b := 0; b < min_batch; b++ {
				for l := 0; l < in_wid; l++ {
					idx := 16*in_wid*in_wid*b + pos*in_wid*in_wid + in_wid*j + l
					tmp[idx] = input[idx]
				}
			}
			rot := 7*j*in_wid - pos*in_wid*in_wid
			mid_out = addSlice(mid_out, rRot(tmp, rot))
		}

		for i := 0; i < in_wid; i++ { // kinds of mov depends on i
			tmp := make([]int, len(input))
			for b := 0; b < min_batch; b++ {
				for j := 0; j < in_wid; j++ {
					idx := 16*in_wid*in_wid*b + 8*in_wid*j + i
					tmp[idx] = mid_out[idx]
				}
			}
			rot := i
			output = addSlice(output, rRot(tmp, rot))
		}
	} else {
		batch := len(input) / (2 * in_wid * 2 * in_wid)
		pos = int(reverseBits(uint32(pos), 2))
		min_batch := batch / 4
		if batch%4 != 0 {
			panic("batch size not divisible by 4")
		}

		for j := 0; j < in_wid; j++ { // kinds of mov depends on j
			tmp := make([]int, len(input))
			for b := 0; b < min_batch; b++ {
				for i := 0; i < in_wid; i++ {
					idx := 16*in_wid*in_wid*b + 4*in_wid*in_wid*pos + 4*in_wid*j + 2*i
					tmp[idx] = input[idx]
				}
			}
			rot := 4*in_wid*j - 4*in_wid*in_wid*pos
			output = addSlice(output, rRot(tmp, rot))
		}
	}

	return output
}

// reverse of extend_vec
// compress a strided  vector (with in_wid * in_wid elt) inside a full-sized input vector to a smaller vector (in_wid/2 * in_wid/2) (both are bitreversed)
// assume that the full vector is filled with sm vectors
// e.g., 1020 // 0000 // 3040 // 0000  -> 1234(before bitreversed)
// 0 <= pos < 4 determines to which part the output is positioned at the final output
func comprs_vec(input []int, in_wid int, pos int) []int {
	pos = int(reverseBits(uint32(pos), 2))
	output := make([]int, len(input))
	batch := len(input) / (in_wid * in_wid)
	min_wid := in_wid / 2
	if in_wid%2 != 0 {
		panic("input width not divisible by 2")
	}

	for j := 0; j < min_wid; j++ { // kinds of mov depends on j
		tmp := make([]int, len(input))
		for b := 0; b < batch; b++ {
			for i := 0; i < min_wid; i++ {
				idx := 4*min_wid*min_wid*b + in_wid*j + i
				tmp[idx] = input[idx]
			}
		}
		rot := -j*min_wid + pos*min_wid*min_wid
		output = addSlice(output, rRot(tmp, rot))
	}

	return output
}

// reverse of extend_full
// padding = true: result sm vector is inside the 4*len(sm_vector) size vector with zeros
// e.g., 12 00 // 34 00 // 00 00 // 00 00
// compress a strided vector (with in_wid * in_wid) inside a full-sized input vector to a smaller vector (in_wid/2 * in_wid/2) (both are bitreversed)
// assume that the full vector is filled with sm vectors
// 0 <= pos < 4 determines to which part the output is positioned at the final output
// padding = false: then, 0 <= pos < 16, and result sm vector has no zeros
func comprs_full(input []int, in_wid int, pos int, padding bool) []int {
	mid_out := make([]int, len(input))
	output := make([]int, len(input))

	if !padding {
		output = mid_out
	} else {
		pos = int(reverseBits(uint32(pos), 2))
		batch := len(input) / (in_wid * in_wid)
		min_wid := in_wid / 4
		if in_wid%4 != 0 {
			panic("input wid not divisible by 4")
		}

		for j := 0; j < min_wid; j++ { // kinds of mov depends on j
			tmp := make([]int, len(input))
			for b := 0; b < batch; b++ {
				for i := 0; i < min_wid; i++ {
					idx := 16*min_wid*min_wid*b + 8*min_wid*j + 2*i
					tmp[idx] = input[idx]
				}
			}
			rot := -4*j*min_wid + 4*pos*min_wid*min_wid
			output = addSlice(output, rRot(tmp, rot))
		}
	}

	return output
}
