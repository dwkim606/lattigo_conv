package main

import (
	"fmt"
	"math"
	"time"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/rlwe"
	"github.com/ldsec/lattigo/v2/utils"
)

func main() {
	var start time.Time
	var err error

	const log_c_scale = 30
	const log_in_scale = 30
	const log_out_scale = 20
	const logN = 13

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
	btpParams := ckks.DefaultBootstrapParams[2]
	params, err = btpParams.Params()
	if err != nil {
		panic(err)
	}

	fmt.Println()
	fmt.Printf("CKKS parameters: logN = %d, logSlots = %d, h = %d, logQP = %d, levels = %d, scale= 2^%f, sigma = %f \n",
		params.LogN(), params.LogSlots(), btpParams.H, params.LogQP(), params.QCount(), math.Log2(params.Scale()), params.Sigma())

	// Scheme context and keys
	kgen = ckks.NewKeyGenerator(params)
	sk, _ = kgen.GenKeyPairSparse(btpParams.H)
	encoder = ckks.NewEncoder(params)
	decryptor = ckks.NewDecryptor(params, sk)
	encryptor = ckks.NewEncryptor(params, sk)

	fmt.Println()
	fmt.Println("Generating bootstrapping keys...")
	rotations := btpParams.RotationsForBootstrapping(params.LogSlots())
	rotkeys := kgen.GenRotationKeysForRotations(rotations, true, sk)
	rlk := kgen.GenRelinearizationKey(sk, 2)
	btpKey := ckks.BootstrappingKey{Rlk: rlk, Rtks: rotkeys}
	if btp, err = ckks.NewBootstrapper(params, btpParams, btpKey); err != nil {
		panic(err)
	}
	fmt.Println("Done")

	// Generate a random plaintext
	valuesWant := make([]complex128, params.Slots())
	for i := range valuesWant {
		valuesWant[i] = utils.RandComplex128(-1, 1)
	}

	// plaintext = encoder.EncodeNew(valuesWant, params.LogSlots())
	cfs_tmp = make([]float64, params.N())
	for i := range cfs_tmp {
		cfs_tmp[i] = 0.0001 * float64(i)
	}
	fmt.Print("Boot in: ")
	prt_vec(cfs_tmp)
	plaintext = ckks.NewPlaintext(params, params.MaxLevel(), params.Scale()) // contain plaintext values
	encoder.EncodeCoeffs(cfs_tmp, plaintext)

	// Encrypt
	ciphertext1 := encryptor.EncryptNew(plaintext)

	// Decrypt, print and compare with the plaintext values
	// fmt.Println()
	// fmt.Println("Precision of values vs. ciphertext")
	// valuesTest1 := printDebug(params, ciphertext1, valuesWant, decryptor, encoder)

	// Bootstrap the ciphertext (homomorphic re-encryption)
	// It takes a ciphertext at level 0 (if not at level 0, then it will reduce it to level 0)
	// and returns a ciphertext at level MaxLevel - k, where k is the depth of the bootstrapping circuit.
	// CAUTION: the scale of the ciphertext MUST be equal (or very close) to params.Scale
	// To equalize the scale, the function evaluator.SetScale(ciphertext, parameters.Scale) can be used at the expense of one level.
	fmt.Println()
	fmt.Println("Bootstrapping...")

	start = time.Now()
	// ciphertext2 := btp.Bootstrapp(ciphertext1)
	// fmt.Printf("Done in %s \n", time.Since(start))

	// decryptor.Decrypt(ciphertext2, plaintext)
	// cfs_tmp = encoder.DecodeCoeffs(plaintext)
	// prt_vec(cfs_tmp)

	ciphertext2, ciphertext3 := btp.BootstrappConv(ciphertext1)
	fmt.Printf("Done in %s \n", time.Since(start))

	decryptor.Decrypt(ciphertext2, plaintext)
	value_out := encoder.Decode(plaintext, params.LogSlots())
	fmt.Printf("Boot out1: ")
	prt_vecc(value_out)

	decryptor.Decrypt(ciphertext3, plaintext)
	value_out = encoder.Decode(plaintext, params.LogSlots())
	fmt.Printf("Boot out2: ")
	prt_vecc(value_out)

	// Decrypt, print and compare with the plaintext values
	// fmt.Println()
	// fmt.Println("Precision of ciphertext vs. Bootstrapp(ciphertext)")
	// printDebug(params, ciphertext2, valuesTest1, decryptor, encoder)
}

func printDebug(params ckks.Parameters, ciphertext *ckks.Ciphertext, valuesWant []complex128, decryptor ckks.Decryptor, encoder ckks.Encoder) (valuesTest []complex128) {

	valuesTest = encoder.Decode(decryptor.DecryptNew(ciphertext), params.LogSlots())

	fmt.Println()
	fmt.Printf("Level: %d (logQ = %d)\n", ciphertext.Level(), params.LogQLvl(ciphertext.Level()))
	fmt.Printf("Scale: 2^%f\n", math.Log2(ciphertext.Scale))
	fmt.Printf("ValuesTest: %6.10f %6.10f %6.10f %6.10f...\n", valuesTest[0], valuesTest[1], valuesTest[2], valuesTest[3])
	fmt.Printf("ValuesWant: %6.10f %6.10f %6.10f %6.10f...\n", valuesWant[0], valuesWant[1], valuesWant[2], valuesWant[3])

	precStats := ckks.GetPrecisionStats(params, encoder, nil, valuesWant, valuesTest, params.LogSlots(), 0)

	fmt.Println(precStats.String())
	fmt.Println()

	return
}
