package main

import (
	"fmt"
	"math"
	"time"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/rlwe"
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
	// rlk := kgen.GenRelinearizationKey(sk, 2)
	// rots := []int{}
	// for i := 0; i < logN; i++ {
	// 	rots = append(rots, 1<<i)
	// }
	// rotkeys := kgen.GenRotationKeysForRotations(rots, true, sk)

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
	// slots := params.Slots()
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
	// fmt.Println("Input matrix: ", input)
	// fmt.Println("Ker1_in (1st part): ", ker1[0])

	fmt.Println()
	fmt.Println("=========================================")
	fmt.Println("   PLAINTEXT CREATION & ENCRYPTION       ")
	fmt.Println("=========================================")
	fmt.Println()

	start = time.Now()

	cfs_tmp := make([]float64, N)                                             // contain coefficient msgs
	plain_tmp := ckks.NewPlaintext(params, params.MaxLevel(), params.Scale()) // contain plaintext values

	encoder.EncodeCoeffs(input, plain_tmp)
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
		// pack_evaluator.Mul(ctxt_input, pl_ker[i], ctxt_out[i])
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
	cfs_tmp = encoder.DecodeCoeffs(plain_tmp)
	for i, val := range cfs_tmp {
		cfs_tmp[i] = math.Round(val*100) / 100
	}

	fmt.Println(reshape_conv_out(cfs_tmp, in_wid, ker_wid, batch))

	fmt.Printf("Done in %s \n", time.Since(start))

	// const cnum = 16                                                           // number of ciphertexts to be packed
	// cfs_tmp := make([]float64, N)                                             // contain coefficient msgs
	// plain_tmp := ckks.NewPlaintext(params, params.MaxLevel(), params.Scale()) // contain plaintext values
	// ctxts := make([]*ckks.Ciphertext, cnum)                                   // ctxts to be packed

	// for i := 0; i < cnum; i++ {
	// 	for j := 0; j < N; j += cnum {
	// 		cfs_tmp[j] = float64(j + i)
	// 	}
	// 	encoder.EncodeCoeffs(cfs_tmp, plain_tmp)
	// 	ctxts[i] = encryptor.EncryptNew(plain_tmp)
	// 	cfs_tmp = make([]float64, N)
	// }

	// fmt.Printf("Done in %s \n", time.Since(start))

	// fmt.Println()
	// fmt.Println("===============================================")
	// fmt.Println("     			   EVALUATION					")
	// fmt.Println("===============================================")
	// fmt.Println()

	// start = time.Now()

	// result := pack_ctxts(pack_evaluator, ctxts, cnum, plain_idx, params)

	// fmt.Printf("Done in %s \n", time.Since(start))

	// fmt.Println()
	// fmt.Println("=========================================")
	// fmt.Println("              DECRYPTION                 ")
	// fmt.Println("=========================================")
	// fmt.Println()

	// start = time.Now()

	// decryptor.Decrypt(result, plain_tmp)
	// cfs_tmp = encoder.DecodeCoeffs(plain_tmp)
	// for i, val := range cfs_tmp {
	// 	cfs_tmp[i] = math.Round(val*100) / 100
	// }
	// // fmt.Println(cfs_tmp)

	// fmt.Printf("Done in %s \n", time.Since(start))

	// printDebug(params, ciphertext, values, decryptor, encoder)

	// decryptor.Decrypt(ctxts[i], plain_tmp)
	// cfs_tmp = encoder.DecodeCoeffs(plain_idx[i])

	// for i := 0; i < slots; i++ {
	// 	cvalues[i] = ring.NewComplex(ring.NewFloat(0.0, 0), ring.NewFloat(0.0, 0))
	// }

	// for i := range values {
	// 	values[i] *= complex(0, 1)
	// }

	// printDebug(params, ciphertext, values, decryptor, encoder)
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
