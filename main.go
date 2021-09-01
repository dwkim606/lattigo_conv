package main

import (
	"fmt"
	"math"
	"strconv"
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

	logN := 10
	in_wid := 8
	ker_wid := 3
	print := true
	N := (1 << logN)
	batch := N / (in_wid * in_wid)
	pos := 0
	ECD_LV := 3

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
	r_idx, m_idx := gen_extend_full(N/2, 2*in_wid, pos, true, true)
	var rotations []int
	for k := range r_idx {
		rotations = append(rotations, k)
	}
	for k := range m_idx {
		rotations = append(rotations, k)
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
	if btp, err = ckks.NewBootstrapper(params, btpParams, btpKey); err != nil {
		panic(err)
	}
	fmt.Printf("Done in %s \n", time.Since(start))

	ext_input := testBRrot(logN, in_wid) // Takes arranged input (assume intermediate layers)

	input := make([]float64, N)
	for i := range input {
		input[i] = 1.0 * float64(ext_input[i]) / float64(N)
	}
	start := time.Now()
	plain_in := ckks.NewPlaintext(params, params.MaxLevel(), params.Scale()) // contain plaintext values
	encoder.EncodeCoeffs(input, plain_in)
	ctxt_in := encryptor.EncryptNew(plain_in)
	fmt.Printf("Encryption: Done in %s \n", time.Since(start))

	fmt.Println("Input matrix: ")
	prt_vec(input)
	int_tmp := make([]int, N)
	for i := range input {
		int_tmp[i] = int(float64(N) * input[i])
	}
	fmt.Print("Input: \n")
	for b := 0; b < batch/4; b++ {
		print_vec("input ("+strconv.Itoa(b)+")", int_tmp, 2*in_wid, b)
	}

	pl_ker := prepKer(params, encoder, encryptor, N, in_wid, ker_wid, ECD_LV, true)

	fmt.Print("Boot in: ")
	fmt.Println()
	fmt.Println("Precision of values vs. ciphertext")
	values_test := printDebugCfs(params, ctxt_in, input, decryptor, encoder)

	fmt.Println("Bootstrapping... Ours:")
	start = time.Now()
	ctxt1, ctxt2 := btp.BootstrappConv_PreStoC(ctxt_in)
	fmt.Printf("Done in %s \n", time.Since(start))
	fmt.Println("after Boot: LV = ", ctxt1.Level(), " Scale = ", math.Log2(ctxt1.Scale))

	ctxt1 = ext_ctxt(evaluator, encoder, ctxt1, r_idx, m_idx, params)
	ctxt2 = ext_ctxt(evaluator, encoder, ctxt2, r_idx, m_idx, params)

	ciphertext := btp.BootstrappConv_StoC(ctxt1, ctxt2)

	fmt.Printf("Boot out: ")

	// Only for checking the correctness
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

	// Move the ctxt position before convolution
	xi_tmp := make([]float64, N)
	xi_tmp[(ker_wid-1)*(4*in_wid+1)] = 1.0
	xi_plain := ckks.NewPlaintext(params, ECD_LV, 1.0)
	encoder.EncodeCoeffs(xi_tmp, xi_plain)
	encoder.ToNTT(xi_plain)
	evaluator.Mul(ciphertext, xi_plain, ciphertext)

	ctxt_result := conv_then_pack(params, pack_evaluator, ciphertext, pl_ker, plain_idx, batch/16)

	fmt.Println()
	fmt.Println("=========================================")
	fmt.Println("              DECRYPTION                 ")
	fmt.Println("=========================================")
	fmt.Println()

	plain_tmp := ckks.NewPlaintext(params, params.MaxLevel(), params.Scale())
	start = time.Now()
	decryptor.Decrypt(ctxt_result, plain_tmp)
	cfs_tmp := reshape_conv_out(encoder.DecodeCoeffs(plain_tmp), 4*in_wid, ker_wid, batch/16)
	// cfs_tmp := make([]float64, N)
	// for i := range cfs_tmp {
	// 	cfs_tmp[i] = float64(i + 1)
	// }

	if print {
		fmt.Print("Result: \n")
		prt_mat(cfs_tmp, batch/16, 2*in_wid)
	}
	fmt.Printf("Done in %s \n", time.Since(start))

	int_tmpn := make([]int, N)
	for i := range cfs_tmp {
		int_tmpn[i] = int(cfs_tmp[i])
	}
	fmt.Print("Output: \n")
	for b := 0; b < batch/16; b++ {
		print_vec("output ("+strconv.Itoa(b)+")", int_tmpn, 4*in_wid, b)
	}

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

// print slice as a 2D slice with rowLen row length, only shows (show, show) entries show = 0 : print all
func prt_mat(vec []float64, rowLen int, show int) {
	mat_size := len(vec) / rowLen
	j, k := 1, 1
	for i := 0; i < len(vec); i += rowLen {
		if (show == 0) || ((j <= show) && (k <= show)) {
			fmt.Printf("(%d, %d): ", j, k)
			prt_vec(vec[i : i+rowLen])
		}
		k++
		if k*k > mat_size {
			k = 1
			j++
		}
	}
}
