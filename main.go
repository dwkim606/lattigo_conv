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
	print := true
	logN := 10
	in_wid := 8
	ker_wid := 5
	N := (1 << logN)
	st_batch := N / (2 * in_wid * 2 * in_wid) // We also consider zero-paddings  // must be adjusted when in_wid is not power of 2
	end_batch := 1
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
	var rotations []int
	r_idx := make([]map[int][]int, 4)
	m_idx := make([]map[int][]int, 4)
	for pos := 0; pos < 4; pos++ {
		r_idx[pos], m_idx[pos] = gen_extend_full(N/2, 2*in_wid, pos, true, true)
		for k := range r_idx[pos] {
			rotations = append(rotations, k)
		}
		for k := range m_idx[pos] {
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
	if btp, err = ckks.NewBootstrapper(params, btpParams, btpKey); err != nil {
		panic(err)
	}
	fmt.Printf("Done in %s \n", time.Since(start))

	ext_input := testBRrot(logN, in_wid, false) // Takes arranged input (assume intermediate layers)  // print only outputs first (st_batch) batches

	input := make([]float64, N)
	for i := range input {
		input[i] = 1.0 * float64(ext_input[i]) / float64(in_wid*in_wid*st_batch)
	}
	start := time.Now()
	plain_in := ckks.NewPlaintext(params, params.MaxLevel(), params.Scale()) // contain plaintext values
	encoder.EncodeCoeffs(input, plain_in)
	ctxt_in := encryptor.EncryptNew(plain_in)
	fmt.Printf("Encryption: Done in %s \n", time.Since(start))

	if print {
		fmt.Println("vec size: ", N)
		fmt.Println("input width: ", in_wid)
		fmt.Println("kernel width: ", ker_wid)
		fmt.Println("num batches (in 1 ctxt with padding): ", st_batch)
	}

	pl_ker := prepKer(params, encoder, encryptor, in_wid, ker_wid, st_batch, end_batch, ECD_LV)

	fmt.Print("Boot in: ")
	fmt.Println()
	fmt.Println("Precision of values vs. ciphertext")
	in_cfs := printDebugCfs(params, ctxt_in, input, decryptor, encoder)

	fmt.Println("Bootstrapping... Ours (Pre StoC):")
	start = time.Now()
	ctxt1, ctxt2, _ := btp.BootstrappConv_PreStoC(ctxt_in)
	fmt.Printf("Done in %s \n", time.Since(start))
	fmt.Println("after Boot: LV = ", ctxt1.Level(), " Scale = ", math.Log2(ctxt1.Scale))

	// Only for checking the correctness
	in_cfs_1_pBoot := make([]float64, params.Slots())
	in_cfs_2_pBoot := make([]float64, params.Slots())
	in_slots := make([]complex128, params.Slots()) // first part of ceffs
	for i := range in_cfs_1_pBoot {
		in_cfs_1_pBoot[i] = in_cfs[reverseBits(uint32(i), params.LogSlots())] // first part of coeffs
		in_cfs_2_pBoot[i] = in_cfs[reverseBits(uint32(i), params.LogSlots())+uint32(params.Slots())]
		in_slots[i] = complex(in_cfs_1_pBoot[i], 0)
	}
	ext1_tmp := extend_full_fl(in_cfs_1_pBoot, 2*in_wid, 3, true, true)
	ext2_tmp := extend_full_fl(in_cfs_2_pBoot, 2*in_wid, 3, true, true)
	for i := range in_cfs_1_pBoot {
		in_cfs_1_pBoot[i] = ext1_tmp[reverseBits(uint32(i), params.LogSlots())]
		in_cfs_2_pBoot[i] = ext2_tmp[reverseBits(uint32(i), params.LogSlots())]
	}
	in_cfs_pBoot := append(in_cfs_1_pBoot, in_cfs_2_pBoot...) // After rot(ext) and boot

	in_slots = printDebug(params, ctxt1, in_slots, decryptor, encoder)
	ctxt1 = evalReLU(params, evaluator, ctxt1, 1.0)
	fmt.Println("ReLU done.")

	values_ReLU := make([]complex128, len(in_slots))
	for i := range values_ReLU {
		values_ReLU[i] = complex(math.Max(0, real(in_slots[i])), 0)
	}
	printDebug(params, ctxt1, values_ReLU, decryptor, encoder)

	ext_ctxt1 := make([]*ckks.Ciphertext, 4) // for extend (rotation) of ctxt_in
	// new_ctxt2 := make([]*ckks.Ciphertext, 4)		// do not need if we use po2 inputs dims
	ciphertext := make([]*ckks.Ciphertext, 4) // after Bootstrapping

	for pos := 0; pos < 4; pos++ {
		ext_ctxt1[pos] = ext_ctxt(evaluator, encoder, ctxt1, r_idx[pos], m_idx[pos], params)
		// new_ctxt2[pos] = ext_ctxt(evaluator, encoder, ctxt2, r_idx[pos], m_idx[pos], params)
		ciphertext[pos] = btp.BootstrappConv_StoC(ext_ctxt1[pos], ctxt2)
	}

	fmt.Printf("Boot out: ")

	printDebugCfs(params, ciphertext[3], in_cfs_pBoot, decryptor, encoder)

	ctxt_result := conv_then_pack(params, pack_evaluator, ciphertext, pl_ker, plain_idx, end_batch)

	fmt.Println()
	fmt.Println("=========================================")
	fmt.Println("              DECRYPTION                 ")
	fmt.Println("=========================================")
	fmt.Println()

	plain_tmp := ckks.NewPlaintext(params, params.MaxLevel(), params.Scale())
	start = time.Now()
	decryptor.Decrypt(ctxt_result, plain_tmp)
	cfs_tmp := reshape_conv_out(encoder.DecodeCoeffs(plain_tmp), 2*in_wid, end_batch)

	if print {
		fmt.Print("Result: \n")
		prt_mat(cfs_tmp, end_batch, 2*in_wid)
	}
	fmt.Printf("Done in %s \n", time.Since(start))

	cfs_tmp = encoder.DecodeCoeffs(plain_tmp)
	int_tmpn := make([]int, N)
	for i := range cfs_tmp {
		int_tmpn[i] = int(cfs_tmp[i])
	}
	fmt.Print("Output: \n")
	for b := 0; b < end_batch; b++ {
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
