package main

import (
	"bufio"
	"fmt"
	"log"
	"math"
	"os"
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

const pow = 4

type context struct {
	logN           int
	N              int
	pad            int
	ECD_LV         int
	in_wids        []int           // possible input widths
	ext_idx        map[int][][]int // ext_idx for keep_vec (saved for each possible input width)
	pl_idx         []*ckks.Plaintext
	params         ckks.Parameters
	encoder        ckks.Encoder
	encryptor      ckks.Encryptor
	decryptor      ckks.Decryptor
	evaluator      ckks.Evaluator
	pack_evaluator ckks.Evaluator
	btp            *ckks.Bootstrapper
}

func newContext(logN, pad, ECD_LV int, in_wids []int) *context {
	cont := context{N: (1 << logN), logN: logN, pad: pad, ECD_LV: ECD_LV}
	cont.in_wids = make([]int, len(in_wids))
	copy(cont.in_wids, in_wids)

	btpParams := ckks.DefaultBootstrapParams[6]
	cont.params, err = btpParams.Params()
	if err != nil {
		panic(err)
	}
	fmt.Printf("CKKS parameters: logN = %d, logSlots = %d, h = %d, logQP = %d, levels = %d, scale= 2^%f, sigma = %f \n",
		cont.params.LogN(), cont.params.LogSlots(), btpParams.H, cont.params.LogQP(), cont.params.QCount(), math.Log2(cont.params.Scale()), cont.params.Sigma())

	// Generate ext_idx for extracting valid values from conv with "same" padding

	cont.ext_idx = make(map[int][][]int)
	for _, elt := range cont.in_wids {
		cont.ext_idx[elt] = make([][]int, 2)
		for i := 0; i < 2; i++ {
			cont.ext_idx[elt][i] = gen_keep_vec(cont.logN, elt, cont.pad, i)
		}
	}

	// Scheme context and keys for evaluation (no Boot)
	kgen := ckks.NewKeyGenerator(cont.params)
	sk, _ := kgen.GenKeyPairSparse(btpParams.H)
	rlk := kgen.GenRelinearizationKey(sk, 2)
	cont.encoder = ckks.NewEncoder(cont.params)
	cont.decryptor = ckks.NewDecryptor(cont.params, sk)
	cont.encryptor = ckks.NewEncryptor(cont.params, sk)
	cont.evaluator = ckks.NewEvaluator(cont.params, rlwe.EvaluationKey{Rlk: rlk})

	cont.pl_idx, cont.pack_evaluator = gen_idxNlogs(cont.ECD_LV, kgen, sk, cont.encoder, cont.params)

	fmt.Println("Generating bootstrapping keys...")
	start = time.Now()
	rotations := btpParams.RotationsForBootstrapping(cont.params.LogSlots())
	rotkeys := kgen.GenRotationKeysForRotations(rotations, true, sk)
	btpKey := ckks.BootstrappingKey{Rlk: rlk, Rtks: rotkeys}
	if cont.btp, err = ckks.NewBootstrapper_mod(cont.params, btpParams, btpKey); err != nil {
		panic(err)
	}
	fmt.Printf("Done in %s \n", time.Since(start))

	return &cont
}

func main() {

	testBRrot(8, 8, true)

	// logN := 8
	// in_wids := []int{8, 4}
	// ker_wid := 3
	// input_pad := (ker_wid - 1) / 2
	// ECD_LV := 1
	// cont := newContext(logN, input_pad, ECD_LV, in_wids)

	// ker_size := ker_wid * ker_wid
	// in_wid := in_wids[0]
	// batch := cont.N / (in_wid * in_wid)
	// alpha := 0.0 // 0.3 => leakyrelu

	// input := make([]float64, cont.N)

	// k := 0.0
	// for i := 0; i < in_wid; i++ {
	// 	for j := 0; j < in_wid; j++ {
	// 		for b := 0; b < batch; b++ {
	// 			if (i < in_wid-input_pad) && (j < in_wid-input_pad) {
	// 				input[i*in_wid*batch+j*batch+b] = k
	// 				k += (1.0 / float64(batch*(in_wid-input_pad)*(in_wid-input_pad)))
	// 			}
	// 		}
	// 	}
	// }
	// ker_in := make([]float64, batch*batch*ker_size)
	// for i := range ker_in {
	// 	ker_in[i] = 1.0 * float64(i) / float64(batch*batch*ker_size)
	// }
	// bn_a := make([]float64, batch)
	// bn_b := make([]float64, batch)
	// for i := range bn_a {
	// 	bn_a[i] = 0.08 // * float64(i) / float64(batch)
	// 	bn_b[i] = 0.0 * float64(i) / float64(batch)
	// }

	// fmt.Println("vec size: ", cont.N)
	// fmt.Println("input width: ", in_wid)
	// fmt.Println("kernel width: ", ker_wid)
	// fmt.Println("num batches: ", batch)
	// fmt.Println("Input matrix: ")
	// prt_vec(input)

	// start = time.Now()
	// pl_input := ckks.NewPlaintext(cont.params, cont.ECD_LV, cont.params.Scale()) // contain plaintext values
	// cont.encoder.EncodeCoeffs(input, pl_input)
	// ct_input := cont.encryptor.EncryptNew(pl_input)
	// fmt.Printf("Encryption done in %s \n", time.Since(start))

	// // ResNet Block 1
	// num_blc1 := 7
	// ct_layer := make([]*ckks.Ciphertext, num_blc1+1)
	// ct_layer[0] = ct_input
	// prt_result := false
	// for i := 1; i <= num_blc1; i++ {
	// 	if i == num_blc1 {
	// 		prt_result = true
	// 	}
	// 	ct_layer[i] = evalConv_BNRelu(cont, ct_layer[i-1], ker_in, bn_a, bn_b, alpha, in_wid, ker_wid, prt_result)
	// 	fmt.Println("Layer ", i, "done!")
	// }

	// testConv_BNRelu(8, 5, 2, true)

	// testConv_noBoot(7, 8, 7, true)

	// testDCGAN()

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

func check(e error) {
	if e != nil {
		panic(e)
	}
}

func readTxt(name_file string) []float64 {

	file, err := os.Open(name_file)
	check(err)
	scanner := bufio.NewScanner(file)
	scanner.Split(bufio.ScanWords)

	var input []float64
	for scanner.Scan() {
		add, _ := strconv.ParseFloat(scanner.Text(), 64)
		input = append(input, add)
	}
	file.Close()
	// fmt.Print(input)

	return input

}

func writeTxt(name_file string, input []float64) {
	file, err := os.OpenFile(name_file, os.O_TRUNC|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Fatalf("failed creating file: %s", err)
	}

	datawriter := bufio.NewWriter(file)
	for _, data := range input {
		_, _ = datawriter.WriteString(strconv.FormatFloat(data, 'e', -1, 64) + "\n")
	}

	datawriter.Flush()
	file.Close()
}
