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

const pow = 5 // making sure that ReLu can cover values in [-2^pow, 2^pow].

type context struct {
	logN    int
	N       int
	ECD_LV  int
	in_wids []int // input widths including padding
	kp_wids []int // keep widths among input widths
	// pads           map[int]int
	ext_idx        map[int][][]int         // ext_idx for keep_vec (saved for each possible input width) map: in_wid, [up/low]
	r_idx          map[int][]map[int][]int // r_idx for compr_vec (or ext_vec) map: in_wid [pos] map: rot
	m_idx          map[int][]map[int][]int // m_idx , map: in_wid [pos] map: rot
	pl_idx         []*ckks.Plaintext
	params         ckks.Parameters
	encoder        ckks.Encoder
	encryptor      ckks.Encryptor
	decryptor      ckks.Decryptor
	evaluator      ckks.Evaluator
	pack_evaluator ckks.Evaluator
	btp            *ckks.Bootstrapper
}

func newContext(logN int, in_wids, kp_wids []int, boot bool, kind string) *context {
	cont := context{N: (1 << logN), logN: logN, ECD_LV: 1}
	cont.in_wids = make([]int, len(in_wids))
	copy(cont.in_wids, in_wids)
	cont.kp_wids = make([]int, len(kp_wids))
	copy(cont.kp_wids, kp_wids)

	btpParams := ckks.DefaultBootstrapParams[6]
	cont.params, err = btpParams.Params()
	if err != nil {
		panic(err)
	}
	fmt.Printf("CKKS parameters: logN = %d, logSlots = %d, h = %d, logQP = %d, levels = %d, scale= 2^%f, sigma = %f \n",
		cont.params.LogN(), cont.params.LogSlots(), btpParams.H, cont.params.LogQP(), cont.params.QCount(), math.Log2(cont.params.Scale()), cont.params.Sigma())
	if cont.params.N() != cont.N {
		fmt.Println("Set Boot logN to", logN)
		panic("Boot N != N")
	}
	// Gen rotations
	var rotations []int
	cont.ext_idx = make(map[int][][]int)
	cont.r_idx = make(map[int][]map[int][]int)
	cont.m_idx = make(map[int][]map[int][]int)
	var iter int
	half := true // only for DCGAN

	switch kind {
	case "Conv": // we assume manual padding using kp_wid
		if boot {
			iter = 2 // we assume full padding, i.e., up and low is both nonzero
			for i, elt := range cont.in_wids {
				cont.ext_idx[elt] = make([][]int, iter)
				for ul := 0; ul < iter; ul++ {
					cont.ext_idx[elt][ul] = gen_keep_vec(cont.N/2, elt, cont.kp_wids[i], ul)
				}
			}
		}
	case "TransConv": // we assume manual padding using kp_wid
		if boot {
			iter = 2 // we assume full padding, i.e., up and low is both nonzero
			for i, elt := range cont.in_wids {
				cont.r_idx[elt] = make([]map[int][]int, 4)
				for ul := 0; ul < iter; ul++ {
					cont.r_idx[elt][ul] = gen_extend_full(cont.N/2, elt, kp_wids[i], 0, ul)
					for k := range cont.r_idx[elt][ul] {
						rotations = append(rotations, k)
					}
				}
			}
		}
	case "Resnet": // Generate ext_idx for extracting valid values from conv with "same" padding
		iter = 1 // since we use half padding, i.e., lower part is all zero
		for _, elt := range cont.in_wids {
			cont.ext_idx[elt] = make([][]int, iter)
			for ul := 0; ul < iter; ul++ {
				cont.ext_idx[elt][ul] = gen_keep_vec(cont.N/2, elt, elt/2, ul)
			}
			cont.r_idx[elt] = make([]map[int][]int, 4)
			for pos := 0; pos < 4; pos++ {
				cont.r_idx[elt][pos] = gen_comprs_full(cont.N/2, elt, elt/2, pos, 0)
				for k := range cont.r_idx[elt][pos] {
					rotations = append(rotations, k)
				}
			}
		}
	case "DCGAN": // Generate rotations for EXT_FULL
		for _, elt := range cont.in_wids {
			cont.r_idx[elt] = make([]map[int][]int, 4)
			cont.m_idx[elt] = make([]map[int][]int, 4)
			for pos := 0; pos < 4; pos++ {
				cont.r_idx[elt][pos], cont.m_idx[elt][pos] = gen_extend_full_nhf(cont.N/2, elt, pos, half, half)
				for k := range cont.r_idx[elt][pos] {
					rotations = append(rotations, k)
				}
				for k := range cont.m_idx[elt][pos] {
					rotations = append(rotations, k)
				}
			}
		}
	default:
		panic("Wrong kinds!")
	}

	// Scheme context and keys for evaluation (no Boot)
	kgen := ckks.NewKeyGenerator(cont.params)
	sk, _ := kgen.GenKeyPairSparse(btpParams.H)
	rlk := kgen.GenRelinearizationKey(sk, 2)
	rotkeys := kgen.GenRotationKeysForRotations(rotations, false, sk)
	cont.encoder = ckks.NewEncoder(cont.params)
	cont.decryptor = ckks.NewDecryptor(cont.params, sk)
	cont.encryptor = ckks.NewEncryptor(cont.params, sk)
	cont.evaluator = ckks.NewEvaluator(cont.params, rlwe.EvaluationKey{Rlk: rlk, Rtks: rotkeys})

	cont.pl_idx, cont.pack_evaluator = gen_idxNlogs(cont.ECD_LV, kgen, sk, cont.encoder, cont.params)

	if boot {
		fmt.Println("Generating bootstrapping keys...")
		start = time.Now()
		rotations = btpParams.RotationsForBootstrapping(cont.params.LogSlots())
		rotkeys = kgen.GenRotationKeysForRotations(rotations, true, sk)
		btpKey := ckks.BootstrappingKey{Rlk: rlk, Rtks: rotkeys}
		if cont.btp, err = ckks.NewBootstrapper_mod(cont.params, btpParams, btpKey); err != nil {
			panic(err)
		}
		fmt.Printf("Done in %s \n", time.Since(start))
	}

	return &cont
}

func main() {

	// testConv_noBoot(7, 8, 8, true)

	// iter, _ := strconv.Atoi(os.Args[1])
	// testResNet_in(0)

	// testConv_BNRelu_BL(15, 16, 3, true)
	// testConv_noBoot_BL(6, 4, 3, false)

	// testBRrot()
	// testConv_noBoot(true, true)
	// testConv_noBoot(7, 8, 5, true, true)
	testConv_BNRelu(false, false, true)
	// testReduceMean()
	// testResNet()
	// testDCGAN()

	// input := testBRrot(logN, in_wid)
	// testPoly()
	// testBoot()
	// testBootFast_Conv(input, logN, in_wid, ker_wid, print)
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

// vec = (B0 input, B1 input, ... ) format for BaseLine Test, each batch is sqr-sized
// print (i,j)-th position in [batches], only shows (show, show) entries show = 0 : print all
func prt_mat_BL(vec []complex128, batch, show int) {
	in_wid := int(math.Sqrt(float64(len(vec) / batch)))
	tmp := make([]float64, batch)
	for i := 1; i < in_wid+1; i++ {
		for j := 1; j < in_wid+1; j++ {
			if (show == 0) || (((i == 1) || (i == show)) && ((j <= 3) || (j >= show-3))) {
				fmt.Printf("(%d, %d): ", i, j)
				for b := 0; b < batch; b++ {
					tmp[b] = real(vec[in_wid*in_wid*b+(i-1)*in_wid+(j-1)])
				}
				prt_vec(tmp)
			}
		}
	}
}

// vec = arrgvec with batch batches, each batch is sqr-sized
// print (i,j)-th position in [batches], only shows (show, show) entries show = 0 : print all
func prt_mat(vec []float64, batch, show int) {
	mat_size := len(vec) / batch
	row := int(math.Sqrt(float64(mat_size)))
	j, k := 1, 1
	for i := 0; i < len(vec); i += batch {
		if (show == 0) || (((j <= show) || (j > row-show)) && ((k <= show) || (k > (row - show)))) {
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

// only (sj,sk) element in all batches
func prt_mat_one(vec []float64, batch, sj, sk int) (out []float64) {
	mat_size := len(vec) / batch
	j, k := 1, 1
	for i := 0; i < len(vec); i += batch {
		if (j == sj) && (k == sk) {
			fmt.Print(vec[i : i+batch])
			out = vec[i : i+batch]
		}
		k++
		if k*k > mat_size {
			k = 1
			j++
		}
	}
	return out
}

func check(e error) {
	if e != nil {
		panic(e)
	}
}

func readTxt(name_file string, size int) (input []float64) {

	file, err := os.Open(name_file)
	check(err)
	scanner := bufio.NewScanner(file)
	scanner.Split(bufio.ScanWords)

	for scanner.Scan() {
		add, _ := strconv.ParseFloat(scanner.Text(), 64)
		input = append(input, add)
	}
	file.Close()
	// fmt.Print(input)

	if (size != 0) && (len(input) != size) {
		panic("input size inconsistent!")
	}

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

func prep_Input(input []float64, raw_in_wid, in_wid, N int, trans, printResult bool) (out []float64) {
	out = make([]float64, N)
	batch := N / (in_wid * in_wid)
	k := 0

	if trans {
		for i := 0; i < in_wid/2; i++ {
			for j := 0; j < in_wid/2; j++ {
				for b := 0; b < batch; b++ {
					if (i < raw_in_wid) && (j < raw_in_wid) {
						out[(2*i+1)*in_wid*batch+(2*j+1)*batch+b] = input[k]
						k++
					}
				}
			}
		}
	} else {
		for i := 0; i < in_wid; i++ {
			for j := 0; j < in_wid; j++ {
				for b := 0; b < batch; b++ {
					if (i < raw_in_wid) && (j < raw_in_wid) {
						out[i*in_wid*batch+j*batch+b] = input[k]
						k++
					}
				}
			}
		}
	}

	if printResult {
		fmt.Println("Input matrix: ")
		prt_mat(out, batch, 3)
	}

	return out
}
