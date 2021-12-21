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

func newContext(logN, ker_wid int, in_wids, kp_wids []int, boot bool, kind string) *context {
	cont := context{N: (1 << logN), logN: logN, ECD_LV: 1}
	cont.in_wids = make([]int, len(in_wids))
	copy(cont.in_wids, in_wids)
	cont.kp_wids = make([]int, len(kp_wids))
	copy(cont.kp_wids, kp_wids)

	btpParams := ckks.DefaultBootstrapParams[6]
	if (kind == "BL_Conv") || (kind == "BL_StrConv") || (kind == "BL_TransConv") || (kind == "BL_Resnet") || (kind == "BL_Imagenet") {
		btpParams = ckks.DefaultBootstrapParams[7]
	}
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
	case "BL_Conv":
		for _, elt := range cont.in_wids {
			for k := -(ker_wid / 2); k <= ker_wid/2; k++ {
				for k2 := -(ker_wid / 2); k2 <= ker_wid/2; k2++ {
					rotations = append(rotations, k*elt+k2)
				}
			}
			out_batch := (cont.N / 2) / (elt * elt)
			for k := 1; k < out_batch; k++ {
				rotations = append(rotations, k*elt*elt)
			}
		}
	case "BL_StrConv":
		for _, elt := range cont.in_wids {
			for k := -(ker_wid / 2); k <= ker_wid/2; k++ { // rotations for conv
				for k2 := -(ker_wid / 2); k2 <= ker_wid/2; k2++ {
					rotations = append(rotations, k*elt+k2)
				}
			}
			out_batch := (cont.N / 2) / (elt * elt)
			for k := 1; k < out_batch; k++ { // rotations for conv
				rotations = append(rotations, k*elt*elt)
			}
			for pos := 0; pos < 4; pos++ { // for final rotations for after strides
				rotations = append(rotations, -pos*elt*elt/4)
			}
			cont.m_idx[elt] = make([]map[int][]int, 1)
			cont.r_idx[elt] = make([]map[int][]int, 1)
			cont.m_idx[elt][0], cont.r_idx[elt][0] = gen_comprs_BL(cont.N/2, elt)

			for k := range cont.m_idx[elt][0] {
				rotations = append(rotations, k)
			}
			for k := range cont.r_idx[elt][0] {
				rotations = append(rotations, k)
			}
		}
	case "BL_TransConv":
		for _, elt := range cont.in_wids {
			for k := -(ker_wid / 2); k <= ker_wid/2; k++ { // rotations for conv
				for k2 := -(ker_wid / 2); k2 <= ker_wid/2; k2++ {
					rotations = append(rotations, k*2*elt+k2)
				}
			}
			out_batch := (cont.N / 2) / (4 * elt * elt)
			for k := 1; k < out_batch; k++ { // rotations for conv
				rotations = append(rotations, 4*k*elt*elt)
			}
			for pos := 0; pos < 4; pos++ { // for final rotations for prep expand
				rotations = append(rotations, pos*elt*elt)
			}
			cont.m_idx[elt] = make([]map[int][]int, 1)
			cont.r_idx[elt] = make([]map[int][]int, 1)
			cont.m_idx[elt][0], cont.r_idx[elt][0] = gen_expand_BL(cont.N/2, elt)

			for k := range cont.m_idx[elt][0] {
				rotations = append(rotations, k)
			}
			for k := range cont.r_idx[elt][0] {
				rotations = append(rotations, k)
			}
		}
	case "BL_Resnet": // need rots for strConv and Conv
		for i, elt := range cont.in_wids {
			for k := -(ker_wid / 2); k <= ker_wid/2; k++ { // rotations for conv
				for k2 := -(ker_wid / 2); k2 <= ker_wid/2; k2++ {
					rotations = append(rotations, k*elt+k2)
				}
			}
			max_out_batch := cont.N / (2 * elt * elt) // originally (cont.N / 2) / (elt * elt)
			norm := 1 << (i + 1)
			if i == 0 {
				norm = 1
			}
			for k := 1; k < max_out_batch/norm; k++ { // rotations for post conv
				rotations = append(rotations, norm*k*elt*elt)
			}
			for pos := 0; pos < 4; pos++ { // for final rotations for after strides
				rotations = append(rotations, -pos*elt*elt/4)
			}
			cont.m_idx[elt] = make([]map[int][]int, 1)
			cont.r_idx[elt] = make([]map[int][]int, 1)
			cont.m_idx[elt][0], cont.r_idx[elt][0] = gen_comprs_BL(cont.N/2, elt)

			for k := range cont.m_idx[elt][0] {
				rotations = append(rotations, k)
			}
			for k := range cont.r_idx[elt][0] {
				rotations = append(rotations, k)
			}
			for i := 1; i < 64; i *= 2 { // for reduce mean & FC
				rotations = append(rotations, i)
			}
			for i := 1; i < 4; i *= 2 {
				rotations = append(rotations, i*16*64*8)
			}
			for i := 1; i < 16; i++ {
				rotations = append(rotations, i*64*8)
			}
			rotations = removeDuplicateInt(rotations)
		}
	case "BL_Imagenet":
		for _, elt := range cont.in_wids {
			for k := -(ker_wid / 2); k <= ker_wid/2; k++ { // rotations for conv
				for k2 := -(ker_wid / 2); k2 <= ker_wid/2; k2++ {
					rotations = append(rotations, k*elt+k2)
				}
			}
			out_batch := (cont.N / 2) / (elt * elt)
			for k := 1; k < out_batch; k++ { // rotations for conv
				rotations = append(rotations, k*elt*elt)
			}
			for pos := 0; pos < 4; pos++ { // for final rotations for after strides
				rotations = append(rotations, -pos*elt*elt/4)
			}
			cont.m_idx[elt] = make([]map[int][]int, 1)
			cont.r_idx[elt] = make([]map[int][]int, 1)
			cont.m_idx[elt][0], cont.r_idx[elt][0] = gen_comprs_BL(cont.N/2, elt)

			for k := range cont.m_idx[elt][0] {
				rotations = append(rotations, k)
			}
			for k := range cont.r_idx[elt][0] {
				rotations = append(rotations, k)
			}
			rotations = removeDuplicateInt(rotations)
		}
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
	case "StrConv":
		if boot {
			iter = 2 // we assume full padding, i.e., up and low is both nonzero
			for i, elt := range cont.in_wids {
				cont.r_idx[elt] = make([]map[int][]int, 4)
				for ul := 0; ul < iter; ul++ {
					cont.r_idx[elt][ul] = gen_comprs_full(cont.N/2, elt, cont.kp_wids[i], 0, ul)
					for k := range cont.r_idx[elt][ul] {
						rotations = append(rotations, k)
					}
				}
			}
		}
	case "TransConv": // we assume manual padding using kp_wid
		if boot {
			iter = 2 // we assume full padding, i.e., up and low is both nonzero
			for i, elt := range cont.in_wids {
				cont.r_idx[elt] = make([]map[int][]int, 4)
				for ul := 0; ul < iter; ul++ {
					cont.r_idx[elt][ul] = gen_extend_full(cont.N/2, elt, cont.kp_wids[i], 0, ul)
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
	fmt.Println("Num Rotations: ", len(rotations))
	rotkeys := kgen.GenRotationKeysForRotations(rotations, false, sk)
	cont.encoder = ckks.NewEncoder(cont.params)
	cont.decryptor = ckks.NewDecryptor(cont.params, sk)
	cont.encryptor = ckks.NewEncryptor(cont.params, sk)
	cont.evaluator = ckks.NewEvaluator(cont.params, rlwe.EvaluationKey{Rlk: rlk, Rtks: rotkeys})

	if !((kind == "BL_Conv") || (kind == "BL_StrConv") || (kind == "BL_TransConv") || (kind == "BL_Resnet") || (kind == "BL_Imagenet")) {
		cont.pl_idx, cont.pack_evaluator = gen_idxNlogs(cont.ECD_LV, kgen, sk, cont.encoder, cont.params)
	}

	if boot {
		fmt.Println("Generating bootstrapping keys...")
		start = time.Now()
		rotations = btpParams.RotationsForBootstrapping(cont.params.LogSlots())
		rotkeys = kgen.GenRotationKeysForRotations(rotations, true, sk)
		btpKey := ckks.BootstrappingKey{Rlk: rlk, Rtks: rotkeys}

		if (kind == "BL_Conv") || (kind == "BL_StrConv") || (kind == "BL_TransConv") || (kind == "BL_Resnet") || (kind == "BL_Imagenet") {
			if cont.btp, err = ckks.NewBootstrapper(cont.params, btpParams, btpKey); err != nil {
				panic(err)
			}
		} else {
			if cont.btp, err = ckks.NewBootstrapper_mod(cont.params, btpParams, btpKey); err != nil {
				panic(err)
			}
		}
		fmt.Printf("Done in %s \n", time.Since(start))
	}

	return &cont
}

func main() {

	// testConv_noBoot(7, 8, 8, true)

	// testImageNet_BL()

	iter, _ := strconv.Atoi(os.Args[1])
	testResNet_in_BL(iter)
	// testResNet_in(0)

	// testConv_BNRelu_BL("TransConv", true)
	// testConv_noBoot_BL("TransConv", true)
	// testResNet_BL()
	// testReduceMean_BL()

	// basic()

	// testBRrot()
	// testConv_noBoot("Conv", true)
	// testConv_BNRelu("StrConv", true)
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
	prt_size := 32
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
			if (show == 0) || (((i <= show) || (i+show > in_wid)) && ((j <= show) || (j+show > in_wid))) {
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

// only 10, (1,1) element in all batches (1,0,0,0,0,0,0,0,2,0,0,0,0,0,...)
func prt_mat_one_BL(vec []complex128, max_bat int) (out []float64) {
	mat_size := len(vec) / max_bat
	out = make([]float64, 10)

	for i := range out {
		out[i] = real(vec[i*mat_size*8])
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

func prep_Input(input []float64, raw_in_wid, in_wid, N, norm int, trans, printResult bool) (out []float64) {
	out = make([]float64, N)
	batch := N / (in_wid * in_wid)
	k := 0

	if trans {
		for i := 0; i < in_wid/2; i++ {
			for j := 0; j < in_wid/2; j++ {
				for b := 0; b < batch/norm; b++ {
					if (i < raw_in_wid) && (j < raw_in_wid) {
						out[(2*i+1)*in_wid*batch+(2*j+1)*batch+b*norm] = input[k]
						k++
					}
				}
			}
		}
	} else {
		for i := 0; i < in_wid; i++ {
			for j := 0; j < in_wid; j++ {
				for b := 0; b < batch/norm; b++ {
					if (i < raw_in_wid) && (j < raw_in_wid) {
						out[i*in_wid*batch+j*batch+b*norm] = input[k]
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

func removeDuplicateInt(intSlice []int) []int {
	allKeys := make(map[int]bool)
	list := []int{}
	for _, item := range intSlice {
		if _, value := allKeys[item]; !value {
			allKeys[item] = true
			list = append(list, item)
		}
	}
	return list
}
