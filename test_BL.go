package main

import (
	"fmt"
	"math"
	"time"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/rlwe"
)

// BaseLine Conv without boot, Assume full batch with Po2 in_wid & N
// Normal Conv without output modification (e.g., trimming or expanding)
// Input does not need padding
func testConv_noBoot_BL(in_kind string, printResult bool) {
	if (in_kind != "TransConv") && (in_kind != "Conv") && (in_kind != "StrConv") {
		panic("Wrong in_kind!")
	}
	in_batch := 8
	raw_in_wid := 8 // = in_wid
	ker_wid := 5

	in_size := raw_in_wid * raw_in_wid
	slots := in_batch * in_size
	log_slots := 0
	for ; (1 << log_slots) < slots; log_slots++ {
	}
	out_batch := in_batch
	if in_kind == "TransConv" {
		out_batch = in_batch / 4
	}

	kp_wid := 0
	strides := false
	trans := false
	switch in_kind {
	case "StrConv":
		strides = true
	case "TransConv":
		trans = true
	}
	kind := "BL_" + in_kind

	input := make([]float64, raw_in_wid*raw_in_wid*in_batch)
	ker_in := make([]float64, in_batch*out_batch*ker_wid*ker_wid)
	bn_a := make([]float64, out_batch)
	bn_b := make([]float64, out_batch)
	for i := range input {
		input[i] = 1.0 * float64(i) / float64(len(input))
	}
	for i := range ker_in {
		ker_in[i] = 1.0 - 1.0*float64(i)/float64(len(ker_in))
	}
	for i := range bn_a {
		bn_a[i] = 1.0
		bn_b[i] = 0.0
	}

	// generate Context: params, Keys, rotations, general plaintexts
	cont := newContext(log_slots+1, ker_wid, []int{raw_in_wid}, []int{kp_wid}, false, kind)
	fmt.Println("vec size: log2 = ", cont.logN)
	fmt.Println("raw input width: ", raw_in_wid)
	fmt.Println("kernel width: ", ker_wid)
	fmt.Println("num batches in & out: ", in_batch, ", ", out_batch)

	// input encryption
	fmt.Println()
	fmt.Println("===============  ENCRYPTION  ===============")
	fmt.Println()
	input_rs := reshape_input_BL(input, raw_in_wid)
	if printResult {
		prt_mat_BL(input_rs, in_batch, 0)
	}
	start = time.Now()
	ct_input := cont.encryptor.EncryptNew(cont.encoder.EncodeAtLvlNew(cont.ECD_LV, input_rs, log_slots))
	fmt.Printf("Encryption done in %s \n", time.Since(start))

	var ct_input_rot, ct_result *ckks.Ciphertext
	rot_time := time.Duration(0)
	eval_time := time.Duration(0)
	if trans {
		for pos := 0; pos < 4; pos++ {
			start = time.Now()
			ct_input_rot = evalRot_BL(cont, ct_input, raw_in_wid, pos, trans)
			rot_time += time.Since(start)
			start = time.Now()
			if pos == 0 {
				ct_result = evalConv_BN_BL(cont, ct_input_rot, ker_in, bn_a, bn_b, 2*raw_in_wid, ker_wid, in_batch, out_batch, pos, 1, trans, printResult)
			} else {
				ct_tmp := evalConv_BN_BL(cont, ct_input_rot, ker_in, bn_a, bn_b, 2*raw_in_wid, ker_wid, in_batch, out_batch, pos, 1, trans, printResult)
				cont.evaluator.Add(ct_result, ct_tmp, ct_result)
			}
			eval_time += time.Since(start)
		}
		fmt.Printf("Rotation (for transConv) Done in %s \n", rot_time)
		fmt.Printf("EvalConv total (for transConv) Done in %s \n", eval_time)

	} else {
		ct_result = evalConv_BN_BL(cont, ct_input, ker_in, bn_a, bn_b, raw_in_wid, ker_wid, in_batch, out_batch, 0, 1, trans, printResult)
		if strides {
			start = time.Now()
			ct_result = evalRot_BL(cont, ct_result, raw_in_wid, 0, trans)
			fmt.Printf("Rotation (for strided Conv) Done in %s \n", time.Since(start))
		}
	}

	fmt.Println()
	fmt.Println("===============  DECRYPTION  ===============")
	fmt.Println()
	start = time.Now()
	vals_tmp := cont.encoder.Decode(cont.decryptor.DecryptNew(ct_result), log_slots)
	fmt.Printf("Decryption Done in %s \n", time.Since(start))

	f_out_batch := out_batch
	if strides {
		f_out_batch = out_batch * 4
	}

	if printResult {
		fmt.Print("Result: \n")
		prt_mat_BL(vals_tmp, f_out_batch, 0)
	}
}

// BaseLine Conv without boot, Assume full batch with Po2 in_wid & N
// Normal Conv without output modification (e.g., trimming or expanding)
// Input does not need padding
func testConv_BNRelu_BL(in_kind string, printResult bool) {
	in_batch := 8
	raw_in_wid := 8 // = in_wid
	ker_wid := 5
	alpha := 0.0

	in_size := raw_in_wid * raw_in_wid
	slots := in_batch * in_size
	log_slots := 0
	for ; (1 << log_slots) < slots; log_slots++ {
	}
	out_batch := in_batch
	if in_kind == "TransConv" {
		out_batch = in_batch / 4
	}

	kp_wid := 0
	strides := false
	trans := false
	switch in_kind {
	case "StrConv":
		strides = true
	case "TransConv":
		trans = true
	}
	kind := "BL_" + in_kind

	input := make([]float64, raw_in_wid*raw_in_wid*in_batch)
	ker_in := make([]float64, in_batch*out_batch*ker_wid*ker_wid)
	bn_a := make([]float64, out_batch)
	bn_b := make([]float64, out_batch)
	for i := range input {
		input[i] = 1.0 * float64(i) / float64(len(input))
	}
	for i := range ker_in {
		ker_in[i] = 1.0 - 1.0*float64(i)/float64(len(ker_in))
	}
	for i := range bn_a {
		bn_a[i] = 1.0
		bn_b[i] = 0.0
	}

	// generate Context: params, Keys, rotations, general plaintexts
	cont := newContext(log_slots+1, ker_wid, []int{raw_in_wid}, []int{kp_wid}, true, kind)
	fmt.Println("vec size: log2 = ", cont.logN)
	fmt.Println("raw input width: ", raw_in_wid)
	fmt.Println("kernel width: ", ker_wid)
	fmt.Println("num batches in & out: ", in_batch, ", ", out_batch)

	// input encryption
	fmt.Println()
	fmt.Println("===============  ENCRYPTION  ===============")
	fmt.Println()
	input_rs := reshape_input_BL(input, raw_in_wid)
	if printResult {
		prt_mat_BL(input_rs, in_batch, 0)
	}
	start = time.Now()
	ct_input := cont.encryptor.EncryptNew(cont.encoder.EncodeAtLvlNew(cont.ECD_LV, input_rs, log_slots))
	fmt.Printf("Encryption done in %s \n", time.Since(start))

	ct_result := evalConv_BNRelu_BL(cont, ct_input, ker_in, bn_a, bn_b, alpha, raw_in_wid, ker_wid, in_batch, out_batch, 1, strides, trans, printResult)

	fmt.Println()
	fmt.Println("===============  DECRYPTION  ===============")
	fmt.Println()
	start = time.Now()
	vals_tmp := cont.encoder.Decode(cont.decryptor.DecryptNew(ct_result), cont.logN-1)
	fmt.Printf("Decryption (Relu) Done in %s \n", time.Since(start))
	fmt.Println("after relu: LV = ", ct_result.Level(), " Scale = 2^", math.Log2(ct_result.Scale))
	f_out_batch := out_batch
	if strides {
		f_out_batch = out_batch * 4
	}

	if printResult {
		fmt.Print("Result: \n")
		prt_mat_BL(vals_tmp, f_out_batch, 0)
	}
}

// alomst the same as ResNet, but use smaller batches for testing
func testResNet_BL() {
	logN := 12
	in_wids := []int{32, 16, 8}  // = raw_in_wids = same as python
	real_batch := []int{1, 2, 4} // same as python
	py_bn_a := []float64{0.3, 0.3, 0.1}
	ker_wid := 3
	kp_wids := make([]int, len(in_wids)) // NOT used in BL
	copy(kp_wids, in_wids)
	cont := newContext(logN, ker_wid, in_wids, kp_wids, true, "BL_Resnet")

	ker_size := ker_wid * ker_wid
	max_batch := make([]int, len(real_batch)) // the max batch
	for i := range max_batch {
		max_batch[i] = cont.N / (2 * in_wids[i] * in_wids[i])
	}

	alpha := 0.0 // 0.3 => leakyrelu
	input := make([]float64, cont.N/2)
	k := 0.0
	for i := 0; i < in_wids[0]; i++ {
		for j := 0; j < in_wids[0]; j++ {
			for b := 0; b < max_batch[0]; b++ {
				if (i < in_wids[0]) && (j < in_wids[0]) && (b < real_batch[0]) {
					input[i*in_wids[0]*max_batch[0]+j*max_batch[0]+b] = k
					k += (10.0 / float64(real_batch[0]*(in_wids[0])*(in_wids[0])))
				}
			}
		}
	}
	fmt.Println("Input: ")
	prt_mat(input, max_batch[0], 3)
	ker_in := make([]float64, real_batch[0]*real_batch[0]*ker_size)
	for i := range ker_in {
		ker_in[i] = 0.5 * float64(i) / float64(len(ker_in))
	}
	ker_in12 := make([]float64, real_batch[0]*real_batch[1]*ker_size)
	for i := range ker_in12 {
		ker_in12[i] = 0.5 * float64(i) / float64(len(ker_in12))
	}
	ker_in2 := make([]float64, real_batch[1]*real_batch[1]*ker_size)
	for i := range ker_in2 {
		ker_in2[i] = 0.5 * float64(i) / float64(len(ker_in2))
	}
	ker_in23 := make([]float64, real_batch[1]*real_batch[2]*ker_size)
	for i := range ker_in23 {
		// ker_in23[i] = 1.0
		ker_in23[i] = 0.5 * float64(i) / float64(len(ker_in23))
	}
	ker_in23_0 := make([]float64, len(ker_in23)/2)
	ker_in23_1 := make([]float64, len(ker_in23)/2)
	// ker_in23_0 part outputs (0,2,4,6,... ) outbatches
	// ker_in23_1 part outputs (1,3,5,7,... ) outbatches
	for k := 0; k < ker_size; k++ {
		for i := 0; i < real_batch[1]; i++ {
			for j := 0; j < real_batch[2]/2; j++ {
				ker_in23_0[k*real_batch[1]*real_batch[2]/2+(i*real_batch[2]/2+j)] = ker_in23[k*real_batch[1]*real_batch[2]+(i*real_batch[2]+2*j)]   // [i][2*j]
				ker_in23_1[k*real_batch[1]*real_batch[2]/2+(i*real_batch[2]/2+j)] = ker_in23[k*real_batch[1]*real_batch[2]+(i*real_batch[2]+2*j+1)] // [i][2*j+1]
			}
		}
	}

	ker_in3 := make([]float64, real_batch[2]*real_batch[2]*ker_size)
	for i := range ker_in3 {
		ker_in3[i] = 0.5 * float64(i) / float64(len(ker_in3))
	}
	bn_a := make([]float64, real_batch[0])
	bn_b := make([]float64, real_batch[0])
	for i := range bn_a {
		bn_a[i] = py_bn_a[0] // * float64(i) / float64(batch)
		bn_b[i] = 0.0        //0.1 * float64(i) // float64(real_batch[0])
	}
	bn_a2 := make([]float64, real_batch[1])
	bn_b2 := make([]float64, real_batch[1])
	for i := range bn_a2 {
		bn_a2[i] = py_bn_a[1] // * float64(i) / float64(batch)
		bn_b2[i] = 0.0        //0.1 * float64(i)
	}
	bn_a3 := make([]float64, real_batch[2])
	bn_b3 := make([]float64, real_batch[2])
	for i := range bn_a3 {
		bn_a3[i] = py_bn_a[2] // * float64(i) / float64(batch)
		bn_b3[i] = 0.0        //0.1 * float64(i)
	}
	bn_b3_0 := make([]float64, real_batch[2]/2)
	bn_b3_1 := make([]float64, real_batch[2]/2)
	for i := range bn_b3_0 {
		bn_b3_0[i] = bn_b3[2*i]
		bn_b3_1[i] = bn_b3[2*i+1]
	}

	fmt.Println("vec size: ", cont.N)
	fmt.Println("input width: ", in_wids)
	fmt.Println("kernel width: ", ker_wid)
	fmt.Println("num batches: ", real_batch)
	fmt.Println("Input matrix: ")
	prt_vec(input)

	// input encryption
	fmt.Println()
	fmt.Println("===============  ENCRYPTION  ===============")
	fmt.Println()
	input_rs := reshape_input_BL(input, in_wids[0])
	// prt_mat_BL(input_rs, max_batch[0], 0)
	start = time.Now()
	ct_input := cont.encryptor.EncryptNew(cont.encoder.EncodeAtLvlNew(cont.ECD_LV, input_rs, cont.logN-1))
	fmt.Printf("Encryption done in %s \n", time.Since(start))

	timings := make([]float64, 6)
	begin_start := time.Now()
	new_start := time.Now()

	// ResNet Block 1
	num_blc1 := 2
	ct_layer := make([]*ckks.Ciphertext, num_blc1+1)
	ct_layer[0] = ct_input
	prt_result := false
	for i := 1; i <= num_blc1; i++ {
		if i == num_blc1 {
			prt_result = true
		}
		ct_layer[i] = evalConv_BNRelu_BL(cont, ct_layer[i-1], ker_in, bn_a, bn_b, alpha, in_wids[0], ker_wid, real_batch[0], real_batch[0], 1, false, false, prt_result)
		fmt.Println("Block1, Layer ", i, "done!")
	}
	fmt.Println("done.")
	timings[0] = time.Since(new_start).Seconds()
	new_start = time.Now()
	ct_result := evalConv_BNRelu_BL(cont, ct_layer[num_blc1], ker_in12, bn_a2, bn_b2, alpha, in_wids[0], ker_wid, real_batch[0], real_batch[1], 1, true, false, prt_result)
	timings[1] = time.Since(new_start).Seconds()
	fmt.Println("Block1 to 2 done!")

	// ResNet Block 2
	num_blc2 := 2
	ct_layer2 := make([]*ckks.Ciphertext, num_blc2+1)
	ct_layer2[0] = ct_result
	prt_result = false
	for i := 1; i <= num_blc2; i++ {
		if i == num_blc2 {
			prt_result = true
		}
		ct_layer2[i] = evalConv_BNRelu_BL(cont, ct_layer2[i-1], ker_in2, bn_a2, bn_b2, alpha, in_wids[1], ker_wid, real_batch[1], real_batch[1], 4, false, false, prt_result)

		fmt.Println("Block2, Layer ", i, "done!")
	}
	timings[2] = time.Since(new_start).Seconds()
	new_start = time.Now()
	// ct_result = evalConv_BNRelu_BL(cont, ct_layer2[num_blc2], ker_in23, bn_a3, bn_b3, alpha, in_wids[1], ker_wid, real_batch[1], real_batch[2], 4, true, false, prt_result)

	ct_result1 := evalConv_BN_BL(cont, ct_layer2[num_blc2], ker_in23_0, bn_a3, bn_b3_0, in_wids[1], ker_wid, real_batch[1], real_batch[2]/2, 0, 4, false, prt_result)
	ct_result2 := evalConv_BN_BL(cont, ct_layer2[num_blc2], ker_in23_1, bn_a3, bn_b3_1, in_wids[1], ker_wid, real_batch[1], real_batch[2]/2, 0, 4, false, prt_result)
	ct_result1 = evalRot_BL(cont, ct_result1, in_wids[1], 0, false) // ct_result2 = evalRot_BL(cont, ct_result2, in_wids[1], 0, false)
	ct_result2 = cont.evaluator.RotateNew(evalRot_BL(cont, ct_result2, in_wids[1], 0, false), -in_wids[1]*in_wids[1]*2)
	ct_result = cont.evaluator.AddNew(ct_result1, ct_result2)

	ct_result.Scale = ct_result.Scale * math.Pow(2, pow)
	vals_preB := cont.encoder.Decode(cont.decryptor.DecryptNew(ct_result), cont.logN-1)
	fmt.Println("\n ========= Bootstrapping... (original) ========= ")
	start_boot := time.Now()
	fmt.Println("initial (before boot): LV = ", ct_result.Level(), " Scale = ", math.Log2(ct_result.Scale))

	ct_boot := cont.btp.Bootstrapp(ct_result)
	fmt.Printf("Done in %s \n", time.Since(start_boot))
	fmt.Println("after Boot: LV = ", ct_boot.Level(), " Scale = ", math.Log2(ct_boot.Scale))

	// Only for checking the correctness (for Boot)
	vals_postB := printDebug(cont.params, ct_boot, vals_preB, cont.decryptor, cont.encoder)
	vals_relu := make([]complex128, len(vals_postB))
	for i, elt := range vals_postB {
		vals_relu[i] = complex((math.Max(0, real(elt))+math.Min(0, real(elt)*alpha))*math.Pow(2, pow), 0)
	}

	start = time.Now()
	pl_scale := ckks.NewPlaintext(cont.params, ct_boot.Level(), math.Pow(2, 30)*float64(cont.params.Q()[14])*float64(cont.params.Q()[13])/ct_boot.Scale)
	val_scale := make([]complex128, cont.N/2)
	for i := range val_scale {
		val_scale[i] = complex(1.0, 0) // val_scale[i] = complex(1.0/math.Pow(2, pow), 0)
	}
	cont.encoder.EncodeNTT(pl_scale, val_scale, cont.logN-1)
	cont.evaluator.Mul(ct_boot, pl_scale, ct_boot)
	cont.evaluator.Rescale(ct_boot, cont.params.Scale(), ct_boot)

	fmt.Println("after Rescale: LV = ", ct_boot.Level(), " Scale = 2^", math.Log2(ct_boot.Scale))
	ct_result = evalReLU(cont.params, cont.evaluator, ct_boot, alpha)
	cont.evaluator.MulByPow2(ct_result, pow, ct_result)
	cont.evaluator.SetScale(ct_result, cont.params.Scale())
	fmt.Printf("Relu Done in %s \n", time.Since(start))
	printDebug(cont.params, ct_result, vals_relu, cont.decryptor, cont.encoder)

	if prt_result {
		vals_tmp := cont.encoder.Decode(cont.decryptor.DecryptNew(ct_result), cont.logN-1)
		fmt.Print("Result: \n")
		prt_mat_BL(vals_tmp, (cont.N / (in_wids[1] * in_wids[1] / 2)), 3)
	}

	fmt.Println("Block2 to 3 done!")
	timings[3] = time.Since(new_start).Seconds()
	new_start = time.Now()

	// ResNet Block 3
	num_blc3 := 2
	ct_layer3 := make([]*ckks.Ciphertext, num_blc3+1)
	ct_layer3[0] = ct_result
	prt_result = false
	for i := 1; i <= num_blc3; i++ {
		if i == num_blc3 {
			prt_result = true
		}
		ct_layer3[i] = evalConv_BNRelu_BL(cont, ct_layer3[i-1], ker_in3, bn_a3, bn_b3, alpha, in_wids[2], ker_wid, real_batch[2], real_batch[2], 8, false, false, prt_result)
		fmt.Println("Block3, Layer ", i, "done!")
	}
	timings[4] = time.Since(new_start).Seconds()

	fmt.Println()
	fmt.Println("===============  DECRYPTION  ===============")
	fmt.Println()
	start = time.Now()
	vals_tmp := cont.encoder.Decode(cont.decryptor.DecryptNew(ct_layer3[num_blc3]), cont.logN-1)
	fmt.Printf("Decryption Done in %s \n", time.Since(start))
	if prt_result {
		fmt.Print("Result: \n")
		prt_mat_BL(vals_tmp, max_batch[2], 3)
	}

	// // for final reduce_mean & FC

	fmt.Println("Blc1: ", timings[0], " sec")
	fmt.Println("Blc1->2: ", timings[1], " sec")
	fmt.Println("Blc2: ", timings[2], " sec")
	fmt.Println("Blc2->3: ", timings[3], " sec")
	fmt.Println("Blc3: ", timings[4], " sec")
	fmt.Println("Final (reduce_mean & FC): ", timings[5], " sec")
	fmt.Printf("Total done in %s \n", time.Since(begin_start))
}

func testResNet_in_BL(iter int) {

}

func testReduceMean_BL() {
	logN := 16
	ker_wid := 3
	in_wids := []int{32, 16, 8}
	real_batch := []int{16, 32, 64}
	kp_wids := make([]int, len(in_wids)) // NOT used in BL
	copy(kp_wids, in_wids)

	max_batch := make([]int, len(real_batch)) // the max batch
	for i := range max_batch {
		max_batch[i] = (1 << (logN - 1)) / (in_wids[i] * in_wids[i])
	}
	fmt.Println(max_batch)

	input := make([]float64, (1 << (logN - 1)))
	k := 0.0
	for i := 0; i < in_wids[2]; i++ {
		for j := 0; j < in_wids[2]; j++ {
			for b := 0; b < max_batch[2]; b++ {
				if b < real_batch[2] {
					input[i*in_wids[2]*max_batch[2]+j*max_batch[2]+b] = k
					k += 0.1
					if k >= 2.7 {
						k = 0
					}
					// k += (1.0 / float64(real_batch[2]*in_wids[2]*in_wids[2]))
				}
			}
		}
	}
	fmt.Println("Input: ")
	prt_mat(input, max_batch[2], 0)
	ker_inf := make([]float64, real_batch[2]*10)
	for i := range ker_inf {
		ker_inf[i] = float64(i%27) * 0.1 // 1.0 * float64(i) / 640.0 // * float64(i) / float64(len(ker_inf))
	}
	fmt.Println("ker: ", ker_inf)
	bias := make([]float64, 10)
	for i := range bias {
		bias[i] = 1.0 * float64(i)
	}

	cont := newContext(logN, ker_wid, in_wids, kp_wids, false, "BL_Resnet")
	// input encryption
	fmt.Println()
	fmt.Println("===============  ENCRYPTION  ===============")
	fmt.Println()
	input_rs := reshape_input_BL(input, in_wids[2])
	prt_mat_BL(input_rs, max_batch[2], 0)
	start = time.Now()
	ct_input := cont.encryptor.EncryptNew(cont.encoder.EncodeAtLvlNew(cont.ECD_LV, input_rs, cont.logN-1))
	fmt.Printf("Encryption done in %s \n", time.Since(start))

	start = time.Now()
	ct_result := evalRMFC_BL(cont, ct_input, ker_inf, bias, true)
	fmt.Printf("Eval done in %s \n", time.Since(start))

	fmt.Println()
	fmt.Println("===============  DECRYPTION  ===============")
	fmt.Println()
	start = time.Now()
	vals_tmp := cont.encoder.Decode(cont.decryptor.DecryptNew(ct_result), cont.logN-1)
	fmt.Printf("Decryption Done in %s \n", time.Since(start))
	fmt.Print("Result: \n")
	prt_mat_BL(vals_tmp, max_batch[2], 0)

}

func basic() {
	logN := 5
	N := (1 << logN)

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
	start = time.Now()

	kgen := ckks.NewKeyGenerator(params)
	sk := kgen.GenSecretKey()
	encryptor := ckks.NewEncryptor(params, sk)
	decryptor := ckks.NewDecryptor(params, sk)
	encoder := ckks.NewEncoder(params)
	rlk := kgen.GenRelinearizationKey(sk, 2)
	evaluator := ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk})

	fmt.Printf("Done in %s \n", time.Since(start))

	fmt.Println()
	fmt.Printf("CKKS parameters: logN = %d, logSlots = %d, logQP = %d, levels = %d, scale= %f, sigma = %f \n",
		params.LogN(), params.LogSlots(), params.LogQP(), params.MaxLevel()+1, params.Scale(), params.Sigma())

	min_slot := 1 << 1 // assume full slot = 1 << 15
	tmp := make([]complex128, min_slot)
	for j := 0; j < min_slot; j++ {
		tmp[j] = complex(float64(j), 0.0)
	}
	ptxt := encoder.EncodeNTTNew(tmp, 1)

	input := make([]complex128, N/2)
	for j := 0; j < N/2; j++ {
		input[j] = complex(float64(j), 0.0)
	}
	ctxt := encryptor.EncryptNew(encoder.EncodeNew(input, logN-1))

	evaluator.Mul(ctxt, ptxt, ctxt)

	res := encoder.Decode(decryptor.DecryptNew(ctxt), logN-1)
	for i := range res {
		fmt.Println(real(res[i]))
	}

}
