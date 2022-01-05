package main

import (
	"fmt"
	"math"
	"strconv"
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
	in_batch := 4
	raw_in_wid := 7 // = in_wid
	in_wid := 8
	ker_wid := 3
	pad := in_wid - raw_in_wid

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
	pad_input := make([]float64, in_wid*in_wid*in_batch)
	for i := 0; i < raw_in_wid; i++ {
		for j := 0; j < raw_in_wid; j++ {
			for b := 0; b < in_batch; b++ {
				pad_input[b+j*in_batch+i*in_batch*in_wid] = input[b+j*in_batch+i*in_batch*raw_in_wid]
			}
		}
	}

	for i := range ker_in {
		ker_in[i] = 1.0 * float64(i) / float64(len(ker_in))
	}
	for i := range bn_a {
		bn_a[i] = 1.0
		bn_b[i] = 0.0
	}

	// generate Context: params, Keys, rotations, general plaintexts
	cont := newContext(log_slots+1, ker_wid, []int{in_wid}, []int{kp_wid}, false, kind)
	fmt.Println("vec size: log2 = ", cont.logN)
	fmt.Println("raw input width: ", raw_in_wid)
	fmt.Println("kernel width: ", ker_wid)
	fmt.Println("num batches in & out: ", in_batch, ", ", out_batch)

	// input encryption
	fmt.Println()
	fmt.Println("===============  ENCRYPTION  ===============")
	fmt.Println()
	input_rs := reshape_input_BL(pad_input, in_wid)
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
			ct_input_rot = evalRot_BL(cont, ct_input, in_wid, pos, trans)
			rot_time += time.Since(start)
			start = time.Now()
			if pos == 0 {
				ct_result = evalConv_BN_BL(cont, ct_input_rot, ker_in, bn_a, bn_b, 2*in_wid, ker_wid, in_batch, out_batch, pos, 1, pad, trans, printResult)
			} else {
				ct_tmp := evalConv_BN_BL(cont, ct_input_rot, ker_in, bn_a, bn_b, 2*in_wid, ker_wid, in_batch, out_batch, pos, 1, pad, trans, printResult)
				cont.evaluator.Add(ct_result, ct_tmp, ct_result)
			}
			eval_time += time.Since(start)
		}
		fmt.Printf("Rotation (for transConv) Done in %s \n", rot_time)
		fmt.Printf("EvalConv total (for transConv) Done in %s \n", eval_time)

	} else {
		ct_result = evalConv_BN_BL(cont, ct_input, ker_in, bn_a, bn_b, in_wid, ker_wid, in_batch, out_batch, 0, 1, pad, trans, printResult)
		if strides {
			start = time.Now()
			ct_result = evalRot_BL(cont, ct_result, in_wid, 0, trans)
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
	in_batch := 4
	raw_in_wid := 14
	in_wid := 16
	pad := in_wid - raw_in_wid
	ker_wid := 3
	alpha := 0.0

	in_size := in_wid * in_wid
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
		ker_in[i] = 1.0 * float64(i) / float64(len(ker_in))
	}
	for i := range bn_a {
		bn_a[i] = 1.0
		bn_b[i] = 0.0
	}
	pad_input := make([]float64, in_wid*in_wid*in_batch)
	for i := 0; i < raw_in_wid; i++ {
		for j := 0; j < raw_in_wid; j++ {
			for b := 0; b < in_batch; b++ {
				pad_input[b+j*in_batch+i*in_batch*in_wid] = input[b+j*in_batch+i*in_batch*raw_in_wid]
			}
		}
	}

	// generate Context: params, Keys, rotations, general plaintexts
	cont := newContext(log_slots+1, ker_wid, []int{in_wid}, []int{kp_wid}, true, kind)
	fmt.Println("vec size: log2 = ", cont.logN)
	fmt.Println("raw input width: ", raw_in_wid)
	fmt.Println("kernel width: ", ker_wid)
	fmt.Println("num batches in & out: ", in_batch, ", ", out_batch)

	// input encryption
	fmt.Println()
	fmt.Println("===============  ENCRYPTION  ===============")
	fmt.Println()
	input_rs := reshape_input_BL(pad_input, in_wid)
	if printResult {
		prt_mat_BL(input_rs, in_batch, 0)
	}
	start = time.Now()
	ct_input := cont.encryptor.EncryptNew(cont.encoder.EncodeAtLvlNew(cont.ECD_LV, input_rs, log_slots))
	fmt.Printf("Encryption done in %s \n", time.Since(start))

	ct_result := evalConv_BNRelu_BL(cont, ct_input, ker_in, bn_a, bn_b, alpha, in_wid, ker_wid, in_batch, out_batch, 1, pad, strides, trans, printResult)

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
	num_blc1 := 3 // 3 // 5 // 7
	num_blc2 := 1 // 1 // 3 // 5
	num_blc3 := 1 // 1 // 3 // 5
	pad := 0
	logN := 14
	in_wids := []int{32, 16, 8}   // = raw_in_wids = same as python
	real_batch := []int{4, 8, 16} // same as python
	py_bn_a := []float64{0.2, 0.2, 0.1}
	ker_wid := 7
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
				if (i < in_wids[0]) && (j < in_wids[0]) && (b < 3) {
					input[i*in_wids[0]*max_batch[0]+j*max_batch[0]+b] = k
					k += (5.0 / float64(3*(in_wids[0])*(in_wids[0])))
				}
			}
		}
	}
	fmt.Println("Input: ")
	prt_mat(input, max_batch[0], 3)
	ker_in0 := make([]float64, 3*real_batch[0]*ker_size)
	for i := range ker_in0 {
		ker_in0[i] = 0.25 * float64(i) / float64(len(ker_in0))
	}
	ker_in := make([]float64, real_batch[0]*real_batch[0]*ker_size)
	for i := range ker_in {
		ker_in[i] = 0.25 * float64(i) / float64(len(ker_in))
	}
	ker_in12 := make([]float64, real_batch[0]*real_batch[1]*ker_size)
	for i := range ker_in12 {
		ker_in12[i] = 0.25 * float64(i) / float64(len(ker_in12))
	}
	ker_in2 := make([]float64, real_batch[1]*real_batch[1]*ker_size)
	for i := range ker_in2 {
		ker_in2[i] = 0.25 * float64(i) / float64(len(ker_in2))
	}
	ker_in23 := make([]float64, real_batch[1]*real_batch[2]*ker_size)
	for i := range ker_in23 {
		// ker_in23[i] = 1.0
		ker_in23[i] = 0.25 * float64(i) / float64(len(ker_in23))
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
		ker_in3[i] = 0.25 * float64(i) / float64(len(ker_in3))
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
	bn_a3_0 := make([]float64, real_batch[2]/2)
	bn_a3_1 := make([]float64, real_batch[2]/2)
	bn_b3_0 := make([]float64, real_batch[2]/2)
	bn_b3_1 := make([]float64, real_batch[2]/2)
	for i := range bn_b3_0 {
		bn_a3_0[i] = bn_a3[2*i]
		bn_a3_1[i] = bn_a3[2*i+1]
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

	ct_layer := ct_input
	prt_result := false
	for i := 1; i <= num_blc1; i++ {
		if i == num_blc1 {
			prt_result = true
		}
		if i == 1 {
			ct_layer = evalConv_BNRelu_BL(cont, ct_layer, ker_in0, bn_a, bn_b, alpha, in_wids[0], ker_wid, 3, real_batch[0], 1, pad, false, false, prt_result)
		} else {
			ct_layer = evalConv_BNRelu_BL(cont, ct_layer, ker_in, bn_a, bn_b, alpha, in_wids[0], ker_wid, real_batch[0], real_batch[0], 1, pad, false, false, prt_result)
		}
		fmt.Println("Block1, Layer ", i, "done!")
	}
	fmt.Println("done.")
	timings[0] = time.Since(new_start).Seconds()
	new_start = time.Now()
	ct_result := evalConv_BNRelu_BL(cont, ct_layer, ker_in12, bn_a2, bn_b2, alpha, in_wids[0], ker_wid, real_batch[0], real_batch[1], 1, pad, true, false, prt_result)
	timings[1] = time.Since(new_start).Seconds()
	fmt.Println("Block1 to 2 done!")

	// ResNet Block 2
	ct_layer2 := ct_result
	prt_result = false
	for i := 1; i <= num_blc2; i++ {
		if i == num_blc2 {
			prt_result = true
		}
		ct_layer2 = evalConv_BNRelu_BL(cont, ct_layer2, ker_in2, bn_a2, bn_b2, alpha, in_wids[1], ker_wid, real_batch[1], real_batch[1], 4, pad, false, false, prt_result)

		fmt.Println("Block2, Layer ", i, "done!")
	}
	timings[2] = time.Since(new_start).Seconds()
	new_start = time.Now()

	ct_result1 := evalConv_BN_BL(cont, ct_layer2, ker_in23_0, bn_a3, bn_b3_0, in_wids[1], ker_wid, real_batch[1], real_batch[2]/2, 0, 4, pad, false, prt_result)
	ct_result2 := evalConv_BN_BL(cont, ct_layer2, ker_in23_1, bn_a3, bn_b3_1, in_wids[1], ker_wid, real_batch[1], real_batch[2]/2, 0, 4, pad, false, prt_result)
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
	ct_layer3 := ct_result
	prt_result = false
	for i := 1; i <= num_blc3; i++ {
		if i == num_blc3 {
			prt_result = true
		}
		ct_layer3 = evalConv_BNRelu_BL(cont, ct_layer3, ker_in3, bn_a3, bn_b3, alpha, in_wids[2], ker_wid, real_batch[2], real_batch[2], 8, pad, false, false, prt_result)
		fmt.Println("Block3, Layer ", i, "done!")
	}
	timings[4] = time.Since(new_start).Seconds()

	// no RMFC

	fmt.Println()
	fmt.Println("===============  DECRYPTION  ===============")
	fmt.Println()
	start = time.Now()
	vals_tmp := cont.encoder.Decode(cont.decryptor.DecryptNew(ct_layer3), cont.logN-1)
	fmt.Printf("Decryption Done in %s \n", time.Since(start))
	fmt.Println("result: ")
	prt_mat_BL(vals_tmp, max_batch[2], 3)

	fmt.Println("Blc1: ", timings[0], " sec")
	fmt.Println("Blc1->2: ", timings[1], " sec")
	fmt.Println("Blc2: ", timings[2], " sec")
	fmt.Println("Blc2->3: ", timings[3], " sec")
	fmt.Println("Blc3: ", timings[4], " sec")
	fmt.Println("Final (reduce_mean & FC): ", timings[5], " sec")
	fmt.Printf("Total done in %s \n", time.Since(begin_start))
}

func testResNet_in_BL(iter int) {
	weight_dir := "weight_ker7_h5/"
	ker_name := "ker7"
	num_blc1 := 3 // 3 // 5 // 7
	num_blc2 := 1 // 1 // 3 // 5
	num_blc3 := 1 // 1 // 3 // 5
	logN := 16
	pad := 0
	in_wids := []int{32, 16, 8}     // = raw_in_wids = same as python
	real_batch := []int{16, 32, 64} // same as python
	ker_wid := 7
	kp_wids := make([]int, len(in_wids)) // NOT used in BL
	copy(kp_wids, in_wids)
	cont := newContext(logN, ker_wid, in_wids, kp_wids, true, "BL_Resnet")

	ker_size := ker_wid * ker_wid
	max_batch := make([]int, len(real_batch)) // the max batch
	for i := range max_batch {
		max_batch[i] = cont.N / (2 * in_wids[i] * in_wids[i])
	}

	image := readTxt("test_data/test_image_"+strconv.Itoa(iter)+".csv", 32*32*3)

	alpha := 0.0 // 0.3 => leakyrelu
	input := make([]float64, cont.N/2)
	k := 0
	for i := 0; i < in_wids[0]; i++ {
		for j := 0; j < in_wids[0]; j++ {
			for b := 0; b < max_batch[0]; b++ {
				if (i < in_wids[0]) && (j < in_wids[0]) && (b < 3) {
					input[i*in_wids[0]*max_batch[0]+j*max_batch[0]+b] = image[k]
					k++
				}
			}
		}
	}
	fmt.Println("Input: ")
	prt_mat(input, max_batch[0], 3)

	fmt.Println("vec size: ", cont.N)
	fmt.Println("input width: ", in_wids)
	fmt.Println("kernel width: ", ker_wid)
	fmt.Println("num batches: ", real_batch)

	// input encryption
	fmt.Println()
	fmt.Println("===============  ENCRYPTION  ===============")
	fmt.Println()
	input_rs := reshape_input_BL(input, in_wids[0])
	start = time.Now()
	ct_input := cont.encryptor.EncryptNew(cont.encoder.EncodeAtLvlNew(cont.ECD_LV, input_rs, cont.logN-1))
	fmt.Printf("Encryption done in %s \n", time.Since(start))

	timings := make([]float64, 6)
	begin_start := time.Now()
	new_start := time.Now()

	// ResNet Block 1
	ct_layer := ct_input
	prt_result := true
	for i := 1; i <= num_blc1; i++ {
		if i == num_blc1 {
			prt_result = true
		}
		bn_a := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-a.csv", real_batch[0])
		bn_b := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-b.csv", real_batch[0])
		if i == 1 {
			ker_in := readTxt(weight_dir+"w0-conv.csv", 3*real_batch[0]*ker_size)
			ct_layer = evalConv_BNRelu_BL(cont, ct_layer, ker_in, bn_a, bn_b, alpha, in_wids[0], ker_wid, 3, real_batch[0], 1, pad, false, false, prt_result)
		} else {
			ker_in := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-conv.csv", real_batch[0]*real_batch[0]*ker_size)
			ct_layer = evalConv_BNRelu_BL(cont, ct_layer, ker_in, bn_a, bn_b, alpha, in_wids[0], ker_wid, real_batch[0], real_batch[0], 1, pad, false, false, prt_result)
		}
		fmt.Println("Block1, Layer ", i, "done!")
	}
	fmt.Println("done.")
	timings[0] = time.Since(new_start).Seconds()
	new_start = time.Now()
	ker_in := readTxt(weight_dir+"w"+strconv.Itoa(num_blc1)+"-conv.csv", real_batch[0]*real_batch[1]*ker_size)
	bn_a := readTxt(weight_dir+"w"+strconv.Itoa(num_blc1)+"-a.csv", real_batch[1])
	bn_b := readTxt(weight_dir+"w"+strconv.Itoa(num_blc1)+"-b.csv", real_batch[1])
	ct_result := evalConv_BNRelu_BL(cont, ct_layer, ker_in, bn_a, bn_b, alpha, in_wids[0], ker_wid, real_batch[0], real_batch[1], 1, pad, true, false, prt_result)
	timings[1] = time.Since(new_start).Seconds()
	fmt.Println("Block1 to 2 done!")

	// ResNet Block 2
	ct_layer2 := ct_result
	prt_result = true
	for i := 1; i <= num_blc2; i++ {
		if i == num_blc2 {
			prt_result = true
		}
		bn_a2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blc1+i)+"-a.csv", real_batch[1])
		bn_b2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blc1+i)+"-b.csv", real_batch[1])
		ker_in2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blc1+i)+"-conv.csv", real_batch[1]*real_batch[1]*ker_size)
		ct_layer2 = evalConv_BNRelu_BL(cont, ct_layer2, ker_in2, bn_a2, bn_b2, alpha, in_wids[1], ker_wid, real_batch[1], real_batch[1], 4, pad, false, false, prt_result)

		fmt.Println("Block2, Layer ", i, "done!")
	}
	timings[2] = time.Since(new_start).Seconds()
	new_start = time.Now()

	ker_in23 := readTxt(weight_dir+"w"+strconv.Itoa(num_blc1+num_blc2+1)+"-conv.csv", real_batch[1]*real_batch[2]*ker_size)
	bn_a3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blc1+num_blc2+1)+"-a.csv", real_batch[2])
	bn_b3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blc1+num_blc2+1)+"-b.csv", real_batch[2])
	ker_in23_0 := make([]float64, len(ker_in23)/2) // ker_in23_0 part outputs (0,2,4,6,... ) outbatches
	ker_in23_1 := make([]float64, len(ker_in23)/2) // ker_in23_1 part outputs (1,3,5,7,... ) outbatches
	bn_a3_0 := make([]float64, real_batch[2]/2)
	bn_a3_1 := make([]float64, real_batch[2]/2)
	bn_b3_0 := make([]float64, real_batch[2]/2)
	bn_b3_1 := make([]float64, real_batch[2]/2)
	for i := range bn_b3_0 {
		bn_a3_0[i] = bn_a3[2*i]
		bn_a3_1[i] = bn_a3[2*i+1]
		bn_b3_0[i] = bn_b3[2*i]
		bn_b3_1[i] = bn_b3[2*i+1]
	}

	for k := 0; k < ker_size; k++ {
		for i := 0; i < real_batch[1]; i++ {
			for j := 0; j < real_batch[2]/2; j++ {
				ker_in23_0[k*real_batch[1]*real_batch[2]/2+(i*real_batch[2]/2+j)] = ker_in23[k*real_batch[1]*real_batch[2]+(i*real_batch[2]+2*j)]   // [i][2*j]
				ker_in23_1[k*real_batch[1]*real_batch[2]/2+(i*real_batch[2]/2+j)] = ker_in23[k*real_batch[1]*real_batch[2]+(i*real_batch[2]+2*j+1)] // [i][2*j+1]
			}
		}
	}

	ct_result1 := evalConv_BN_BL(cont, ct_layer2, ker_in23_0, bn_a3_0, bn_b3_0, in_wids[1], ker_wid, real_batch[1], real_batch[2]/2, 0, 4, pad, false, prt_result)
	ct_result2 := evalConv_BN_BL(cont, ct_layer2, ker_in23_1, bn_a3_1, bn_b3_1, in_wids[1], ker_wid, real_batch[1], real_batch[2]/2, 0, 4, pad, false, prt_result)
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
	ct_layer3 := ct_result
	prt_result = true
	for i := 1; i <= num_blc3; i++ {
		if i == num_blc3 {
			prt_result = true
		}
		ker_in3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blc1+num_blc2+i+1)+"-conv.csv", real_batch[2]*real_batch[2]*ker_size)
		bn_a3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blc1+num_blc2+i+1)+"-a.csv", real_batch[2])
		bn_b3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blc1+num_blc2+i+1)+"-b.csv", real_batch[2])
		ct_layer3 = evalConv_BNRelu_BL(cont, ct_layer3, ker_in3, bn_a3, bn_b3, alpha, in_wids[2], ker_wid, real_batch[2], real_batch[2], 8, pad, false, false, prt_result)
		fmt.Println("Block3, Layer ", i, "done!")
	}
	timings[4] = time.Since(new_start).Seconds()

	// // for final reduce_mean & FC
	new_start = time.Now()
	ker_inf := readTxt(weight_dir+"final-fckernel.csv", real_batch[2]*10)
	bn_bf := readTxt(weight_dir+"final-fcbias.csv", 10)
	ct_result = evalRMFC_BL(cont, ct_layer3, ker_inf, bn_bf, true)
	timings[5] = time.Since(new_start).Seconds()
	fmt.Println("Reduce Mean then, FC done.")

	fmt.Println()
	fmt.Println("===============  DECRYPTION  ===============")
	fmt.Println()
	new_start = time.Now()
	vals_tmp := cont.encoder.Decode(cont.decryptor.DecryptNew(ct_result), cont.logN-1)
	fmt.Printf("Decryption Done in %s \n", time.Since(new_start))
	final_result := prt_mat_one_BL(vals_tmp, max_batch[2], 10)
	fmt.Println("result: ", final_result)
	writeTxt("class_result_BL_"+ker_name+"/class_result_BL_"+ker_name+"_"+strconv.Itoa(iter)+".csv", final_result)

	fmt.Println("Blc1: ", timings[0], " sec")
	fmt.Println("Blc1->2: ", timings[1], " sec")
	fmt.Println("Blc2: ", timings[2], " sec")
	fmt.Println("Blc2->3: ", timings[3], " sec")
	fmt.Println("Blc3: ", timings[4], " sec")
	fmt.Println("Final (reduce_mean & FC): ", timings[5], " sec")
	fmt.Printf("Total done in %s \n", time.Since(begin_start))
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
					input[i*in_wids[2]*max_batch[2]+j*max_batch[2]+b*8] = k
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
	final_result := prt_mat_one_BL(vals_tmp, max_batch[2], 10)
	fmt.Println(final_result)
}

func testImageNet_BL() {
	pad := 0
	num_blc1 := 4
	num_blc2 := 4
	logN := 16
	in_wids := []int{16, 32}      // = raw_in_wids = same as python
	real_batch := []int{256, 128} // same as python
	py_bn_a := []float64{0.1, 0.1}
	ker_wid := 3
	kp_wids := make([]int, len(in_wids)) // NOT used in BL
	copy(kp_wids, in_wids)
	cont := newContext(logN, ker_wid, in_wids, kp_wids, true, "BL_Imagenet")

	ker_size := ker_wid * ker_wid
	max_batch := make([]int, len(real_batch)) // the max batch
	for i := range max_batch {
		max_batch[i] = cont.N / (2 * in_wids[i] * in_wids[i])
	}

	alpha := 0.0 // 0.3 => leakyrelu
	input := make([]float64, cont.N)
	k := 0.0
	for i := 0; i < in_wids[0]; i++ {
		for j := 0; j < in_wids[0]; j++ {
			for b := 0; b < real_batch[0]; b++ {
				input[i*in_wids[0]*real_batch[0]+j*real_batch[0]+b] = k
				k += (1.0 / float64(real_batch[0]*in_wids[0]*in_wids[0]))
			}
		}
	}
	fmt.Println("Input: ")
	prt_mat(input, real_batch[0], 3)
	input_1 := make([]float64, len(input)/2)
	input_2 := make([]float64, len(input)/2)
	for k := 0; k < in_wids[0]*in_wids[0]; k++ {
		for b := 0; b < real_batch[0]/2; b++ {
			input_1[k*real_batch[0]/2+b] = input[k*real_batch[0]+b]
			input_2[k*real_batch[0]/2+b] = input[k*real_batch[0]+real_batch[0]/2+b]
		}
	}

	ker_in := make([]float64, real_batch[0]*real_batch[0]*ker_size)
	for i := range ker_in {
		ker_in[i] = 0.1 * float64(i) / float64(len(ker_in))
	}
	ker_in_11 := make([]float64, len(ker_in)/4)
	ker_in_12 := make([]float64, len(ker_in)/4)
	ker_in_21 := make([]float64, len(ker_in)/4)
	ker_in_22 := make([]float64, len(ker_in)/4)
	for k := 0; k < ker_size; k++ {
		for i := 0; i < real_batch[0]/2; i++ {
			for j := 0; j < real_batch[0]/2; j++ {
				pos := k*real_batch[0]*real_batch[0]/4 + (i*real_batch[0]/2 + j) // [k][i][j]
				b_pos := k * real_batch[0] * real_batch[0]
				ker_in_11[pos] = ker_in[b_pos+i*real_batch[0]+j]
				ker_in_12[pos] = ker_in[b_pos+i*real_batch[0]+real_batch[0]/2+j]
				ker_in_21[pos] = ker_in[b_pos+(i+real_batch[0]/2)*real_batch[0]+j]
				ker_in_22[pos] = ker_in[b_pos+(i+real_batch[0]/2)*real_batch[0]+real_batch[0]/2+j]
			}
		}
	}

	ker_in12 := make([]float64, real_batch[0]*real_batch[1]*ker_size)
	for i := range ker_in12 {
		ker_in12[i] = 0.1 * float64(i) / float64(len(ker_in12))
	}
	ker_in12_11 := make([]float64, len(ker_in12)/8)
	ker_in12_21 := make([]float64, len(ker_in12)/8)
	ker_in12_12 := make([]float64, len(ker_in12)/8)
	ker_in12_22 := make([]float64, len(ker_in12)/8)
	ker_in12_13 := make([]float64, len(ker_in12)/8)
	ker_in12_23 := make([]float64, len(ker_in12)/8)
	ker_in12_14 := make([]float64, len(ker_in12)/8)
	ker_in12_24 := make([]float64, len(ker_in12)/8)
	for k := 0; k < ker_size; k++ {
		for i := 0; i < real_batch[0]/2; i++ {
			for j := 0; j < real_batch[1]/4; j++ {
				pos := k*real_batch[0]/2*real_batch[1]/4 + (i*real_batch[1]/4 + j) // [k][i][j]
				b_pos1 := k*real_batch[0]*real_batch[1] + i*real_batch[1]
				b_pos2 := k*real_batch[0]*real_batch[1] + (i+real_batch[0]/2)*real_batch[1]
				ker_in12_11[pos] = ker_in12[b_pos1+j*4]
				ker_in12_21[pos] = ker_in12[b_pos2+j*4]
				ker_in12_12[pos] = ker_in12[b_pos1+j*4+1]
				ker_in12_22[pos] = ker_in12[b_pos2+j*4+1]
				ker_in12_13[pos] = ker_in12[b_pos1+j*4+2]
				ker_in12_23[pos] = ker_in12[b_pos2+j*4+2]
				ker_in12_14[pos] = ker_in12[b_pos1+j*4+3]
				ker_in12_24[pos] = ker_in12[b_pos2+j*4+3]
			}
		}
	}

	ker_in2 := make([]float64, real_batch[1]*real_batch[1]*ker_size)
	for i := range ker_in2 {
		ker_in2[i] = 0.1 * float64(i) / float64(len(ker_in2))
	}

	bn_a := make([]float64, real_batch[0])
	bn_b := make([]float64, real_batch[0])
	for i := range bn_a {
		bn_a[i] = py_bn_a[0] // * float64(i) / float64(batch)
		bn_b[i] = 0.0        //0.1 * float64(i) // float64(real_batch[0])
	}
	bn_a_1 := make([]float64, len(bn_a)/2)
	bn_a_2 := make([]float64, len(bn_a)/2)
	bn_b_1 := make([]float64, len(bn_b)/2)
	bn_b_2 := make([]float64, len(bn_b)/2)
	for i := range bn_a_1 {
		bn_a_1[i] = bn_a[i]
		bn_a_2[i] = bn_a[i+len(bn_a_1)]
		bn_b_1[i] = bn_b[i]
		bn_b_2[i] = bn_b[i+len(bn_b_1)]
	}
	bn_a12 := make([]float64, real_batch[1])
	bn_b12 := make([]float64, real_batch[1])
	for i := range bn_a12 {
		bn_a12[i] = py_bn_a[1] // * float64(i) / float64(batch)
		bn_b12[i] = 0.0        //0.1 * float64(i)
	}
	bn_a12_1 := make([]float64, len(bn_a12)/4)
	bn_a12_2 := make([]float64, len(bn_a12)/4)
	bn_a12_3 := make([]float64, len(bn_a12)/4)
	bn_a12_4 := make([]float64, len(bn_a12)/4)
	bn_b12_1 := make([]float64, len(bn_b12)/4)
	bn_b12_2 := make([]float64, len(bn_b12)/4)
	bn_b12_3 := make([]float64, len(bn_b12)/4)
	bn_b12_4 := make([]float64, len(bn_b12)/4)
	for i := range bn_a12_1 {
		bn_a12_1[i] = bn_a12[4*i]
		bn_a12_2[i] = bn_a12[4*i+1]
		bn_a12_3[i] = bn_a12[4*i+2]
		bn_a12_4[i] = bn_a12[4*i+3]
		bn_b12_1[i] = bn_b12[4*i]
		bn_b12_2[i] = bn_b12[4*i+1]
		bn_b12_3[i] = bn_b12[4*i+2]
		bn_b12_4[i] = bn_b12[4*i+3]
	}
	bn_a2 := make([]float64, real_batch[1])
	bn_b2 := make([]float64, real_batch[1])
	for i := range bn_a2 {
		bn_a2[i] = py_bn_a[1] // * float64(i) / float64(batch)
		bn_b2[i] = 0.0        //0.1 * float64(i)
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
	input_rs1 := reshape_input_BL(input_1, in_wids[0])
	input_rs2 := reshape_input_BL(input_2, in_wids[0])
	start = time.Now()
	ct_input1 := cont.encryptor.EncryptNew(cont.encoder.EncodeAtLvlNew(cont.ECD_LV, input_rs1, cont.logN-1))
	ct_input2 := cont.encryptor.EncryptNew(cont.encoder.EncodeAtLvlNew(cont.ECD_LV, input_rs2, cont.logN-1))
	fmt.Printf("Encryption done in %s \n", time.Since(start))

	timings := make([]float64, 6)
	begin_start := time.Now()
	new_start := time.Now()

	// ResNet Block 1
	prt_result := false
	for i := 1; i <= num_blc1; i++ {
		if i == num_blc1 {
			prt_result = true
		}
		ct_out1 := cont.evaluator.AddNew(evalConv_BNRelu_BL(cont, ct_input1, ker_in_11, bn_a_1, bn_b_1, alpha, in_wids[0], ker_wid, real_batch[0]/2, real_batch[0]/2, 1, pad, false, false, prt_result),
			evalConv_BNRelu_BL(cont, ct_input2, ker_in_21, bn_a_1, bn_b_1, alpha, in_wids[0], ker_wid, real_batch[0]/2, real_batch[0]/2, 1, pad, false, false, prt_result))
		ct_out2 := cont.evaluator.AddNew(evalConv_BNRelu_BL(cont, ct_input1, ker_in_12, bn_a_2, bn_b_2, alpha, in_wids[0], ker_wid, real_batch[0]/2, real_batch[0]/2, 1, pad, false, false, prt_result),
			evalConv_BNRelu_BL(cont, ct_input2, ker_in_22, bn_a_2, bn_b_2, alpha, in_wids[0], ker_wid, real_batch[0]/2, real_batch[0]/2, 1, pad, false, false, prt_result))
		ct_input1 = ct_out1
		ct_input2 = ct_out2
		fmt.Println("Block1, Layer ", i, "done!")
	}
	fmt.Println("done.")
	timings[0] = time.Since(new_start).Seconds()
	new_start = time.Now()

	// ct_result := evalConv_BNRelu_BL(cont, ct_layer, ker_in12, bn_a2, bn_b2, alpha, in_wids[0], ker_wid, real_batch[0], real_batch[1], 1, true, false, prt_result)

	ct_res1 := cont.evaluator.AddNew(evalConv_BNRelu_BL(cont, ct_input1, ker_in12_11, bn_a12_1, bn_b12_1, alpha, in_wids[0], ker_wid, real_batch[0]/2, real_batch[0]/2, 1, pad, false, false, prt_result),
		evalConv_BNRelu_BL(cont, ct_input2, ker_in12_21, bn_a12_1, bn_b12_1, alpha, in_wids[0], ker_wid, real_batch[0]/2, real_batch[0]/2, 1, pad, false, false, prt_result))
	ct_res2 := cont.evaluator.AddNew(evalConv_BNRelu_BL(cont, ct_input1, ker_in12_12, bn_a12_2, bn_b12_2, alpha, in_wids[0], ker_wid, real_batch[0]/2, real_batch[0]/2, 1, pad, false, false, prt_result),
		evalConv_BNRelu_BL(cont, ct_input2, ker_in12_22, bn_a12_2, bn_b12_2, alpha, in_wids[0], ker_wid, real_batch[0]/2, real_batch[0]/2, 1, pad, false, false, prt_result))
	ct_res3 := cont.evaluator.AddNew(evalConv_BNRelu_BL(cont, ct_input1, ker_in12_13, bn_a12_3, bn_b12_3, alpha, in_wids[0], ker_wid, real_batch[0]/2, real_batch[0]/2, 1, pad, false, false, prt_result),
		evalConv_BNRelu_BL(cont, ct_input2, ker_in12_23, bn_a12_3, bn_b12_3, alpha, in_wids[0], ker_wid, real_batch[0]/2, real_batch[0]/2, 1, pad, false, false, prt_result))
	ct_res4 := cont.evaluator.AddNew(evalConv_BNRelu_BL(cont, ct_input1, ker_in12_14, bn_a12_4, bn_b12_4, alpha, in_wids[0], ker_wid, real_batch[0]/2, real_batch[0]/2, 1, pad, false, false, prt_result),
		evalConv_BNRelu_BL(cont, ct_input2, ker_in12_24, bn_a12_4, bn_b12_4, alpha, in_wids[0], ker_wid, real_batch[0]/2, real_batch[0]/2, 1, pad, false, false, prt_result))

	ct_res1 = evalRot_BL(cont, ct_res1, in_wids[0], 0, false)
	ct_res2 = evalRot_BL(cont, ct_res2, in_wids[0], 1, false)
	ct_res3 = evalRot_BL(cont, ct_res3, in_wids[0], 2, false)
	ct_res4 = evalRot_BL(cont, ct_res4, in_wids[0], 3, false)
	ct_result := cont.evaluator.AddNew(cont.evaluator.AddNew(ct_res1, ct_res2), cont.evaluator.AddNew(ct_res3, ct_res4))

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
		prt_mat_BL(vals_tmp, (cont.N / (2 * in_wids[1] * in_wids[1])), 3)
	}

	timings[1] = time.Since(new_start).Seconds()
	fmt.Println("Block1 to 2 done!")

	// ResNet Block 2
	ct_in := ct_result
	prt_result = false
	for i := 1; i <= num_blc2; i++ {
		if i == num_blc2 {
			prt_result = true
		}
		ct_in = evalConv_BNRelu_BL(cont, ct_in, ker_in2, bn_a2, bn_b2, alpha, in_wids[1], ker_wid, real_batch[1], real_batch[1], 1, pad, false, false, prt_result)

		fmt.Println("Block2, Layer ", i, "done!")
	}
	timings[2] = time.Since(new_start).Seconds()
	new_start = time.Now()

	// no RMFC

	fmt.Println()
	fmt.Println("===============  DECRYPTION  ===============")
	fmt.Println()
	start = time.Now()
	vals_tmp := cont.encoder.Decode(cont.decryptor.DecryptNew(ct_in), cont.logN-1)
	fmt.Printf("Decryption Done in %s \n", time.Since(start))
	fmt.Println("result: ")
	prt_mat_BL(vals_tmp, max_batch[1], 3)

	fmt.Println("Blc1: ", timings[0], " sec")
	fmt.Println("Blc1->2: ", timings[1], " sec")
	fmt.Println("Blc2: ", timings[2], " sec")
	fmt.Println("Final (reduce_mean & FC): ", timings[5], " sec")
	fmt.Printf("Total done in %s \n", time.Since(begin_start))

}

func testImageNet_BL_final() {
	pad0 := 2
	pad1 := 1
	num_blc := 3
	logN := 12
	raw_in_wids := []int{14, 7} // = raw_in_wids = same as python
	in_wids := []int{16, 8}
	real_batch := []int{16, 32} // same as python
	py_bn_a := []float64{0.2}
	ker_wid := 3
	final_out := 60
	prt_result := true
	kp_wids := make([]int, len(in_wids)) // NOT used in BL
	copy(kp_wids, in_wids)
	cont := newContext(logN, ker_wid, in_wids, kp_wids, true, "BL_Imagenet_final")

	ker_size := ker_wid * ker_wid
	max_batch := make([]int, len(real_batch)) // the max batch
	for i := range max_batch {
		max_batch[i] = cont.N / (2 * in_wids[i] * in_wids[i])
	}

	alpha := 0.0 // 0.3 => leakyrelu
	input := make([]float64, raw_in_wids[0]*raw_in_wids[0]*real_batch[0])
	for i := range input {
		input[i] = 1.0 - 1.0*float64(i)/float64(len(input))
	}
	pad_input1 := make([]float64, in_wids[0]*in_wids[0]*real_batch[0]/2)
	pad_input2 := make([]float64, in_wids[0]*in_wids[0]*real_batch[0]/2)
	for i := 0; i < raw_in_wids[0]; i++ {
		for j := 0; j < raw_in_wids[0]; j++ {
			for b := 0; b < real_batch[0]/2; b++ {
				pad_input1[b+j*real_batch[0]/2+i*real_batch[0]/2*in_wids[0]] = input[b+j*real_batch[0]+i*real_batch[0]*raw_in_wids[0]]
				pad_input2[b+j*real_batch[0]/2+i*real_batch[0]/2*in_wids[0]] = input[b+real_batch[0]/2+j*real_batch[0]+i*real_batch[0]*raw_in_wids[0]]
			}
		}
	}

	ker_in12 := make([]float64, real_batch[0]*real_batch[1]*ker_size)
	for i := range ker_in12 {
		ker_in12[i] = 0.1 * float64(i) / float64(len(ker_in12))
	}

	ker_in12_sep := make([][][]float64, 4) // number of output ctxts
	for out := 0; out < 4; out++ {
		ker_in12_sep[out] = make([][]float64, 2) // number of input ctxts
		for in := 0; in < 2; in++ {
			ker_in12_sep[out][in] = make([]float64, len(ker_in12)/(2*4))
			for k := 0; k < ker_size; k++ {
				for i := 0; i < real_batch[0]/2; i++ { // in
					for j := 0; j < real_batch[1]/4; j++ { // out
						ker_in12_sep[out][in][k*real_batch[0]*real_batch[1]/8+i*real_batch[1]/4+j] =
							ker_in12[k*real_batch[0]*real_batch[1]+(i+in*real_batch[0]/2)*real_batch[1]+4*j+out] // [i][4*j]
					}
				}
			}
		}
	}

	ker_in2 := make([]float64, real_batch[1]*real_batch[1]*ker_size)
	for i := range ker_in2 {
		ker_in2[i] = 0.1 * float64(i) / float64(len(ker_in2))
	}

	bn_a12 := make([]float64, real_batch[1])
	bn_b12 := make([]float64, real_batch[1])
	for i := range bn_a12 {
		bn_a12[i] = py_bn_a[0] // * float64(i) / float64(batch)
		bn_b12[i] = 0.0        //0.1 * float64(i)
	}
	bn_a12_sep := make([][]float64, 4)
	bn_b12_sep := make([][]float64, 4)
	for out := 0; out < 4; out++ {
		bn_a12_sep[out] = make([]float64, real_batch[1]/4)
		bn_b12_sep[out] = make([]float64, real_batch[1]/4)
		for i := 0; i < real_batch[1]/4; i++ {
			bn_a12_sep[out][i] = bn_a12[4*i+out]
			bn_b12_sep[out][i] = bn_b12[4*i+out]
		}
	}

	bn_a2 := make([]float64, real_batch[1])
	bn_b2 := make([]float64, real_batch[1])
	for i := range bn_a2 {
		bn_a2[i] = py_bn_a[0] // * float64(i) / float64(batch)
		bn_b2[i] = 0.0        //0.1 * float64(i)
	}

	fmt.Println("vec size: ", cont.N)
	fmt.Println("input width: ", raw_in_wids)
	fmt.Println("kernel width: ", ker_wid)
	fmt.Println("num batches: ", real_batch)

	// input encryption
	fmt.Println()
	fmt.Println("===============  ENCRYPTION  ===============")
	fmt.Println()
	input1_rs := reshape_input_BL(pad_input1, in_wids[0])
	input2_rs := reshape_input_BL(pad_input2, in_wids[0])
	fmt.Println("Input: ")
	if prt_result {
		prt_mat_BL(input1_rs, real_batch[0]/2, 4)
		prt_mat_BL(input2_rs, real_batch[0]/2, 4)
	}
	start = time.Now()
	ct_input1 := cont.encryptor.EncryptNew(cont.encoder.EncodeAtLvlNew(cont.ECD_LV, input1_rs, cont.logN-1))
	ct_input2 := cont.encryptor.EncryptNew(cont.encoder.EncodeAtLvlNew(cont.ECD_LV, input2_rs, cont.logN-1))
	fmt.Printf("Encryption done in %s \n", time.Since(start))

	timings := make([]float64, 6)
	begin_start := time.Now()
	new_start := time.Now()

	// ResNet Block 1 to 2
	new_start = time.Now()

	ct_res := make([]*ckks.Ciphertext, 4)
	for pos := 0; pos < 4; pos++ {
		ct_res[pos] = cont.evaluator.AddNew(evalConv_BNRelu_BL(cont, ct_input1, ker_in12_sep[pos][0], bn_a12_sep[pos], bn_b12_sep[pos], alpha, in_wids[0], ker_wid, real_batch[0]/2, real_batch[0]/2, 1, pad0, false, false, false),
			evalConv_BNRelu_BL(cont, ct_input2, ker_in12_sep[pos][1], bn_a12_sep[pos], bn_b12_sep[pos], alpha, in_wids[0], ker_wid, real_batch[0]/2, real_batch[0]/2, 1, pad0, false, false, false))
		ct_res[pos] = evalRot_BL(cont, ct_res[pos], in_wids[0], pos, false)
	}
	ct_result := cont.evaluator.AddNew(cont.evaluator.AddNew(ct_res[0], ct_res[1]), cont.evaluator.AddNew(ct_res[2], ct_res[3]))

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
		prt_mat_BL(vals_tmp, max_batch[1], 3)
	}

	timings[0] = time.Since(new_start).Seconds()
	fmt.Println("Block1 to 2 done!")

	// ResNet Block 2
	ct_in := ct_result
	prt_result = false
	for i := 1; i <= num_blc; i++ {
		if i == num_blc {
			prt_result = true
		}
		ct_in = evalConv_BNRelu_BL(cont, ct_in, ker_in2, bn_a2, bn_b2, alpha, in_wids[1], ker_wid, real_batch[1], real_batch[1], 1, pad1, false, false, prt_result)

		fmt.Println("Block2, Layer ", i, "done!")
	}
	timings[1] = time.Since(new_start).Seconds()
	new_start = time.Now()

	// RMFC
	ker_inf := make([]float64, real_batch[1]*final_out)
	for i := range ker_inf {
		ker_inf[i] = 1.0 * float64(i) / (float64(real_batch[1] * final_out))
	}
	ker_inf1 := make([]float64, real_batch[1]*final_out/2)
	ker_inf2 := make([]float64, real_batch[1]*final_out/2)
	for i := 0; i < real_batch[1]; i++ {
		for j := 0; j < final_out/2; j++ {
			ker_inf1[i*final_out/2+j] = ker_inf[i*final_out+j]
			ker_inf2[i*final_out/2+j] = ker_inf[i*final_out+j+final_out/2]
		}
	}
	ct_result1 := evalRMFC_BL_img(cont, ct_in, ker_inf1, real_batch[1], final_out/2, true)
	ct_result2 := evalRMFC_BL_img(cont, ct_in, ker_inf2, real_batch[1], final_out/2, true)
	fmt.Printf("RMFC done in %s \n", time.Since(start))
	timings[2] = time.Since(new_start).Seconds()

	fmt.Println()
	fmt.Println("===============  DECRYPTION  ===============")
	fmt.Println()
	start = time.Now()
	vals_tmp1 := cont.encoder.Decode(cont.decryptor.DecryptNew(ct_result1), cont.logN-1)
	vals_tmp2 := cont.encoder.Decode(cont.decryptor.DecryptNew(ct_result2), cont.logN-1)
	fmt.Printf("Decryption Done in %s \n", time.Since(start))
	final_result1 := prt_mat_one_BL_img(vals_tmp1, max_batch[1], final_out/2)
	final_result2 := prt_mat_one_BL_img(vals_tmp2, max_batch[1], final_out/2)
	fmt.Println("result1: ", final_result1)
	fmt.Println("result2: ", final_result2)

	fmt.Println("Blc1->2: ", timings[0], " sec")
	fmt.Println("Blc2: ", timings[1], " sec")
	fmt.Println("Final (reduce_mean & FC): ", timings[2], " sec")
	fmt.Printf("Total done in %s \n", time.Since(begin_start))

}

func testImageNet_BL_final_in(st, end int) {
	pad0 := 2
	pad1 := 1
	ker_name := "ker3"
	weight_dir := "weight_imgnet_ker3_h5/"
	logN := 16
	num_blc := 3
	raw_in_wids := []int{14, 7} // same as python
	in_wids := []int{16, 8}
	real_batch := []int{256, 512} // same as python
	ker_wid := 3
	kp_wids := []int{16, 8}
	final_out := 1000
	prt_result := true
	cont := newContext(logN, ker_wid, in_wids, kp_wids, true, "BL_Imagenet_final")

	ker_size := ker_wid * ker_wid
	max_batch := make([]int, len(real_batch)) // the max batch
	for i := range max_batch {
		max_batch[i] = cont.N / (2 * in_wids[i] * in_wids[i])
	}

	alpha := 0.0 // 0.3 => leakyrelu

	for name_iter := st; name_iter < end; name_iter++ {
		fmt.Println("Start ", name_iter, "-th iter..")
		raw_input := readTxt("./Imagenet/ker3_ct_in_one2/input_"+strconv.Itoa(name_iter)+".csv", raw_in_wids[0]*raw_in_wids[0]*real_batch[0])
		pad_input1 := make([]float64, in_wids[0]*in_wids[0]*real_batch[0]/2)
		pad_input2 := make([]float64, in_wids[0]*in_wids[0]*real_batch[0]/2)
		for i := 0; i < raw_in_wids[0]; i++ {
			for j := 0; j < raw_in_wids[0]; j++ {
				for b := 0; b < real_batch[0]/2; b++ {
					pad_input1[b+j*real_batch[0]/2+i*real_batch[0]/2*in_wids[0]] = raw_input[b+j*real_batch[0]+i*real_batch[0]*raw_in_wids[0]]
					pad_input2[b+j*real_batch[0]/2+i*real_batch[0]/2*in_wids[0]] = raw_input[b+real_batch[0]/2+j*real_batch[0]+i*real_batch[0]*raw_in_wids[0]]
				}
			}
		}

		weight_num := 13 // starting from w9-conv.csv
		ker_in12 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-conv.csv", real_batch[0]*real_batch[1]*ker_size)
		weight_num++
		bn_a12 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-a.csv", real_batch[1])
		bn_b12 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-b.csv", real_batch[1])

		ker_in12_sep := make([][][]float64, 4) // number of output ctxts
		for out := 0; out < 4; out++ {
			ker_in12_sep[out] = make([][]float64, 2) // number of input ctxts
			for in := 0; in < 2; in++ {
				ker_in12_sep[out][in] = make([]float64, len(ker_in12)/(2*4))
				for k := 0; k < ker_size; k++ {
					for i := 0; i < real_batch[0]/2; i++ { // in
						for j := 0; j < real_batch[1]/4; j++ { // out
							ker_in12_sep[out][in][k*real_batch[0]*real_batch[1]/8+i*real_batch[1]/4+j] =
								ker_in12[k*real_batch[0]*real_batch[1]+(i+in*real_batch[0]/2)*real_batch[1]+4*j+out] // [i][4*j]
						}
					}
				}
			}
		}

		bn_a12_sep := make([][]float64, 4)
		bn_b12_sep := make([][]float64, 4)
		for out := 0; out < 4; out++ {
			bn_a12_sep[out] = make([]float64, real_batch[1]/4)
			bn_b12_sep[out] = make([]float64, real_batch[1]/4)
			for i := 0; i < real_batch[1]/4; i++ {
				bn_a12_sep[out][i] = bn_a12[4*i+out]
				bn_b12_sep[out][i] = bn_b12[4*i+out]
			}
		}

		fmt.Println("vec size: ", cont.N)
		fmt.Println("input width: ", raw_in_wids)
		fmt.Println("kernel width: ", ker_wid)
		fmt.Println("num batches: ", real_batch)

		// input encryption
		fmt.Println()
		fmt.Println("===============  ENCRYPTION  ===============")
		fmt.Println()
		input1_rs := reshape_input_BL(pad_input1, in_wids[0])
		input2_rs := reshape_input_BL(pad_input2, in_wids[0])
		fmt.Println("Input: ")
		if prt_result {
			prt_mat_BL(input1_rs, real_batch[0]/2, 4)
			prt_mat_BL(input2_rs, real_batch[0]/2, 4)
		}
		start = time.Now()
		ct_input1 := cont.encryptor.EncryptNew(cont.encoder.EncodeAtLvlNew(cont.ECD_LV, input1_rs, cont.logN-1))
		ct_input2 := cont.encryptor.EncryptNew(cont.encoder.EncodeAtLvlNew(cont.ECD_LV, input2_rs, cont.logN-1))
		fmt.Printf("Encryption done in %s \n", time.Since(start))

		timings := make([]float64, 6)
		begin_start := time.Now()
		new_start := time.Now()

		// ResNet Block 1 to 2
		new_start = time.Now()
		ct_res := make([]*ckks.Ciphertext, 4)
		for pos := 0; pos < 4; pos++ {
			ct_res[pos] = cont.evaluator.AddNew(evalConv_BNRelu_BL(cont, ct_input1, ker_in12_sep[pos][0], bn_a12_sep[pos], bn_b12_sep[pos], alpha, in_wids[0], ker_wid, real_batch[0]/2, real_batch[0]/2, 1, pad0, false, false, false),
				evalConv_BNRelu_BL(cont, ct_input2, ker_in12_sep[pos][1], bn_a12_sep[pos], bn_b12_sep[pos], alpha, in_wids[0], ker_wid, real_batch[0]/2, real_batch[0]/2, 1, pad0, false, false, false))
			ct_res[pos] = evalRot_BL(cont, ct_res[pos], in_wids[0], pos, false)
		}
		ct_result := cont.evaluator.AddNew(cont.evaluator.AddNew(ct_res[0], ct_res[1]), cont.evaluator.AddNew(ct_res[2], ct_res[3]))

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
			prt_mat_BL(vals_tmp, max_batch[1], 3)
		}

		timings[0] = time.Since(new_start).Seconds()
		new_start = time.Now()
		fmt.Println("Block1 to 2 done!")

		// ResNet Block 2
		ct_in := ct_result
		prt_result = false
		for i := 1; i <= num_blc; i++ {
			ker_in2 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-conv.csv", real_batch[1]*real_batch[1]*ker_size)
			weight_num++
			bn_a2 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-a.csv", real_batch[1])
			bn_b2 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-b.csv", real_batch[1])
			if i == num_blc {
				prt_result = true
			}
			ct_in = evalConv_BNRelu_BL(cont, ct_in, ker_in2, bn_a2, bn_b2, alpha, in_wids[1], ker_wid, real_batch[1], real_batch[1], 1, pad1, false, false, prt_result)

			fmt.Println("Block2, Layer ", i, "done!")
		}
		timings[1] = time.Since(new_start).Seconds()
		new_start = time.Now()

		// RMFC
		ker_inf := readTxt(weight_dir+"fc.csv", real_batch[1]*final_out)
		bn_af := make([]float64, real_batch[1]*2)
		for i := range bn_af {
			bn_af[i] = 1.0 / (7 * 7) // for reduce mean on 8*8 elements
		}
		ker_inf1 := make([]float64, real_batch[1]*final_out/2)
		ker_inf2 := make([]float64, real_batch[1]*final_out/2)
		for i := 0; i < real_batch[1]; i++ {
			for j := 0; j < final_out/2; j++ {
				ker_inf1[i*final_out/2+j] = ker_inf[i*final_out+j]
				ker_inf2[i*final_out/2+j] = ker_inf[i*final_out+j+final_out/2]
			}
		}
		ct_result1 := evalRMFC_BL_img(cont, ct_in, ker_inf1, real_batch[1], final_out/2, true)
		ct_result2 := evalRMFC_BL_img(cont, ct_in, ker_inf2, real_batch[1], final_out/2, true)
		fmt.Printf("RMFC done in %s \n", time.Since(start))
		timings[2] = time.Since(new_start).Seconds()

		fmt.Println()
		fmt.Println("===============  DECRYPTION  ===============")
		fmt.Println()
		start = time.Now()
		vals_tmp1 := cont.encoder.Decode(cont.decryptor.DecryptNew(ct_result1), cont.logN-1)
		vals_tmp2 := cont.encoder.Decode(cont.decryptor.DecryptNew(ct_result2), cont.logN-1)
		fmt.Printf("Decryption Done in %s \n", time.Since(start))
		final_result1 := prt_mat_one_BL_img(vals_tmp1, max_batch[1], final_out/2)
		final_result2 := prt_mat_one_BL_img(vals_tmp2, max_batch[1], final_out/2)
		final_result := append(final_result1[:500], final_result2[:500]...)
		fmt.Println(final_result)

		writeTxt("imgnet_class_result_"+ker_name+"_final_fast_BL/imgnet_enc_class_result_"+ker_name+"_BL_"+strconv.Itoa(name_iter)+".csv", final_result)

		fmt.Println("Blc1->2: ", timings[0], " sec")
		fmt.Println("Blc2: ", timings[1], " sec")
		fmt.Println("Final (reduce_mean & FC): ", timings[2], " sec")
		fmt.Printf("Total done in %s \n", time.Since(begin_start))
	}
}

func basic() {
	logN := 16
	N := (1 << logN)

	// Schemes parameters are created from scratch
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:     logN,
		LogQ:     []int{log_out_scale + log_c_scale, 49},
		LogP:     []int{60},
		Sigma:    rlwe.DefaultSigma,
		LogSlots: logN - 1,
		Scale:    float64(1 << 51),
	})
	if err != nil {
		panic(err)
	}

	fmt.Printf("hex: %s \n", strconv.FormatUint(params.Q()[1], 16))

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
	_ = res
	// for i := range res {
	// 	fmt.Println(real(res[i]))
	// }

}
