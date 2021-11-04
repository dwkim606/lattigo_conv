package main

import (
	"fmt"
	"math"
	"time"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/rlwe"
)

// output bitreversed input with bitwid
func reverseBits(num uint32, bitwid int) uint32 {
	num = num << (32 - bitwid)

	var ret = uint32(0)
	var power = uint32(31)
	for num != 0 {
		ret += (num & 1) << power
		num = num >> 1
		power -= 1
	}
	return ret
}

// output the slice with bitreversed order from input
func reverseOrder(input []float64, bitwid int) []float64 {
	out := make([]float64, len(input))
	for i := range out {
		out[i] = input[reverseBits(uint32(i), bitwid)]
	}

	return out
}

// Extract upper left elements of size prt_wid * prt_wid from the input arrg vec with batch batches
// prt_wid, batch all are those from the output of our conv algorithm (consider padding)
func reshape_conv_out(result []float64, prt_wid, out_num int) []float64 {
	prt_out := make([]float64, prt_wid*prt_wid*out_num)
	in_wid := 2 * prt_wid
	batch := len(result) / (in_wid * in_wid)

	for i := 0; i < out_num; i++ {
		for j := 0; j < prt_wid; j++ {
			for k := 0; k < prt_wid; k++ {
				prt_out[i+out_num*(j*prt_wid+k)] = result[i+batch*(j*in_wid+k)] //[batch*(in_wid+1)*(ker_wid-1)+i+batch*(j*in_wid+k)]
			}
		}
	}

	return prt_out
}

// Reshape 1-D ker_in (from python) into batch number of ker_outs: ker_out[i][j] = j-th kernel (elements then batch order) for i-th output
// i.e., ker_out is of the shape (out_batch, (in_batch * ker_size))
// trans = true for transposed convolution
func reshape_ker_old(ker_in []float64, ker_out [][]float64, k_sz int, trans bool) {
	out_batch := len(ker_out)
	in_batch := len(ker_in) / (k_sz * out_batch)

	for i := 0; i < out_batch; i++ {
		ker_out[i] = make([]float64, k_sz*in_batch)
		for j := 0; j < in_batch; j++ {
			for k := 0; k < k_sz; k++ {
				if trans {
					ker_out[i][j*k_sz+(k_sz-k-1)] = ker_in[j+i*in_batch+k*out_batch*in_batch]
				} else {
					ker_out[i][j*k_sz+k] = ker_in[j+i*in_batch+k*out_batch*in_batch]
				}
			}
		}
	}
}

// Reshape 1-D ker_in (from python) into batch number of ker_outs: ker_out[i][j] = j-th kernel (elements then batch order) for i-th output
// i.e., ker_out is of the shape (out_batch, (in_batch * ker_size))
// trans = true for transposed convolution
func reshape_ker(ker_in []float64, k_sz, out_batch int, trans bool) (ker_out [][]float64) {
	ker_out = make([][]float64, out_batch)
	in_batch := len(ker_in) / (k_sz * out_batch)

	for i := 0; i < out_batch; i++ {
		ker_out[i] = make([]float64, k_sz*in_batch)
		for j := 0; j < in_batch; j++ {
			for k := 0; k < k_sz; k++ {
				if trans {
					ker_out[i][j*k_sz+(k_sz-k-1)] = ker_in[j+i*in_batch+k*out_batch*in_batch]
				} else {
					ker_out[i][j*k_sz+k] = ker_in[i+j*out_batch+k*out_batch*in_batch]
					// ker_out[i][j*k_sz+k] = ker_in[j+i*in_batch+k*out_batch*in_batch]
				}
			}
		}
	}
	return
}

// Encode ker_outs from reshape_ker into the i-th ker vector output
// in_wid, in_batch is those for input (to be convolved) includng padding
func encode_ker(ker_in [][]float64, pos, i, in_wid, in_batch, ker_wid int, trans bool) []float64 {
	vec_size := in_wid * in_wid * in_batch
	output := make([]float64, vec_size)
	bias := pos * ker_wid * ker_wid * in_batch
	k_sz := ker_wid * ker_wid

	// allocate each kernel so that 0-th batch and B-1th batch adds together at B-1th position (B = in_batch)
	for j := 0; j < in_batch; j++ { // j-th input (batch)
		for k := 0; k < k_sz; k++ {
			output[(in_wid*(k/ker_wid)+k%ker_wid)*in_batch+j] = ker_in[i][(in_batch-1-j)*k_sz+(k_sz-1-k)+bias] // * scale;
		}
	}

	// for j := 0; j < in_batch; j++ { // j-th input (batch)
	// 	for k := 0; k < ker_wid*ker_wid; k++ {
	// 		if k == 0 {
	// 			if j == 0 {
	// 				tmp[(vec_size-j)%vec_size] = ker_in[i][j*ker_wid*ker_wid+(ker_wid*ker_wid-1)+bias] // * scale;
	// 			} else {
	// 				tmp[(vec_size-j)%vec_size] = -ker_in[i][j*ker_wid*ker_wid+(ker_wid*ker_wid-1)+bias] // * scale;
	// 			}
	// 		} else {
	// 			tmp[-j+(in_wid*(k/ker_wid)+k%ker_wid)*in_batch] = ker_in[i][j*ker_wid*ker_wid+(ker_wid*ker_wid-1-k)+bias] // * scale;
	// 		}
	// 	}
	// }

	// move the kernel to left adj times, so that the result of "transposed" convolution appears at 0-th position
	// adj := (in_wid+1)*(ker_wid-3)/2 + (in_batch - 1)
	if trans {
		adj := (in_batch - 1) + (in_batch)*(in_wid+1)*(ker_wid-3)/2
		tmp := make([]float64, adj)
		for i := 0; i < adj; i++ {
			tmp[i] = output[vec_size-adj+i]
			output[vec_size-adj+i] = -output[i]
		}
		for i := 0; i < vec_size-2*adj; i++ {
			output[i] = output[i+adj]
		}
		for i := 0; i < adj; i++ {
			output[i+vec_size-2*adj] = tmp[i]
		}
		// fmt.Println("tmp: ", tmp)
	} else {
		adj := (in_batch - 1) + (in_batch)*(in_wid+1)*(ker_wid-1)/2
		tmp := make([]float64, adj)
		for i := 0; i < adj; i++ {
			tmp[i] = output[vec_size-adj+i]
			output[vec_size-adj+i] = -output[i]
		}
		for i := 0; i < vec_size-2*adj; i++ {
			output[i] = output[i+adj]
		}
		for i := 0; i < adj; i++ {
			output[i+vec_size-2*adj] = tmp[i]
		}
	}

	return output
}

// Generate the logN # of plaintexts idx[i] = X^(2^i) and GaloisKeys for each
// Required for Packing
func gen_idxNlogs(E_lv int, keygen rlwe.KeyGenerator, sk *rlwe.SecretKey, encoder ckks.Encoder, params ckks.Parameters) (idx []*ckks.Plaintext, pack_eval ckks.Evaluator) {
	logN := params.LogN()
	N := params.N()
	gals := []uint64{}
	idx = make([]*ckks.Plaintext, logN)
	coeffs := make([]float64, N)

	for i := 0; i < logN; i++ {
		coeffs[1<<i] = 1.0
		idx[i] = ckks.NewPlaintext(params, E_lv, 1.0)
		encoder.EncodeCoeffs(coeffs, idx[i])
		encoder.ToNTT(idx[i])
		coeffs[1<<i] = 0.0

		gals = append(gals, (1<<(i+1) + 1)) // Add galois autos required
	}

	pack_eval = ckks.NewEvaluator(params, rlwe.EvaluationKey{Rtks: keygen.GenRotationKeys(gals, sk)})

	return idx, pack_eval
}

// Pack cnum (alwyas Po2) number of ctxts(in_a[i]) into one
// Each in_a[i] must be arrvec (i.e., sparse with steps) with full size
// Also get the plaintexts idx[i] = X^i from gen_idx
func pack_ctxts(pack_eval ckks.Evaluator, ctxts_in []*ckks.Ciphertext, cnum int, idx []*ckks.Plaintext, params ckks.Parameters) (ctxt *ckks.Ciphertext) {
	step := cnum / 2
	ctxts := make([]*ckks.Ciphertext, cnum)
	for i := 0; i < cnum; i++ {
		ctxts[i] = ctxts_in[i].CopyNew()
		ctxts[i].SetScalingFactor(ctxts[i].Scale * float64(cnum))
	}

	var tmp1, tmp2 *ckks.Ciphertext

	logStep := 0
	for i := step; i > 1; i /= 2 {
		logStep++
	}
	j := params.LogN() - logStep

	for step > 0 {
		for i := 0; i < step; i++ {
			tmp1 = pack_eval.MulNew(ctxts[i+step], idx[logStep])
			tmp2 = pack_eval.SubNew(ctxts[i], tmp1)
			pack_eval.Add(ctxts[i], tmp1, tmp1)
			pack_eval.RotateGal(tmp2, (1<<j)+1, tmp2)
			pack_eval.Add(tmp1, tmp2, ctxts[i])
		}
		step /= 2
		logStep--
		j++
	}

	return ctxts[0]
}

// extend ctxt using given rotations so that it outputs a ctxt to be convolved with filter
func ext_ctxt(eval ckks.Evaluator, encoder ckks.Encoder, input *ckks.Ciphertext, r_idx, m_idx map[int][]int, params ckks.Parameters) (result *ckks.Ciphertext) {
	st := true

	for rot, elt := range r_idx {
		tmp := make([]complex128, params.Slots())
		for i := range elt {
			tmp[i] = complex(float64(elt[i]), 0)
		}
		plain_tmp := encoder.EncodeNTTAtLvlNew(input.Level(), tmp, params.LogSlots())

		if st {
			result = eval.RotateNew(eval.MulNew(input, plain_tmp), rot)
			st = false
		} else {
			ctxt_tmp := eval.RotateNew(eval.MulNew(input, plain_tmp), rot)
			eval.Add(result, ctxt_tmp, result)
		}
	}

	eval.Rescale(result, params.Scale(), result)

	return result
}

// keep ctxt using given idx so that it outputs a ctxt to be convolved with filter
func keep_ctxt(params ckks.Parameters, eval ckks.Evaluator, encoder ckks.Encoder, input *ckks.Ciphertext, idx []int) (result *ckks.Ciphertext) {

	tmp := make([]complex128, params.Slots())
	for i := range idx {
		tmp[i] = complex(float64(idx[i]), 0)
	}
	plain_tmp := encoder.EncodeNTTAtLvlNew(input.Level(), tmp, params.LogSlots())
	result = eval.MulNew(input, plain_tmp)

	eval.Rescale(result, params.Scale(), result)

	return result
}

// block: block size
// N: output vector size (# coeffs)
// Assume 2*len(vec_int) = N
func encode_circ(vec_in []float64, block, N int) []float64 {
	result := make([]float64, N)
	batch := N / (2 * block) // space between each input of result vectors

	for i := 0; i < batch; i++ {
		for j := 0; j < block; j++ {
			result[i+j*batch] = vec_in[i*block+j]
			result[N/2+(i+j*batch)] = -vec_in[i*block+j]
		}
	}

	return result
}

// block: block size
// N: output vector size (# coeffs)
// input is divided into blocks, pos: pos-th block
func encode_circ_in(vec_in []float64, pos, block, N int) []float64 {
	result := make([]float64, N)
	batch := N / (2 * block) // space between each input of result vector

	for i := 0; i < block; i++ {
		result[batch*i] = vec_in[pos*block+i]
	}

	return result
}

// Divide coeff gen later
// alpha: for leaky relu
func evalReLU(params ckks.Parameters, evaluator ckks.Evaluator, ctxt_in *ckks.Ciphertext, alpha float64) (ctxt_out *ckks.Ciphertext) {
	// alpha 10 (from minimax)
	// prescale is multiplied to improve precision (previously done in StoC matmult)
	// coeffs_tmp := []complex128{0.0, 10.8541842577442 * prescale, 0.0, -62.2833925211098 * prescale * prescale * prescale,
	// 	0.0, 114.369227820443 * prescale * prescale * prescale * prescale * prescale, 0.0, -62.8023496973074 * prescale * prescale * prescale * prescale * prescale * prescale * prescale}
	coeffs_tmp := []complex128{0.0, 10.8541842577442, 0.0, -62.2833925211098, 0.0, 114.369227820443, 0.0, -62.8023496973074}
	coeffsReLU := ckks.NewPoly(coeffs_tmp)

	coeffs_tmp2 := []complex128{0.0, 4.13976170985111, 0.0, -5.84997640211679, 0.0, 2.94376255659280, 0.0, -0.454530437460152}
	coeffsReLU2 := ckks.NewPoly(coeffs_tmp2)

	// coeffs_tmp3 := []complex128{0.0, 3.29956739043733, 0.0, -7.84227260291355, 0.0, 12.8907764115564, 0.0, -12.4917112584486, 0.0, 6.94167991428074, 0.0, -2.04298067399942, 0.0, 0.246407138926031}
	// leakyRelu = x (bconst * s(x) + aconst)
	aconst := (alpha + 1) / 2.0
	bconst := (1 - alpha) / 2.0
	coeffs_tmp3 := []complex128{0.0, 3.29956739043733, 0.0, -7.84227260291355, 0.0, 12.8907764115564, 0.0, -12.4917112584486, 0.0, 6.94167991428074, 0.0, -2.04298067399942, 0.0, 0.246407138926031}
	for i := range coeffs_tmp3 {
		coeffs_tmp3[i] = coeffs_tmp3[i] * complex(bconst, 0.0)
	}
	coeffsReLU3 := ckks.NewPoly(coeffs_tmp3)

	fmt.Printf("Eval: ")
	start = time.Now()
	ctxt_sign, _ := evaluator.EvaluatePoly(ctxt_in, coeffsReLU, params.Scale())
	ctxt_sign, _ = evaluator.EvaluatePoly(ctxt_sign, coeffsReLU2, params.Scale())
	ctxt_sign, _ = evaluator.EvaluatePoly(ctxt_sign, coeffsReLU3, params.Scale())

	ctxt_out = evaluator.AddConstNew(ctxt_sign, aconst)

	// Modify c3 scale so that the mult result after rescale has desired scale
	// constPlain := ckks.NewPlaintext(params, ciphertext3.Level(), float64(params.Q()[ciphertext1.Level()])/(ciphertext3.Scale))
	// valuesPlain := make([]float64, params.N())
	// valuesPlain[0] = 1.0
	// encoder.EncodeCoeffs(valuesPlain, constPlain)
	// encoder.ToNTT(constPlain)
	// evaluator.Mul(ciphertext3, constPlain, ciphertext3)

	evaluator.DropLevel(ctxt_in, ctxt_in.Level()-ctxt_out.Level())
	evaluator.Mul(ctxt_out, ctxt_in, ctxt_out)
	evaluator.Relinearize(ctxt_out, ctxt_out)

	return
}

// Encode Kernel and outputs Plain(ker)
// in_wid : width of input (except padding)
// in_batch / out_batch: batches in 1 ctxt (input / output) consider padding
// prepKer with input kernel
// a: coefficient to be multiplied (for BN)
func prepKer_in_trans(params ckks.Parameters, encoder ckks.Encoder, encryptor ckks.Encryptor, in_ker []float64, a []float64, in_wid, ker_wid, in_batch, out_batch, in_batch_real, out_batch_real, ECD_LV int) [][]*ckks.Plaintext {
	ker_size := ker_wid * ker_wid
	in_batch_conv := in_batch / 4 // num batches at convolution 		// strided conv -> /(4)
	in_wid_conv := in_wid * 4     // size of in_wid at convolution 	// strided conv -> *4

	ker_in := make([]float64, in_batch*out_batch*ker_size)
	k := 0
	for i := range ker_in {
		if ((i % in_batch) < in_batch_real) && (((i % (in_batch * out_batch)) / in_batch) < out_batch_real) {
			ker_in[i] = in_ker[k] //* a[((i%(in_batch*out_batch))/in_batch)]
			k++
		} else {
			ker_in[i] = 0.0
		}
	}
	// ker1 := make([][]float64, out_batch) // ker1[i][j] = j-th kernel for i-th output
	ker1 := reshape_ker(ker_in, ker_size, out_batch, true) // ker1[i][j] = j-th kernel for i-th output

	for i := 0; i < out_batch_real; i++ {
		for j := range ker1[i] {
			ker1[i][j] = ker1[i][j] * a[i]
		}
	}

	// prt_mat(ker1[0], 16, 0)
	// fmt.Println("ker11: ", ker1[0])

	pl_ker := make([][]*ckks.Plaintext, 4) // for strided conv
	for pos := 0; pos < 4; pos++ {
		pl_ker[pos] = make([]*ckks.Plaintext, out_batch)
		for i := 0; i < out_batch; i++ {
			pl_ker[pos][i] = ckks.NewPlaintext(params, ECD_LV, params.Scale())
			encoder.EncodeCoeffs(encode_ker(ker1, pos, i, in_wid_conv, in_batch_conv, ker_wid, true), pl_ker[pos][i])
			encoder.ToNTT(pl_ker[pos][i])
		}
	}

	return pl_ker
}

// Encode Kernel and outputs Plain(ker) for normal conv (no stride, no trans)
// in_wid : width of input (include padding)
// in_batch / out_batch: batches in 1 ctxt (input / output) consider padding
// a: coefficient to be multiplied (for BN)
func prepKer_in(params ckks.Parameters, encoder ckks.Encoder, ker_in, BN_a []float64, in_wid, ker_wid, in_batch, out_batch, ECD_LV int) []*ckks.Plaintext {
	ker_size := ker_wid * ker_wid
	// ker_rs := make([][]float64, out_batch) // ker1[i][j] = j-th kernel for i-th output
	// reshape_ker(ker_in, ker_rs, ker_size, false)
	ker_rs := reshape_ker(ker_in, ker_size, out_batch, false) // ker1[i][j] = j-th kernel for i-th output

	for i := 0; i < out_batch; i++ { // apply batch normalization
		for j := range ker_rs[i] {
			ker_rs[i][j] = ker_rs[i][j] * BN_a[i]
		}
	}

	pl_ker := make([]*ckks.Plaintext, out_batch)
	for i := 0; i < out_batch; i++ {
		pl_ker[i] = ckks.NewPlaintext(params, ECD_LV, params.Scale())
		encoder.EncodeCoeffs(encode_ker(ker_rs, 0, i, in_wid, in_batch, ker_wid, false), pl_ker[i])
		encoder.ToNTT(pl_ker[i])
	}

	return pl_ker
}

// Eval Conv, then Pack
// The ciphertexts must be packed into full (without vacant position)
func conv_then_pack_trans(params ckks.Parameters, pack_evaluator ckks.Evaluator, ctxt_in []*ckks.Ciphertext, pl_ker [][]*ckks.Plaintext, plain_idx []*ckks.Plaintext, batch_out int) *ckks.Ciphertext {

	start := time.Now()
	ctxt_out := make([]*ckks.Ciphertext, batch_out)
	for i := 0; i < batch_out; i++ {
		ctxt_out[i] = pack_evaluator.MulNew(ctxt_in[0], pl_ker[0][i])
		for pos := 1; pos < 4; pos++ {
			pack_evaluator.Add(ctxt_out[i], pack_evaluator.MulNew(ctxt_in[pos], pl_ker[pos][i]), ctxt_out[i])
		}
	}

	ctxt_result := pack_ctxts(pack_evaluator, ctxt_out, batch_out, plain_idx, params)
	fmt.Println("Result Scale: ", math.Log2(ctxt_result.Scale))
	fmt.Println("Result LV: ", ctxt_result.Level())
	fmt.Printf("Done in %s \n", time.Since(start))

	return ctxt_result
}

// Eval Conv, then Pack
// The ciphertexts must be packed into full (without vacant position)
func conv_then_pack(params ckks.Parameters, pack_evaluator ckks.Evaluator, ctxt_in *ckks.Ciphertext, pl_ker []*ckks.Plaintext, plain_idx []*ckks.Plaintext, batch_out int) *ckks.Ciphertext {

	start := time.Now()
	ctxt_out := make([]*ckks.Ciphertext, batch_out)
	for i := 0; i < batch_out; i++ {
		ctxt_out[i] = pack_evaluator.MulNew(ctxt_in, pl_ker[i])
	}
	ctxt_result := pack_ctxts(pack_evaluator, ctxt_out, batch_out, plain_idx, params)

	fmt.Println("Result Scale: ", math.Log2(ctxt_result.Scale))
	fmt.Println("Result LV: ", ctxt_result.Level())
	fmt.Printf("Done in %s \n", time.Since(start))

	return ctxt_result
}

// // Perform FFT to encode N coefficients into N/2 complex values (that will be encoded with encode)
// // Always assume that coeffs are of size N
// func cencode(cfsEncoder ckks.EncoderBigComplex, coeffs []float64) (cvals []complex128) {
// 	slots := len(coeffs) / 2
// 	cvals_FFT := make([]*ring.Complex, slots)
// 	for i := 0; i < slots; i++ {
// 		cvals_FFT[i] = ring.NewComplex(ring.NewFloat(coeffs[i], 0), ring.NewFloat(coeffs[i+slots], 0))
// 	}
// 	cfsEncoder.FFT(cvals_FFT, slots)
// 	cvals = make([]complex128, slots)
// 	for i := 0; i < slots; i++ {
// 		cvals[i] = cvals_FFT[i].Float64()
// 	}

// 	return cvals
// }

// // Perform InvFFT to decode N coefficients from N/2 complex values (that would be dencoded from decode)
// // Always assume that complex values are of size N/2
// func cdecode(cfsEncoder ckks.EncoderBigComplex, cvals []complex128) (coeffs []float64) {
// 	slots := len(cvals)
// 	cvals_FFT := make([]*ring.Complex, slots)
// 	for i := 0; i < slots; i++ {
// 		cvals_FFT[i] = ring.NewComplex(ring.NewFloat(real(cvals[i]), 0), ring.NewFloat(imag(cvals[i]), 0))
// 	}
// 	cfsEncoder.InvFFT(cvals_FFT, slots)
// 	coeffs = make([]float64, 2*slots)
// 	for i := 0; i < slots; i++ {
// 		coeffs[i], _ = cvals_FFT[i].Real().Float64()
// 		coeffs[i+slots], _ = cvals_FFT[i].Imag().Float64()
// 	}

// 	return coeffs
// }

// // Generate the logN # of plaintexts idx[i] = X^(2^i) and GaloisKeys for each
// // Required for Packing
// func gen_idxNlogs(keygen rlwe.KeyGenerator, sk *rlwe.SecretKey, cfsEncoder ckks.EncoderBigComplex, encoder ckks.Encoder, params ckks.Parameters) (idx []*ckks.Plaintext, pack_evaluator ckks.Evaluator) {
// 	logN := params.LogN()
// 	slots := params.Slots()
// 	gals := []uint64{}
// 	idx = make([]*ckks.Plaintext, logN)

// 	cvals := make([]*ring.Complex, slots) // keeps degi coeff (at real) and degi+(N/2) coeff (at imag)
// 	for i := 0; i < logN; i++ {
// 		// Initialize cvals then FFT
// 		for j := 0; j < slots; j++ {
// 			cvals[j] = ring.NewComplex(ring.NewFloat(0.0, 0), ring.NewFloat(0.0, 0))
// 		}
// 		if i == logN-1 {
// 			cvals[0].Imag().SetFloat64(1.0)
// 		} else {
// 			cvals[1<<i].Real().SetFloat64(1.0)
// 		}
// 		cfsEncoder.FFT(cvals, slots)

// 		// Copy and transform values to complex128, then encode it
// 		vals := make([]complex128, slots)
// 		for i := range vals {
// 			vals[i] = cvals[i].Float64()
// 		}
// 		idx[i] = encoder.EncodeAtLvlNew(params.MaxLevel(), vals, params.LogSlots())

// 		// Add galois autos required
// 		gals = append(gals, (1<<(i+1) + 1))
// 	}

// 	pack_evaluator = ckks.NewEvaluator(params, rlwe.EvaluationKey{Rtks: keygen.GenRotationKeys(gals, sk)})

// 	return idx, pack_evaluator
// }
