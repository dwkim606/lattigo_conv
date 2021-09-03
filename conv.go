package main

import (
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

// Output the vector containing the valid result of our conv algorithm (format is the same as python; (row, col, batch)-format)
// prt_wid, batch all are those from the output of our conv algorithm (consider padding)
func reshape_conv_out(result []float64, prt_wid, batch int) []float64 {
	prt_out := make([]float64, prt_wid*prt_wid*batch)
	in_wid := 2 * prt_wid

	for i := 0; i < batch; i++ {
		for j := 0; j < prt_wid; j++ {
			for k := 0; k < prt_wid; k++ {
				prt_out[i+batch*(j*prt_wid+k)] = result[i+batch*(j*in_wid+k)] //[batch*(in_wid+1)*(ker_wid-1)+i+batch*(j*in_wid+k)]
			}
		}
	}

	return prt_out
}

// Reshape 1-D ker_in (from python) into batch number of ker_outs: ker_out[i][j] = j-th kernel (elements then batch order) for i-th output
// i.e., ker_out is of the shape (out_batch, (in_batch * ker_size))
// trans = true for transposed convolution
func reshape_ker(ker_in []float64, ker_out [][]float64, k_sz int, trans bool) {
	out_batch := len(ker_out)
	in_batch := len(ker_in) / (k_sz * out_batch)

	for i := 0; i < out_batch; i++ {
		ker_out[i] = make([]float64, k_sz*in_batch)
		for j := 0; j < in_batch; j++ {
			for k := 0; k < k_sz; k++ {
				if trans {
					ker_out[i][j*k_sz+(k_sz-k-1)] = ker_in[i+j*out_batch+k*in_batch*out_batch]
				} else {
					ker_out[i][j*k_sz+k] = ker_in[i+j*out_batch+k*in_batch*out_batch]
				}
			}
		}
	}
}

// Encode ker_outs from reshape_ker into the i-th ker vector output
// in_wid, in_batch is those for input (to be convolved) includng padding
func encode_ker(ker_in [][]float64, pos, i, in_wid, in_batch, ker_wid int) []float64 {
	vec_size := in_wid * in_wid * in_batch
	tmp := make([]float64, vec_size)
	bias := pos * ker_wid * ker_wid * in_batch
	k_sz := ker_wid * ker_wid

	// allocate each kernel so that 0-th batch and B-1th batch adds together at B-1th position (B = in_batch)
	for j := 0; j < in_batch; j++ { // j-th input (batch)
		for k := 0; k < k_sz; k++ {
			tmp[(in_wid*(k/ker_wid)+k%ker_wid)*in_batch+j] = ker_in[i][(in_batch-1-j)*k_sz+(k_sz-1-k)+bias] // * scale;
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
	adj := (in_batch - 1) + (in_batch)*(in_wid+1)*(ker_wid-3)/2
	ttmp := make([]float64, adj)
	for i := 0; i < adj; i++ {
		ttmp[i] = tmp[vec_size-adj+i]
		tmp[vec_size-adj+i] = -tmp[i]
	}
	for i := 0; i < vec_size-2*adj; i++ {
		tmp[i] = tmp[i+adj]
	}
	for i := 0; i < adj; i++ {
		tmp[i+vec_size-2*adj] = ttmp[i]
	}
	// fmt.Println("tmp: ", tmp)

	return tmp
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
// Each in_a[i] must be arrvec (i.e., sparse with steps)
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

	return result
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
