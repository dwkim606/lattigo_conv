package main

import (
	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/lattigo/v2/rlwe"
)

// Generate the logN # of plaintexts idx[i] = X^(2^i) and GaloisKeys for each
// Required for Packing
func gen_idxNlogs(keygen rlwe.KeyGenerator, sk *rlwe.SecretKey, cfsEncoder ckks.EncoderBigComplex, encoder ckks.Encoder, params ckks.Parameters) (idx []*ckks.Plaintext, pack_evaluator ckks.Evaluator) {
	logN := params.LogN()
	slots := params.Slots()
	gals := []uint64{}
	idx = make([]*ckks.Plaintext, logN)

	cvals := make([]*ring.Complex, slots) // keeps degi coeff (at real) and degi+(N/2) coeff (at imag)
	for i := 0; i < logN; i++ {
		// Initialize cvals then FFT
		for j := 0; j < slots; j++ {
			cvals[j] = ring.NewComplex(ring.NewFloat(0.0, 0), ring.NewFloat(0.0, 0))
		}
		if i == logN-1 {
			cvals[0].Imag().SetFloat64(1.0)
		} else {
			cvals[1<<i].Real().SetFloat64(1.0)
		}
		cfsEncoder.FFT(cvals, slots)

		// Copy and transform values to complex128, then encode it
		vals := make([]complex128, slots)
		for i := range vals {
			vals[i] = cvals[i].Float64()
		}
		idx[i] = encoder.EncodeAtLvlNew(params.MaxLevel(), vals, params.LogSlots())

		// Add galois autos required
		gals = append(gals, (1<<(i+1) + 1))
	}

	pack_evaluator = ckks.NewEvaluator(params, rlwe.EvaluationKey{Rtks: keygen.GenRotationKeys(gals, sk)})

	return idx, pack_evaluator
}

// Pack cnum (alwyas Po2) number of ctxts(in_a[i]) into one
// Each in_a[i] must be arrvec (i.e., sparse with steps)
// Also get the plaintexts idx[i] = X^i from gen_idx
func pack_ctxts(pack_evaluator ckks.Evaluator, ctxts_in []*ckks.Ciphertext, cnum int, idx []*ckks.Plaintext, params ckks.Parameters) (ctxt *ckks.Ciphertext) {
	step := cnum / 2
	ctxts := make([]*ckks.Ciphertext, cnum)
	for i := 0; i < cnum; i++ {
		ctxts[i].Copy(ctxts_in[i])
		ctxts[i].SetScalingFactor(ctxts[i].Scale * float64(cnum))
	}

	var tmp1, tmp2 ckks.Ciphertext

	logStep := 0
	for i := step; i > 1; i /= 2 {
		logStep++
	}
	j := params.LogN() - logStep

	for step > 0 {
		for i := 0; i < step; i++ {

			// evaluator.multiply_plain(ctxts[i+step], idx[logStep], tmp1)
			// evaluator.add_inplace(tmp1, ctxts[i])

			// evaluator.multiply_plain(ctxts[i+step], idx[logStep], tmp2)
			// evaluator.sub_inplace(tmp2, ctxts[i])
			// evaluator.negate_inplace(tmp2)

			// evaluator.apply_galois_inplace(tmp2, (1<<j)+1, gal_keys)
			// evaluator.add(tmp1, tmp2, ctxts[i])

		}
		step /= 2
		logStep--
		j++
	}

	return ctxts[0]
}

// Perform FFT to encode N coefficients into N/2 complex values (that will be encoded with encode)
// Always assume that coeffs are of size N
func cencode(cfsEncoder ckks.EncoderBigComplex, coeffs []float64) (cvals []complex128) {
	slots := len(coeffs) / 2
	cvals_FFT := make([]*ring.Complex, slots)
	for i := 0; i < slots; i++ {
		cvals_FFT[i] = ring.NewComplex(ring.NewFloat(coeffs[i], 0), ring.NewFloat(coeffs[i+slots], 0))
	}
	cfsEncoder.FFT(cvals_FFT, slots)
	cvals = make([]complex128, slots)
	for i := 0; i < slots; i++ {
		cvals[i] = cvals_FFT[i].Float64()
	}

	return cvals
}

// Perform InvFFT to decode N coefficients from N/2 complex values (that would be dencoded from decode)
// Always assume that complex values are of size N/2
func cdecode(cfsEncoder ckks.EncoderBigComplex, cvals []complex128) (coeffs []float64) {
	slots := len(cvals)
	cvals_FFT := make([]*ring.Complex, slots)
	for i := 0; i < slots; i++ {
		cvals_FFT[i] = ring.NewComplex(ring.NewFloat(real(cvals[i]), 0), ring.NewFloat(imag(cvals[i]), 0))
	}
	cfsEncoder.InvFFT(cvals_FFT, slots)
	coeffs = make([]float64, 2*slots)
	for i := 0; i < slots; i++ {
		coeffs[i], _ = cvals_FFT[i].Real().Float64()
		coeffs[i+slots], _ = cvals_FFT[i].Imag().Float64()
	}

	return coeffs
}
